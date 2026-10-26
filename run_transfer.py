#!/usr/bin/env python
"""Task 1: Do winning-ticket masks transfer across datasets?

For every seed s:
  theta_0   <- Glorot init of the body (shared by all datasets for this seed)
  SOURCE    IMP on the source dataset           -> masks  m^S_r  (r = 0..R)
            controls on the source: native_reinit, random (Frankle & Carbin)
            optional SGD-noise control: IMP again with another data order
  TARGET t  IMP on t from the same theta_0      -> masks  m^t_r  (native tickets)
            for each round r, train on t with
              transfer         m^S_r + theta_0
              transfer_reinit  m^S_r + theta'      (mask without the matching init)
              random           random mask, same per-layer counts + theta_0
              native_reinit    m^t_r + theta'
            layer-wise ablations at selected rounds (source mask on a subset of
            layers, random mask on the rest)

Everything is resumable: completed runs are logged in results/<dataset>.jsonl and
skipped on restart; masks are saved after every IMP round.

Examples
  python run_transfer.py --preset smoke                      # CPU, 1 minute, fake data
  python run_transfer.py --preset paper --seeds 0 --download
  python run_transfer.py --preset paper --seeds 0 --phase source
  python run_transfer.py --preset paper --seeds 0 --phase target --targets svhn
"""
import argparse
import json
import os
import sys

import torch

from lth.data import GPUData
from lth.masking import (MaskApplier, combine, density, layer_density, load_mask,
                         magnitude_prune, ones_mask, random_mask_like, save_mask)
from lth.models import build_model
from lth.train import train_model
from lth.utils import JsonlStore, Timer, set_seed, stable_seed

PRESETS = {
    # Tiny end-to-end check on synthetic data; runs on CPU in about a minute.
    "smoke": dict(arch="conv2", source="fake10", targets=["fake5"], rounds=3, epochs=2,
                  batch_size=64, seeds=[0], ablation_rounds=[2], eval_every=1,
                  val_size=300, exp="smoke"),
    # A few hours on one GPU: enough for a first look at all the figures.
    "quick": dict(arch="conv4", source="cifar10", targets=["svhn", "cifar100"], rounds=10,
                  rate_conv=0.15, rate_fc=0.25, epochs=12, batch_size=128, lr=6e-4,
                  seeds=[0], ablation_rounds=[5, 10], exp="quick"),
    # Report setting: Conv-4 hyper-parameters of Frankle & Carbin (Adam 3e-4, batch 60,
    # conv 10% / fc 20% per round), 3 seeds, three target datasets.
    "paper": dict(arch="conv4", source="cifar10", targets=["svhn", "cifar100", "fashionmnist"],
                  rounds=15, rate_conv=0.10, rate_fc=0.20, epochs=25, batch_size=60, lr=3e-4,
                  seeds=[0, 1, 2], ablation_rounds=[5, 10, 15], exp="paper"),
}

CONFIG_KEYS = ["arch", "source", "rounds", "rate_conv", "rate_fc", "epochs", "batch_size", "lr",
               "optimizer", "weight_decay", "lr_schedule", "val_size", "train_subset"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default=None)
    p.add_argument("--exp", default="default", help="experiment name (output sub-folder)")
    p.add_argument("--out", default="results")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--download", action="store_true", help="download datasets via torchvision")
    p.add_argument("--device", default="auto")
    p.add_argument("--phase", choices=["all", "source", "target"], default="all")
    # setup
    p.add_argument("--arch", default="conv4", choices=["conv2", "conv4", "conv6", "lenet"])
    p.add_argument("--source", default="cifar10")
    p.add_argument("--targets", nargs="+", default=["svhn", "cifar100"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    # pruning
    p.add_argument("--rounds", type=int, default=15, help="number of IMP pruning rounds R")
    p.add_argument("--rate_conv", type=float, default=0.10)
    p.add_argument("--rate_fc", type=float, default=0.20)
    # training
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--train_subset", type=int, default=None, help="limit #training images")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA)")
    # experiment design
    p.add_argument("--conditions", nargs="+",
                   default=["transfer", "transfer_reinit", "random", "native_reinit"],
                   choices=["transfer", "transfer_reinit", "random", "native_reinit"])
    p.add_argument("--cond_rounds", nargs="+", type=int, default=None,
                   help="rounds at which control conditions are trained (default: all 1..R)")
    p.add_argument("--ablation_rounds", nargs="+", type=int, default=[])
    p.add_argument("--ablation_modes", nargs="+", default=["prefix", "single"],
                   choices=["prefix", "single"])
    p.add_argument("--no_sgd_control", action="store_true",
                   help="skip the second source IMP run with a different data order")
    p.add_argument("--force_config", action="store_true",
                   help="allow resuming an experiment folder written with different settings")

    pre, _ = p.parse_known_args(argv)
    if pre.preset:
        p.set_defaults(**PRESETS[pre.preset])
    return p.parse_args(argv)


def log(*a):
    print(*a, flush=True)


class Experiment:
    def __init__(self, args, seed, device):
        self.a, self.seed, self.device = args, seed, device
        self.dir = os.path.join(args.out, args.exp, f"seed{seed}")
        self.mask_dir = os.path.join(self.dir, "masks")
        os.makedirs(self.mask_dir, exist_ok=True)
        self.theta0_seed = stable_seed("theta0", seed)
        self.reinit_seed = stable_seed("reinit", seed)
        self._data = {}

    # ---- helpers --------------------------------------------------------------
    def data(self, name):
        if name not in self._data:
            a = self.a
            self._data[name] = GPUData(name, a.data_dir, self.device, a.val_size,
                                       a.train_subset, a.download)
        return self._data[name]

    def mask_path(self, tag, r):
        return os.path.join(self.mask_dir, f"{tag}_r{r:02d}.pt")

    def store(self, name):
        return JsonlStore(os.path.join(self.dir, "results", f"{name}.jsonl"))

    def masks_complete(self, tag):
        return all(os.path.exists(self.mask_path(tag, r)) for r in range(self.a.rounds + 1))

    def train(self, ds, mask, body_seed, store, run_id, meta, data_seed=None):
        a = self.a
        set_seed(stable_seed(run_id, self.seed))
        model = build_model(a.arch, ds.num_classes, body_seed,
                            stable_seed("head", self.seed, ds.name)).to(self.device)
        applier = MaskApplier(model, mask, self.device)
        with Timer() as t:
            res = train_model(model, applier, ds, a, self.seed if data_seed is None else data_seed)
        rec = {"run_id": run_id, "seed": self.seed, "arch": a.arch, "dataset": ds.name,
               "source": a.source, **meta, "density": density(mask),
               "layer_density": layer_density(mask), **res, "seconds": round(t.elapsed, 1)}
        store.add(rec)
        log(f"[seed {self.seed}] {ds.name:>12s} {meta['condition']:<16s} r={meta['round']:>2d} "
            f"{meta.get('extra', ''):<12s} dens={100 * rec['density']:6.2f}%  "
            f"test={100 * res['test_acc']:5.2f}  es-test={100 * res['test_acc_at_best_val']:5.2f}  "
            f"({t.elapsed:.0f}s)")
        return model

    # ---- IMP ------------------------------------------------------------------
    def imp(self, ds, tag, condition, data_seed):
        a, store = self.a, self.store(ds.name)
        if not os.path.exists(self.mask_path(tag, 0)):
            ref = build_model(a.arch, ds.num_classes, 0, 0)
            save_mask(ones_mask(ref), self.mask_path(tag, 0))
        for r in range(a.rounds + 1):
            run_id = f"{condition}|r{r}"
            has_next = r == a.rounds or os.path.exists(self.mask_path(tag, r + 1))
            if store.has(run_id) and has_next:
                continue
            mask = load_mask(self.mask_path(tag, r))
            model = self.train(ds, mask, self.theta0_seed, store, run_id,
                               {"condition": condition, "round": r, "mask_tag": tag}, data_seed)
            if r < a.rounds:
                save_mask(magnitude_prune(model, mask, a.rate_conv, a.rate_fc),
                          self.mask_path(tag, r + 1))

    # ---- phases ---------------------------------------------------------------
    def run_source(self):
        a = self.a
        ds = self.data(a.source)
        log(f"== seed {self.seed}: IMP on source {a.source}")
        self.imp(ds, a.source, "native", self.seed)
        if not a.no_sgd_control:
            log(f"== seed {self.seed}: SGD-noise control (same theta_0, different data order)")
            self.imp(ds, f"{a.source}-sgd2", "native_sgd2", self.seed + 1000)
        store = self.store(ds.name)
        for r in self.cond_rounds():
            src = load_mask(self.mask_path(a.source, r))
            if "native_reinit" in a.conditions:
                self.cond(ds, store, "native_reinit", r, src, self.reinit_seed)
            if "random" in a.conditions:
                self.cond(ds, store, "random", r, self.random_mask(ds.name, r, src), self.theta0_seed)

    def run_target(self, target):
        a = self.a
        if not self.masks_complete(a.source):
            sys.exit(f"source masks for seed {self.seed} are incomplete; run --phase source first")
        ds = self.data(target)
        log(f"== seed {self.seed}: native IMP on target {target}")
        self.imp(ds, target, "native", self.seed)
        store = self.store(ds.name)
        for r in self.cond_rounds():
            src = load_mask(self.mask_path(a.source, r))
            rnd = self.random_mask(target, r, src)
            if "transfer" in a.conditions:
                self.cond(ds, store, "transfer", r, src, self.theta0_seed)
            if "transfer_reinit" in a.conditions:
                self.cond(ds, store, "transfer_reinit", r, src, self.reinit_seed)
            if "random" in a.conditions:
                self.cond(ds, store, "random", r, rnd, self.theta0_seed)
            if "native_reinit" in a.conditions:
                self.cond(ds, store, "native_reinit", r,
                          load_mask(self.mask_path(target, r)), self.reinit_seed)
        for r in a.ablation_rounds:
            if not 1 <= r <= a.rounds:
                continue
            src = load_mask(self.mask_path(a.source, r))
            rnd = self.random_mask(target, r, src)
            layers = list(src.keys())
            if "prefix" in a.ablation_modes:
                for j in range(1, len(layers)):
                    self.cond(ds, store, "ablate_prefix", r, combine(src, rnd, set(layers[:j])),
                              self.theta0_seed, extra=f"j{j}", layer_index=j)
            if "single" in a.ablation_modes:
                for j, name in enumerate(layers):
                    self.cond(ds, store, "ablate_single", r, combine(src, rnd, {name}),
                              self.theta0_seed, extra=f"l{j}", layer_index=j, layer=name)

    def cond_rounds(self):
        rs = self.a.cond_rounds or range(1, self.a.rounds + 1)
        return [r for r in rs if 1 <= r <= self.a.rounds]

    def random_mask(self, dataset, r, like):
        return random_mask_like(like, stable_seed("random", self.seed, dataset, r))

    def cond(self, ds, store, condition, r, mask, body_seed, extra="", **meta):
        run_id = f"{condition}|r{r}" + (f"|{extra}" if extra else "")
        if store.has(run_id):
            return
        self.train(ds, mask, body_seed, store, run_id,
                   {"condition": condition, "round": r, "extra": extra, **meta})


def check_config(args):
    path = os.path.join(args.out, args.exp, "config.json")
    cfg = {k: getattr(args, k) for k in CONFIG_KEYS}
    targets = list(args.targets) if args.phase != "source" else []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if diff and not args.force_config:
            sys.exit(f"{path} was written with different settings {diff}; "
                     f"use another --exp or pass --force_config")
        # parallel processes each handle a subset of targets: keep the union
        targets = list(dict.fromkeys(old.get("targets", []) + targets))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({**cfg, "targets": targets, "conditions": args.conditions,
                   "ablation_rounds": args.ablation_rounds}, f, indent=2)
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        args.amp = False
    check_config(args)
    log(f"device={device}  arch={args.arch}  source={args.source}  targets={args.targets}  "
        f"seeds={args.seeds}  rounds={args.rounds}  epochs={args.epochs}")
    for seed in args.seeds:
        exp = Experiment(args, seed, device)
        if args.phase in ("all", "source"):
            exp.run_source()
        if args.phase in ("all", "target"):
            for t in args.targets:
                exp.run_target(t)
    log("done.")


if __name__ == "__main__":
    main()
