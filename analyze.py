#!/usr/bin/env python
"""Aggregate results over seeds and draw the report figures.

  python analyze.py --exp paper            # reads results/paper/seed*/...
  python analyze.py --exp paper --metric test_acc   # final-epoch instead of early-stop acc

Outputs (results/<exp>/analysis/):
  summary.csv / summary.md      mean ± std per (dataset, condition, round)
  fig_accuracy.(pdf|png)        accuracy vs. remaining weights, one panel per dataset
  fig_gain.(pdf|png)            accuracy gain over a random mask with the same layer-wise sparsity
  fig_overlap.(pdf|png)         layer-wise overlap between source and target masks (chance-normalised)
  fig_ablation_r<R>.(pdf|png)   layer-wise transfer ablations
  overlap.csv
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lth.masking import load_mask, overlap_stats  # noqa: E402

# Fixed condition -> colour/marker (categorical slots in fixed order, never cycled).
STYLE = {
    "native":          dict(c="#2a78d6", m="o", label="native ticket (own IMP mask, θ₀)"),
    "transfer":        dict(c="#eb6834", m="s", label="transferred mask (source IMP, θ₀)"),
    "transfer_reinit": dict(c="#1baf7a", m="^", label="transferred mask, re-initialised"),
    "random":          dict(c="#c98500", m="D", label="random mask, same layer sparsity"),
    "native_reinit":   dict(c="#e87ba4", m="v", label="native mask, re-initialised"),
    "native_sgd2":     dict(c="#6250d6", m="P", label="source IMP, other data order"),
}
ORDER = list(STYLE)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def load_results(exp_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "seed*", "results", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    if not rows:
        raise SystemExit(f"no results found under {exp_dir}")
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["seed", "dataset", "run_id"], keep="last")
    for col in ("extra", "layer_index", "layer"):
        if col not in df:
            df[col] = None
    df["extra"] = df["extra"].fillna("")
    return df


def summarise(df, metric):
    g = df.groupby(["dataset", "condition", "round", "extra"], dropna=False)
    s = g.agg(density=("density", "mean"), mean=(metric, "mean"), std=(metric, "std"),
              ece=("test_ece", "mean"), n=("seed", "nunique")).reset_index()
    s["std"] = s["std"].fillna(0.0)
    return s


def layer_labels(names, mask):
    out, nc, nf = {}, 0, 0
    for n in names:
        if mask[n].dim() == 4:
            nc += 1
            out[n] = f"conv{nc}"
        else:
            nf += 1
            out[n] = f"fc{nf}"
    return out


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(out_dir, name + ".pdf"))


def density_axis(ax):
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("weights remaining (%)")
    ticks = [100, 50, 20, 10, 5, 2, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def plot_lines(ax, s, dataset, conditions):
    sub = s[(s.dataset == dataset) & (s.extra == "")]
    dense = sub[(sub.condition == "native") & (sub["round"] == 0)]
    for cond in conditions:
        d = sub[sub.condition == cond].sort_values("round")
        if cond != "native_sgd2":
            d = pd.concat([dense, d[d["round"] > 0]]) if cond != "native" else d
        if d.empty:
            continue
        st = STYLE[cond]
        x, y, e = 100 * d.density.values, 100 * d["mean"].values, 100 * d["std"].values
        ax.plot(x, y, color=st["c"], marker=st["m"], ms=4, lw=1.6, label=st["label"])
        ax.fill_between(x, y - e, y + e, color=st["c"], alpha=0.15, lw=0)
    if not dense.empty:
        ax.axhline(100 * dense["mean"].iloc[0], color=INK2, ls="--", lw=0.9)
    density_axis(ax)


def fig_accuracy(s, source, targets, out_dir, metric_label):
    panels = [source] + targets
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.0), squeeze=False)
    for ax, ds in zip(axes[0], panels):
        conds = (["native", "native_sgd2", "native_reinit", "random"] if ds == source
                 else ["native", "transfer", "transfer_reinit", "native_reinit", "random"])
        plot_lines(ax, s, ds, conds)
        ax.set_title(f"{ds}" + (" (source)" if ds == source else f" (target, mask from {source})"))
        ax.set_ylabel(metric_label)
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    labels = [STYLE[c]["label"] for c in ORDER if STYLE[c]["label"] in handles]
    fig.legend([handles[l] for l in labels], labels, loc="lower center",
               ncol=min(3, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.text(0.995, 0.01, "dashed: dense network", ha="right", color=INK2, fontsize=7)
    save(fig, out_dir, "fig_accuracy")


def fig_gain(s, targets, out_dir):
    fig, axes = plt.subplots(1, len(targets), figsize=(3.3 * len(targets), 2.8), squeeze=False)
    for ax, ds in zip(axes[0], targets):
        sub = s[(s.dataset == ds) & (s.extra == "")]
        rnd = sub[sub.condition == "random"].set_index("round")
        for cond in ["native", "transfer", "transfer_reinit"]:
            d = sub[(sub.condition == cond) & (sub["round"].isin(rnd.index))].set_index("round").sort_index()
            if d.empty:
                continue
            gain = 100 * (d["mean"] - rnd.loc[d.index, "mean"])
            st = STYLE[cond]
            ax.plot(100 * d.density, gain, color=st["c"], marker=st["m"], ms=4, lw=1.6, label=st["label"])
        ax.axhline(0, color=INK2, lw=0.9)
        density_axis(ax)
        ax.set_title(ds)
        ax.set_ylabel("accuracy gain over random mask (pp)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    save(fig, out_dir, "fig_gain")


def compute_overlap(exp_dir, source, targets):
    rows = []
    for seed_dir in sorted(glob.glob(os.path.join(exp_dir, "seed*"))):
        seed = int(re.findall(r"seed(\d+)", seed_dir)[-1])
        mdir = os.path.join(seed_dir, "masks")
        comps = [(f"{source}-sgd2", f"{source} (other data order)")] + [(t, t) for t in targets]
        for tag, name in comps:
            for path in sorted(glob.glob(os.path.join(mdir, f"{tag}_r*.pt"))):
                r = int(path[-5:-3])
                spath = os.path.join(mdir, f"{source}_r{r:02d}.pt")
                if r == 0 or not os.path.exists(spath):
                    continue
                a, b = load_mask(spath), load_mask(path)
                labels = layer_labels(list(a), a)
                for n, st in overlap_stats(a, b).items():
                    rows.append({"seed": seed, "comparison": name, "round": r,
                                 "layer": labels[n], "layer_order": list(a).index(n), **st})
    return pd.DataFrame(rows)


def fig_overlap(ov, out_dir):
    comps = list(dict.fromkeys(ov.comparison))
    agg = ov.groupby(["comparison", "layer", "layer_order", "round"]).normalized.mean().reset_index()
    fig, axes = plt.subplots(1, len(comps), figsize=(3.0 * len(comps), 2.6), squeeze=False)
    im = None
    for ax, comp in zip(axes[0], comps):
        d = agg[agg.comparison == comp]
        piv = d.pivot_table(index=["layer_order", "layer"], columns="round", values="normalized")
        im = ax.imshow(piv.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([l for _, l in piv.index])
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, fontsize=6)
        ax.set_xlabel("IMP round")
        ax.set_title(f"source vs {comp}", fontsize=9)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.9, pad=0.02)
    cb.set_label("mask overlap above chance\n0 = random, 1 = identical", fontsize=7)
    save(fig, out_dir, "fig_overlap")


def fig_ablation(df, s, targets, r, out_dir, metric, labels):
    sub = df[(df["round"] == r) & df.dataset.isin(targets)]
    if sub[sub.condition.str.startswith("ablate")].empty:
        return
    fig, axes = plt.subplots(2, len(targets), figsize=(3.3 * len(targets), 5.0), squeeze=False)
    for k, ds in enumerate(targets):
        d = sub[sub.dataset == ds]
        base = d[d.condition == "random"][metric]
        full = d[d.condition == "transfer"][metric]
        L = len(labels)
        # prefix: x = number of leading layers taking the source mask
        pre = d[d.condition == "ablate_prefix"].groupby("layer_index")[metric].agg(["mean", "std"])
        xs, ys, es = [0], [base.mean()], [base.std()]
        for j in range(1, L):
            if j in pre.index:
                xs.append(j); ys.append(pre.loc[j, "mean"]); es.append(pre.loc[j, "std"])
        xs.append(L); ys.append(full.mean()); es.append(full.std())
        ys, es = 100 * np.array(ys), 100 * np.nan_to_num(np.array(es, dtype=float))
        ax = axes[0][k]
        ax.plot(xs, ys, color=STYLE["transfer"]["c"], marker="s", ms=4, lw=1.6)
        ax.fill_between(xs, ys - es, ys + es, color=STYLE["transfer"]["c"], alpha=0.15, lw=0)
        ax.set_xticks(range(L + 1))
        ax.set_xticklabels(["none"] + [f"≤{labels[i]}" for i in range(L)], rotation=45, fontsize=7)
        ax.set_title(f"{ds}: source mask on first layers", fontsize=9)
        ax.set_ylabel("test acc (%)")
        # single layer: gain over random
        sg = d[d.condition == "ablate_single"].groupby("layer_index")[metric].agg(["mean", "std"])
        ax = axes[1][k]
        gains = 100 * (sg["mean"] - base.mean())
        ax.bar(range(len(sg)), gains.values, width=0.6, color=STYLE["transfer"]["c"],
               yerr=100 * sg["std"].fillna(0).values, error_kw=dict(lw=0.8, ecolor=INK2))
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_xticks(range(len(sg)))
        ax.set_xticklabels([labels[int(i)] for i in sg.index], fontsize=7)
        ax.set_title(f"{ds}: source mask on one layer only", fontsize=9)
        ax.set_ylabel("gain over random mask (pp)")
    dens = s[(s.dataset == targets[0]) & (s.condition == "transfer") & (s["round"] == r)].density
    note = f"IMP round {r}" + (f" ({100 * dens.iloc[0]:.1f}% weights remaining)" if len(dens) else "")
    fig.suptitle(note, fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, out_dir, f"fig_ablation_r{r:02d}")


def write_tables(s, out_dir):
    s.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    main = s[s.extra == ""].copy()
    main["cell"] = [f"{100 * m:.2f} ± {100 * e:.2f}" for m, e in zip(main["mean"], main["std"])]
    main["weights %"] = (100 * main.density).round(2)
    lines = []
    for ds, d in main.groupby("dataset"):
        piv = d.pivot_table(index=["round", "weights %"], columns="condition", values="cell", aggfunc="first")
        piv = piv[[c for c in ORDER if c in piv.columns]]
        try:
            table = piv.to_markdown()
        except ImportError:  # tabulate not installed
            table = "```\n" + piv.to_string() + "\n```"
        lines += [f"### {ds}", "", table, ""]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote", os.path.join(out_dir, "summary.md"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="paper")
    p.add_argument("--out", default="results")
    p.add_argument("--metric", default="test_acc_at_best_val",
                   choices=["test_acc_at_best_val", "test_acc", "test_nll", "test_ece"])
    args = p.parse_args()
    exp_dir = os.path.join(args.out, args.exp)
    out_dir = os.path.join(exp_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json")) as f:
        cfg = json.load(f)
    source = cfg["source"]
    df = load_results(exp_dir)
    present = [d for d in dict.fromkeys(df.dataset) if d != source]
    targets = [t for t in cfg.get("targets", []) if t in present]
    targets += [t for t in present if t not in targets]
    s = summarise(df, args.metric)
    write_tables(s, out_dir)

    label = {"test_acc_at_best_val": "test acc. at early stop (%)", "test_acc": "final test acc. (%)",
             "test_nll": "test NLL (×100)", "test_ece": "test ECE (×100)"}[args.metric]
    fig_accuracy(s, source, targets, out_dir, label)
    if targets:
        fig_gain(s, targets, out_dir)
    ov = compute_overlap(exp_dir, source, targets)
    if not ov.empty:
        ov.to_csv(os.path.join(out_dir, "overlap.csv"), index=False)
        fig_overlap(ov, out_dir)
    if targets:
        any_mask = sorted(glob.glob(os.path.join(exp_dir, "seed*", "masks", f"{source}_r00.pt")))[0]
        m = load_mask(any_mask)
        labels = list(layer_labels(list(m), m).values())
        for r in sorted(df[df.condition.str.startswith("ablate")]["round"].unique()):
            fig_ablation(df, s, targets, int(r), out_dir, args.metric, labels)


if __name__ == "__main__":
    main()
