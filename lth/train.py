import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, data, split, amp, n_bins=15):
    model.eval()
    probs, labels = [], []
    for x, y in data.batches(split, 2000):
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        probs.append(F.softmax(logits.float(), 1))
        labels.append(y)
    p, y = torch.cat(probs), torch.cat(labels)
    conf, pred = p.max(1)
    correct = (pred == y).float()
    nll = -torch.log(p[torch.arange(len(y)), y].clamp_min(1e-12)).mean().item()
    ece = 0.0
    edges = torch.linspace(0, 1, n_bins + 1, device=p.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.float().mean().item() * abs(conf[sel].mean().item() - correct[sel].mean().item())
    model.train()
    return {"acc": correct.mean().item(), "nll": nll, "ece": ece}


def train_model(model, applier, data, cfg, data_seed):
    """Train with a fixed mask; return metrics incl. early-stopping (best-val) test acc."""
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay,
                              nesterov=True)
    steps_per_epoch = data.size("train") // cfg.batch_size
    total = steps_per_epoch * cfg.epochs
    sched = None
    if cfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    best = {"val_acc": -1.0}
    curve = []
    model.train()
    for ep in range(cfg.epochs):
        for x, y in data.batches("train", cfg.batch_size, shuffle=True, seed=data_seed * 1000 + ep):
            with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=cfg.amp):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            applier.apply()
        if (ep + 1) % cfg.eval_every == 0 or ep + 1 == cfg.epochs:
            v = evaluate(model, data, "val", cfg.amp)
            t = evaluate(model, data, "test", cfg.amp)
            curve.append({"epoch": ep + 1, "val_acc": v["acc"], "test_acc": t["acc"]})
            if v["acc"] > best["val_acc"]:
                best = {"val_acc": v["acc"], "test_acc": t["acc"], "epoch": ep + 1}
    final = evaluate(model, data, "test", cfg.amp)
    return {
        "test_acc": final["acc"], "test_nll": final["nll"], "test_ece": final["ece"],
        "best_val_acc": best["val_acc"], "test_acc_at_best_val": best["test_acc"],
        "best_epoch": best["epoch"], "curve": curve,
    }
