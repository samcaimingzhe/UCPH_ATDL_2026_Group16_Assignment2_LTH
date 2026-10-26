"""Masks are dicts {param_name: bool tensor (CPU)} over the prunable body weights."""
import numpy as np
import torch

from .utils import atomic_torch_save


def ones_mask(model):
    params = dict(model.named_parameters())
    return {n: torch.ones_like(params[n], dtype=torch.bool, device="cpu")
            for n in model.prunable_names()}


def magnitude_prune(model, mask, rate_conv: float, rate_fc: float):
    """Layer-wise IMP step: remove `rate` of the *remaining* weights of each layer
    with the smallest |w| (Frankle & Carbin, 2019)."""
    params = dict(model.named_parameters())
    new = {}
    for n, m in mask.items():
        p = params[n].detach().float().cpu()
        rate = rate_conv if model.layer_kind(p) == "conv" else rate_fc
        remaining = int(m.sum())
        k = int(round(remaining * rate))
        flat_m = m.flatten().clone()
        if k > 0:
            scores = p.abs().flatten().masked_fill(~flat_m, float("inf"))
            idx = torch.topk(scores, k, largest=False).indices
            flat_m[idx] = False
        new[n] = flat_m.view_as(m)
    return new


def random_mask_like(mask, seed: int):
    """Uniformly random mask with exactly the same number of kept weights per layer."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n, m in mask.items():
        k = int(m.sum())
        flat = torch.zeros(m.numel(), dtype=torch.bool)
        flat[torch.randperm(m.numel(), generator=g)[:k]] = True
        out[n] = flat.view_as(m)
    return out


def combine(src_mask, other_mask, use_src: set):
    """Layers in `use_src` come from src_mask, the rest from other_mask."""
    return {n: (src_mask[n] if n in use_src else other_mask[n]).clone() for n in src_mask}


def density(mask):
    kept = sum(int(m.sum()) for m in mask.values())
    total = sum(m.numel() for m in mask.values())
    return kept / total


def layer_density(mask):
    return {n: int(m.sum()) / m.numel() for n, m in mask.items()}


def overlap_stats(a, b):
    """Per-layer overlap of two masks with equal per-layer counts.

    overlap = |A∩B| / |A|; chance level for random masks = density d.
    normalized = (overlap - d) / (1 - d): 0 = chance, 1 = identical.
    """
    out = {}
    for n in a:
        ka, kb = int(a[n].sum()), int(b[n].sum())
        d = ka / a[n].numel()
        inter = int((a[n] & b[n]).sum())
        ov = inter / max(ka, 1)
        out[n] = {"density": d, "overlap": ov,
                  "normalized": (ov - d) / (1 - d) if d < 1 else float("nan"),
                  "jaccard": inter / max(ka + kb - inter, 1)}
    return out


# ---- compact storage (bit-packed) -------------------------------------------
def save_mask(mask, path: str) -> None:
    packed = {n: (tuple(m.shape), np.packbits(m.numpy().ravel())) for n, m in mask.items()}
    atomic_torch_save(packed, path)


def load_mask(path: str):
    packed = torch.load(path, weights_only=False)
    out = {}
    for n, (shape, bits) in packed.items():
        numel = int(np.prod(shape))
        out[n] = torch.from_numpy(np.unpackbits(bits)[:numel].astype(bool)).view(shape)
    return out


class MaskApplier:
    """Keeps pruned weights at exactly zero during training.

    Pruned weights are zeroed once, their gradients are masked (so Adam/SGD
    moments stay zero) and they are re-zeroed after every optimizer step.
    """

    def __init__(self, model, mask, device):
        params = dict(model.named_parameters())
        self.pairs = []
        for n, m in mask.items():
            p = params[n]
            md = m.to(device=device, dtype=p.dtype)
            self.pairs.append((p, md))
            p.register_hook(lambda g, md=md: g * md)
        self.apply()

    @torch.no_grad()
    def apply(self):
        for p, md in self.pairs:
            p.mul_(md)
