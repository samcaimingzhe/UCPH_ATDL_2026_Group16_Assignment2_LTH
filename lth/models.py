"""Architectures from Frankle & Carbin (2019): Conv-2/4/6 and LeNet-300-100.

Every model has the same layout: ``features`` -> ``classifier`` -> ``head``.
The *body* (features + classifier) is shared across datasets so that a pruning
mask found on one dataset can be applied verbatim on another. Only ``head``
depends on the number of classes; it is never pruned and never transferred.
"""
import torch
import torch.nn as nn

CONV_CFG = {
    "conv2": [64, 64, "M"],
    "conv4": [64, 64, "M", 128, 128, "M"],
    "conv6": [64, 64, "M", 128, 128, "M", 256, 256, "M"],
}


class Net(nn.Module):
    def __init__(self, arch: str, num_classes: int, in_ch: int = 3, img: int = 32):
        super().__init__()
        self.arch = arch
        if arch in CONV_CFG:
            layers, c, s = [], in_ch, img
            for v in CONV_CFG[arch]:
                if v == "M":
                    layers.append(nn.MaxPool2d(2))
                    s //= 2
                else:
                    layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]
                    c = v
            layers.append(nn.Flatten())
            self.features = nn.Sequential(*layers)
            dims = [c * s * s, 256, 256]
        elif arch == "lenet":
            self.features = nn.Flatten()
            dims = [in_ch * img * img, 300, 100]
        else:
            raise ValueError(f"unknown arch {arch}")
        fcs = []
        for a, b in zip(dims[:-1], dims[1:]):
            fcs += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        self.classifier = nn.Sequential(*fcs)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        return self.head(self.classifier(self.features(x)))

    # ---- pruning bookkeeping -------------------------------------------------
    def prunable_names(self):
        """Weight tensors of the body, in forward order (biases are never pruned)."""
        return [n for n, p in self.named_parameters()
                if n.endswith("weight") and not n.startswith("head.")]

    @staticmethod
    def layer_kind(param: torch.Tensor) -> str:
        return "conv" if param.dim() == 4 else "fc"

    def body_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith("head.")}


def _glorot_(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)


def build_model(arch: str, num_classes: int, body_seed: int, head_seed: int,
                in_ch: int = 3, img: int = 32) -> Net:
    """Body init depends only on (arch, body_seed); head init only on head_seed.

    Hence models for datasets with different numbers of classes share an
    identical body initialisation theta_0 when given the same body_seed.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(body_seed)
        model = Net(arch, num_classes, in_ch, img)
        _glorot_(model.features)
        _glorot_(model.classifier)
        torch.manual_seed(head_seed)
        _glorot_(model.head)
    return model
