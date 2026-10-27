# %% [markdown]
# # Dropout, iterative pruning, and ticket transfer in convolutional networks
#
# This notebook separates dropout used to **find a pruning mask** from dropout
# used to **train a masked ticket**. It runs native experiments on CIFAR-10,
# CIFAR-100, and SVHN, followed by CIFAR-10 mask transfer to CIFAR-100 and SVHN.
# The architecture follows Conv-4 in Frankle & Carbin (2019):
# [paper](https://arxiv.org/abs/1803.03635). The paper's Figure 2 uses 25,000
# iterations for Conv-4 and 20,000 for Conv-2; Figure 6 uses a lower learning
# rate with dropout and Appendix H.5 trains dropout models three times longer.
# This experiment defaults to equal iteration budgets and one common learning
# rate to isolate dropout roles; condition-specific learning rates are configurable.
#
# | Condition | Mask-search dropout | Ticket-training dropout |
# |---|---:|---:|
# | A | off | off |
# | B | on | on |
# | C | on | off |
# | D | off | on |
#
# The definitions imply A/D share masks and B/C share masks. The code checks
# these pairings. Transfer reuses the source body mask and original body weights;
# its target classifier is newly initialized with a fixed seed and kept dense.
# A dense-head native reference is included for fair body-mask sparsity comparisons.
# No augmentation is used. SVHN uses only its standard `train` and `test` splits.

# %% [markdown]
# ## Configuration and reproducibility
#
# Set `quick=True` for a short pipeline check before starting the full experiment.
# Full defaults are intentionally substantial: 3 datasets × 3 seeds × 15 rounds,
# two independent mask-search chains, four factorial conditions, controls, and
# transfer runs. The notebook times a pilot fit and estimates the remaining work.
# Results/checkpoints are written under `output_dir`; rerunning resumes completed
# fits. `synthetic_smoke=True` substitutes generated images only to validate code;
# synthetic metrics are never presented as dataset results.

# %%
import copy, hashlib, json, math, os, random, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler
from sklearn.metrics import accuracy_score

try:
    shell = get_ipython()
except NameError:
    shell = None
if shell is not None:
    shell.run_line_magic("matplotlib", "inline")

CONFIG = {
    "quick": False,
    "synthetic_smoke": False,
    "architecture": "conv4",           # conv4 or conv2
    "datasets": ["cifar10", "cifar100", "svhn"],
    "seeds": [0, 1, 2],
    "dropout_p": 0.5,                   # 0.2 and 0.5 are supported
    "conv_dropout": False,
    "iterations": None,                 # None: 25000 (Conv-4), 20000 (Conv-2)
    "batch_size": 60,
    "eval_every": 500,
    "lr_by_condition": {"A": None, "B": None, "C": None, "D": None},
    "rounds": 15,
    "conv_prune_fraction": 0.10,
    "fc_prune_fraction": 0.20,
    "head_prune_fraction": 0.10,
    "controls": True,
    "transfer": True,
    "validation_size": 5000,
    "data_dir": "data",
    "output_dir": "results_dropout_cnn",
    "download": True,
    "num_workers": 2,
    "pin_memory": True,
    "amp": False,
    "checkpoint_every": 2000,
    "benchmark_iterations": 200,
    "train_limit": None,
    "validation_limit": None,
    "test_limit": None,
    "quick_settings": {
        "architecture": "conv2", "seeds": [0], "rounds": 1,
        "iterations": 4, "eval_every": 2, "batch_size": 4,
        "validation_size": 20, "num_workers": 0, "controls": False,
        "transfer": True, "checkpoint_every": 0,
        "train_limit": 40, "validation_limit": 12, "test_limit": 12,
        "benchmark_iterations": 2,
    },
}

if CONFIG["quick"]:
    CONFIG.update(copy.deepcopy(CONFIG["quick_settings"]))
    CONFIG["output_dir"] += "_quick"
if CONFIG["synthetic_smoke"]:
    CONFIG["output_dir"] += "_synthetic"
    CONFIG["data_dir"] = ""
if CONFIG["iterations"] is None:
    CONFIG["iterations"] = 25000 if CONFIG["architecture"] == "conv4" else 20000
if CONFIG["architecture"] == "conv2" and not CONFIG["quick"]:
    CONFIG["iterations"] = 20000 if CONFIG["iterations"] is None else CONFIG["iterations"]
assert CONFIG["architecture"] in {"conv2", "conv4"}
assert 0 < CONFIG["dropout_p"] < 1
assert CONFIG["datasets"] and CONFIG["seeds"]
assert set(CONFIG["lr_by_condition"]) == {"A", "B", "C", "D"}
BASE_LR = 3e-4 if CONFIG["architecture"] == "conv4" else 2e-4
for _condition in "ABCD":
    if CONFIG["lr_by_condition"][_condition] is None:
        CONFIG["lr_by_condition"][_condition] = BASE_LR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
OUT = Path(CONFIG["output_dir"]); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "masks").mkdir(exist_ok=True); (OUT / "checkpoints").mkdir(exist_ok=True)
(OUT / "config.json").write_text(json.dumps(CONFIG, indent=2))
print(f"Device: {DEVICE}; architecture: {CONFIG['architecture']}; iterations/fit: {CONFIG['iterations']}")

# %% [markdown]
# ## Data loading and preprocessing
#
# Each dataset uses its own training-set channel statistics. The validation
# holdout is selected from the official training split with the run seed.
# Quick limits apply after the holdout is formed. For exact SVHN class labels,
# torchvision already maps digit 0 to class 0. The synthetic path exists only
# for pipeline checks and requires no torchvision installation.

# %%
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _raw_torchvision(name, data_dir, download):
    from torchvision.datasets import CIFAR10, CIFAR100, SVHN
    root = str(data_dir)
    if name == "cifar10":
        tr = CIFAR10(root, train=True, download=download)
        te = CIFAR10(root, train=False, download=download)
        return tr.data, np.asarray(tr.targets), te.data, np.asarray(te.targets)
    if name == "cifar100":
        tr = CIFAR100(root, train=True, download=download)
        te = CIFAR100(root, train=False, download=download)
        return tr.data, np.asarray(tr.targets), te.data, np.asarray(te.targets)
    if name == "svhn":
        tr = SVHN(root, split="train", download=download)
        te = SVHN(root, split="test", download=download)
        return tr.data.transpose(0, 2, 3, 1), np.asarray(tr.labels), te.data.transpose(0, 2, 3, 1), np.asarray(te.labels)
    raise ValueError(name)

def _synthetic_raw(name, seed):
    nclass = 100 if name == "cifar100" else 10
    rng = np.random.default_rng(seed + {"cifar10": 101, "cifar100": 202, "svhn": 303}[name])
    ntrain = max(CONFIG["validation_size"] + 40, 80)
    ntest = max(40, 12)
    xtr = rng.integers(0, 256, size=(ntrain, 32, 32, 3), dtype=np.uint8)
    xte = rng.integers(0, 256, size=(ntest, 32, 32, 3), dtype=np.uint8)
    return xtr, rng.integers(0, nclass, ntrain), xte, rng.integers(0, nclass, ntest)

def _limit_pair(x, y, limit):
    if limit is None or len(x) <= limit: return x, y
    return x[:limit], y[:limit]

def _to_nchw(x):
    x = torch.as_tensor(np.asarray(x).copy(), dtype=torch.uint8)
    if x.ndim == 4 and x.shape[-1] == 3: x = x.permute(0, 3, 1, 2).contiguous()
    return x

def load_dataset(name, seed):
    if CONFIG["synthetic_smoke"]:
        xtr, ytr, xte, yte = _synthetic_raw(name, seed)
    else:
        xtr, ytr, xte, yte = _raw_torchvision(name, CONFIG["data_dir"], CONFIG["download"])
    ytr = torch.as_tensor(np.asarray(ytr).copy(), dtype=torch.long)
    yte = torch.as_tensor(np.asarray(yte).copy(), dtype=torch.long)
    perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed))
    nv = CONFIG["validation_size"]
    if len(ytr) <= nv: raise ValueError("Training split must exceed validation_size")
    ival, itr = perm[:nv], perm[nv:]
    xall = _to_nchw(xtr)
    xtrain, ytrain = xall[itr], ytr[itr]
    xval, yval = xall[ival], ytr[ival]
    xtest = _to_nchw(xte)
    xtrain, ytrain = _limit_pair(xtrain, ytrain, CONFIG["train_limit"])
    xval, yval = _limit_pair(xval, yval, CONFIG["validation_limit"])
    xtest, yte = _limit_pair(xtest, yte, CONFIG["test_limit"])
    # Accumulate moments over training images only without a full float64 copy.
    sums=torch.zeros(3,dtype=torch.float64); squares=torch.zeros(3,dtype=torch.float64); count=0
    for chunk in xtrain.split(512):
        z=chunk.to(torch.float64).div_(255.0)
        sums += z.sum(dim=(0,2,3)); squares += z.square().sum(dim=(0,2,3))
        count += z.shape[0]*z.shape[2]*z.shape[3]
    mean=(sums/count).to(torch.float32)
    std=(squares/count-(sums/count).square()).clamp_min(1e-12).sqrt().to(torch.float32)
    pack = {"name": name, "seed": seed, "mean": mean, "std": std,
            "train_x": xtrain, "train_y": ytrain, "val_x": xval, "val_y": yval,
            "test_x": xtest, "test_y": yte,
            "n_classes": 100 if name == "cifar100" else 10}
    split_path = OUT / f"split_{name}_seed{seed}.json"
    split_path.write_text(json.dumps({"dataset": name, "seed": seed, "n_train": len(xtrain),
        "n_validation": len(xval), "n_test": len(xtest), "mean": mean.tolist(), "std": std.tolist()}))
    return pack

class EpochOffsetSampler(Sampler):
    """Deterministic shuffled epochs that can resume from a consumed batch index."""
    def __init__(self, n, seed): self.n, self.seed, self.epoch, self.offset = n, seed, 0, 0
    def set_position(self, epoch, offset): self.epoch, self.offset = epoch, offset
    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.offset:].tolist())
    def __len__(self): return max(0, self.n - self.offset)

def make_loaders(data, seed):
    pin = bool(CONFIG["pin_memory"] and DEVICE.type == "cuda")
    def fixed_loader(x, y, shuffle=False, offset_sampler=None):
        ds = TensorDataset(x, y)
        if offset_sampler is not None:
            return DataLoader(ds, batch_size=CONFIG["batch_size"], sampler=offset_sampler,
                generator=torch.Generator().manual_seed(seed + 7001),
                num_workers=CONFIG["num_workers"], pin_memory=pin, persistent_workers=CONFIG["num_workers"] > 0)
        return DataLoader(ds, batch_size=CONFIG["eval_batch_size"] if "eval_batch_size" in CONFIG else 512,
            shuffle=shuffle, generator=torch.Generator().manual_seed(seed + 9001), num_workers=CONFIG["num_workers"],
            pin_memory=pin, persistent_workers=CONFIG["num_workers"] > 0)
    sampler = EpochOffsetSampler(len(data["train_y"]), seed + 17)
    return fixed_loader(data["train_x"], data["train_y"], offset_sampler=sampler), \
        fixed_loader(data["val_x"], data["val_y"]), fixed_loader(data["test_x"], data["test_y"]), sampler

# %% [markdown]
# ## Conv-2 / Conv-4 model, initialization, and masks
#
# Convolutional blocks use 3×3 padded convolutions, ReLU, and max pooling.
# Glorot normal initialization and zero biases are used throughout. The body
# initialization is generated independently of the classifier shape, so it is
# exactly shared across datasets for the same seed and architecture. Masks apply
# to weights only; biases remain trainable.

# %%
class ConvTicket(nn.Module):
    def __init__(self, n_classes, p=0.5, conv_dropout=False, architecture="conv4"):
        super().__init__()
        blocks = 2 if architecture == "conv4" else 1
        widths = [64, 128][:blocks]
        layers=[]; c=3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.ReLU(inplace=False),
                       nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=False), nn.MaxPool2d(2)]
            if conv_dropout: layers.append(nn.Dropout2d(p))
            c=w
        self.features=nn.Sequential(*layers)
        side=32 // (2 ** blocks)
        self.fc1=nn.Linear(c*side*side, 256); self.drop1=nn.Dropout(p)
        self.fc2=nn.Linear(256, 256); self.drop2=nn.Dropout(p)
        self.head=nn.Linear(256, n_classes)
        self.architecture=architecture
    def forward(self, x, ticket_dropout=False):
        x=self.features(x)
        x=torch.flatten(x, 1)
        x=F.relu(self.fc1(x))
        if ticket_dropout: x=self.drop1(x)
        x=F.relu(self.fc2(x))
        if ticket_dropout: x=self.drop2(x)
        return self.head(x)

def initialize_model(model, body_seed, head_seed):
    # Reset body using the same seed regardless of output dimension.
    body_modules=[m for m in list(model.features.modules()) + [model.fc1, model.fc2] if isinstance(m, (nn.Conv2d, nn.Linear))]
    torch.manual_seed(body_seed)
    for m in body_modules:
        nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    torch.manual_seed(head_seed)
    nn.init.xavier_normal_(model.head.weight); nn.init.zeros_(model.head.bias)
    return model

def weight_layers(model):
    return [(name, module.weight) for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))]

def dense_masks(model): return {n: torch.ones_like(w, dtype=torch.bool) for n,w in weight_layers(model)}

def apply_masks(model, masks, assert_zero=False):
    with torch.no_grad():
        for name,w in weight_layers(model):
            w.mul_(masks[name].to(device=w.device, dtype=w.dtype))
            if assert_zero and torch.count_nonzero(w[~masks[name].to(w.device)]).item():
                raise AssertionError(f"Pruned weights are nonzero in {name}")

def mask_density(masks, include_head=True):
    vals=[v for k,v in masks.items() if include_head or k != "head"]
    return sum(int(v.sum()) for v in vals)/sum(v.numel() for v in vals)

def layer_densities(masks): return {k: float(v.float().mean()) for k,v in masks.items()}

def prune_mask(masks, model, rates, method="magnitude", seed=0):
    new={k:v.detach().cpu().clone() for k,v in masks.items()}
    ws={n:w.detach().cpu() for n,w in weight_layers(model)}
    gen=torch.Generator().manual_seed(seed)
    for name, old in new.items():
        rate = rates[name]
        active=torch.nonzero(old.flatten(), as_tuple=False).flatten()
        nremove=min(len(active)-1, int(math.floor(len(active)*rate)))
        if nremove <= 0: continue
        if method == "magnitude":
            scores=ws[name].abs().flatten()[active]
            order=torch.argsort(scores, stable=True)
            remove=active[order[:nremove]]
        elif method == "random":
            remove=active[torch.randperm(len(active), generator=gen)[:nremove]]
        else: raise ValueError(method)
        new[name].view(-1)[remove]=False
    return new

def pruning_rates(model):
    return {n:(CONFIG["head_prune_fraction"] if n=="head" else
               CONFIG["conv_prune_fraction"] if n.startswith("features.") and "weight" not in n else
               CONFIG["fc_prune_fraction"]) for n,w in weight_layers(model)}

def masks_jaccard(a,b):
    return {k: float((a[k]&b[k]).sum() / (a[k]|b[k]).sum().clamp_min(1)) for k in a}

def mask_hash(masks):
    h=hashlib.sha256()
    for k in sorted(masks): h.update(k.encode()); h.update(masks[k].cpu().numpy().tobytes())
    return h.hexdigest()[:16]

def state_hash(state):
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()[:20]

# %% [markdown]
# ## Training with validation-only checkpoint selection
#
# We train for a fixed iteration budget. Validation loss is checked at the
# configured interval and once at the end if needed. Test accuracy at the best
# validation checkpoint and final iteration are both recorded. The final weights
# are retained only when needed to derive the next IMP mask. Checkpoints include
# optimizer and random-number states, allowing interrupted fits to resume.

# %%
def _norm(x, mean, std, device):
    x=x.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    mean=mean.to(device=device, dtype=torch.float32, non_blocking=True)
    std=std.to(device=device, dtype=torch.float32, non_blocking=True)
    return (x-mean[None,:,None,None])/std[None,:,None,None]

def _evaluate(model, loader, data, dropout=False, return_loss=True):
    model.eval(); ys=[]; preds=[]; losses=[]
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x=_norm(x,data["mean"],data["std"],DEVICE); y=y.to(DEVICE, non_blocking=True)
            logits=model(x, ticket_dropout=False)
            losses.append((float(criterion(logits,y).item()), len(y)))
            preds.append(logits.argmax(1).cpu()); ys.append(y.cpu())
    pred=torch.cat(preds).numpy(); target=torch.cat(ys).numpy()
    return {"loss":float(sum(v*n for v,n in losses)/sum(n for _,n in losses)),
            "accuracy":float(accuracy_score(target,pred))}

def _fit_id(data, seed, round_idx, label, masks, init_id, dropout, lr, variant):
    key={"dataset":data["name"],"seed":seed,"round":round_idx,"label":label,
         "mask":mask_hash(masks),"init":init_id,"dropout":dropout,"lr":lr,
         "variant":variant,"iterations":CONFIG["iterations"],"arch":CONFIG["architecture"],
         "batch_size":CONFIG["batch_size"],"eval_every":CONFIG["eval_every"],
         "train_n":len(data["train_y"]),"val_n":len(data["val_y"]),"test_n":len(data["test_y"]),
         "mean":data["mean"].tolist(),"std":data["std"].tolist(),"conv_dropout":CONFIG["conv_dropout"]}
    return hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]

def train_fit(data, seed, round_idx, label, masks, init_state, init_id, dropout, lr,
              keep_final=False, variant="native"):
    fitid=_fit_id(data,seed,round_idx,label,masks,init_id,dropout,lr,variant)
    done=OUT/"checkpoints"/f"{fitid}.json"
    final_path=OUT/"checkpoints"/f"{fitid}_final.pt"
    if done.exists():
        rec=json.loads(done.read_text())
        if keep_final and not final_path.exists(): raise RuntimeError("Completed search fit lacks pending final weights")
        return rec, (torch.load(final_path,map_location="cpu",weights_only=True) if keep_final else None), 0.0, True
    set_seed(int(fitid[:8],16))
    theta_hash=state_hash(init_state)
    theta_path=OUT/"masks"/f"theta0_{theta_hash}.pt"
    if not theta_path.exists(): torch.save({k:v.detach().cpu() for k,v in init_state.items()},theta_path)
    model=ConvTicket(data["n_classes"],CONFIG["dropout_p"],CONFIG["conv_dropout"],CONFIG["architecture"]).to(DEVICE)
    model.load_state_dict(init_state)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    apply_masks(model,masks)
    stable_label_seed=int(hashlib.sha256(label.encode()).hexdigest()[:8],16)
    train_loader,val_loader,test_loader,sampler=make_loaders(data,seed*1009+round_idx*97+stable_label_seed%997)
    criterion=nn.CrossEntropyLoss()
    best_loss=float("inf"); best_iter=0; best_state=None; val_hist=[]; step=0
    resume_path=OUT/"checkpoints"/f"{fitid}_resume.pt"
    if resume_path.exists():
        state=torch.load(resume_path,map_location="cpu",weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        step=state["step"]; best_loss=state["best_loss"]; best_iter=state["best_iter"]
        best_state=state["best_state"]; val_hist=state["val_hist"]
        torch.set_rng_state(state["torch_rng"])
        if DEVICE.type=="cuda" and state.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(state["cuda_rng"])
    start=time.time(); n_batches=max(1,math.ceil(len(data["train_y"])/CONFIG["batch_size"]))
    model.train()
    while step < CONFIG["iterations"]:
        epoch,offset=divmod(step,n_batches)
        sampler.set_position(epoch,offset*CONFIG["batch_size"])
        consumed=0
        for xb,yb in train_loader:
            if step>=CONFIG["iterations"]: break
            model.train(); xb=_norm(xb,data["mean"],data["std"],DEVICE); yb=yb.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                 enabled=bool(CONFIG["amp"] and DEVICE.type=="cuda")):
                logits=model(xb,ticket_dropout=dropout); loss=criterion(logits,yb)
            loss.backward()
            for name,w in weight_layers(model):
                if w.grad is not None: w.grad.mul_(masks[name].to(device=w.device,dtype=w.grad.dtype))
            optimizer.step(); apply_masks(model,masks,assert_zero=True)
            step+=1; consumed+=len(yb)
            if step%CONFIG["eval_every"]==0 or step==CONFIG["iterations"]:
                val=_evaluate(model,val_loader,data); val_hist.append({"iteration":step,**val})
                if val["loss"]<best_loss:
                    best_loss=val["loss"]; best_iter=step
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if CONFIG["checkpoint_every"] and step%CONFIG["checkpoint_every"]==0:
                torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
                    "best_loss":best_loss,"best_iter":best_iter,"best_state":best_state,"val_hist":val_hist,
                    "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if DEVICE.type=="cuda" else None},resume_path)
        if consumed==0 and step<CONFIG["iterations"]: raise RuntimeError("Training loader yielded no batches")
    if best_state is None:
        val=_evaluate(model,val_loader,data); best_loss=val["loss"]; best_iter=step
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    final_metric=_evaluate(model,test_loader,data)
    if keep_final: torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},final_path)
    model.load_state_dict(best_state); apply_masks(model,masks,assert_zero=True)
    best_metric=_evaluate(model,test_loader,data)
    rec={"fit_id":fitid,"dataset":data["name"],"seed":seed,"round":round_idx,"label":label,
         "variant":variant,"init_id":init_id,"mask_hash":mask_hash(masks),"dropout":bool(dropout),
         "theta0_file":theta_path.name,
         "dropout_p":CONFIG["dropout_p"] if dropout else 0.0,"learning_rate":lr,
         "early_stopping_iteration":best_iter,"validation_loss_best":best_loss,
         "test_accuracy_best":best_metric["accuracy"],"test_accuracy_final":final_metric["accuracy"],
         "fit_seconds":time.time()-start,"iterations":step,"amp":bool(CONFIG["amp"])}
    done.write_text(json.dumps(rec,indent=2)); resume_path.unlink(missing_ok=True)
    return rec,(torch.load(final_path,map_location="cpu",weights_only=True) if keep_final else None),rec["fit_seconds"],False

# %% [markdown]
# ## Checkpointed factorial experiment, controls, and transfer
#
# Each mask-search chain is trained with one dropout setting. Its terminal
# weights create the next layer-wise magnitude mask, then surviving parameters
# rewind to the same initialization. For a given mask and seed, A/D and B/C use
# identical masks and initial weights. Controls are limited to A and B. The
# transfer study keeps the target output layer dense and uses a separate fixed
# target-head seed; random-reinitialization controls retain the transferred mask.
#
# The expected number of fits is estimated after a timed pilot. The estimate
# counts factorial fits, controls, transfer fits, and the pilot; cache hits on a
# resumed run are reported separately.

# %%
RESULTS=[]; OVERLAPS=[]; _seen=set(); _fit_seconds=[]
def save_results():
    if RESULTS: pd.DataFrame(RESULTS).to_csv(OUT/"results.csv",index=False)
    if OVERLAPS: pd.DataFrame(OVERLAPS).to_csv(OUT/"mask_overlap.csv",index=False)

def save_mask(dataset,seed,chain,rnd,masks):
    path=OUT/"masks"/f"{dataset}_seed{seed}_{chain}_round{rnd:02d}.pt"
    torch.save({k:v.cpu().to(torch.uint8) for k,v in masks.items()},path)
    return path

def load_mask(path):
    raw=torch.load(path,map_location="cpu",weights_only=True)
    return {k:v.bool() for k,v in raw.items()}

def record_metric(rec,data,masks,condition,round_idx,phase="native",extra=None):
    row=dict(rec); row.update({"condition":condition,"phase":phase,"round":round_idx,
        "Pm":mask_density(masks,True),"Pm_body":mask_density(masks,False),
        "per_layer_density":json.dumps(layer_densities(masks),sort_keys=True),
        "mask_hash":mask_hash(masks),"dataset":data["name"]})
    if extra: row.update(extra)
    RESULTS.append(row); save_results(); return row

def _make_init(data,seed,head_seed=None):
    model=ConvTicket(data["n_classes"],CONFIG["dropout_p"],CONFIG["conv_dropout"],CONFIG["architecture"])
    body_seed=1000003+seed*7919+({"conv2":11,"conv4":23}[CONFIG["architecture"]])
    hs=(2000003+seed*104729+({"cifar10":1,"cifar100":2,"svhn":3}[data["name"]])) if head_seed is None else head_seed
    initialize_model(model,body_seed,hs)
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},body_seed,hs

def _model_for_rates(data):
    return ConvTicket(data["n_classes"],CONFIG["dropout_p"],CONFIG["conv_dropout"],CONFIG["architecture"])

def run_pilot(data,seed):
    model=_model_for_rates(data); init,_,_=_make_init(data,seed); model.load_state_dict(init)
    masks=dense_masks(model); n=min(CONFIG["benchmark_iterations"],CONFIG["iterations"])
    old=CONFIG["iterations"]; CONFIG["iterations"]=n
    rec,_,sec,_=train_fit(data,seed,-1,"pilot",masks,init,"pilot-init",False,BASE_LR,False,"pilot")
    CONFIG["iterations"]=old
    # The pilot is a real dense fit and is reusable as the dense A reference.
    return rec,sec

def run_native_dataset(data,seed,pilot_rec=None):
    init,body_seed,head_seed=_make_init(data,seed)
    torch.save(init,OUT/"masks"/f"{data['name']}_seed{seed}_theta0.pt")
    proto=_model_for_rates(data); proto.load_state_dict(init); rates=pruning_rates(proto)
    chains={False:[dense_masks(proto)],True:[dense_masks(proto)]}
    save_mask(data["name"],seed,"no_dropout",0,chains[False][0])
    save_mask(data["name"],seed,"dropout",0,chains[True][0])
    # Restore all masks already derived by earlier completed search rounds.
    for search_drop,chain_name in [(False,"no_dropout"),(True,"dropout")]:
        for prior_round in range(1,CONFIG["rounds"]+1):
            pth=OUT/"masks"/f"{data['name']}_seed{seed}_{chain_name}_round{prior_round:02d}.pt"
            if pth.exists(): chains[search_drop].append(load_mask(pth))
    chain_states={False:[],True:[]}
    for rnd in range(CONFIG["rounds"]+1):
        print(f"[{data['name']} seed {seed}] round {rnd}/{CONFIG['rounds']}")
        round_start=time.time()
        fits={}
        for search_drop,chain in [(False,"search_no_dropout"),(True,"search_dropout")]:
            masks=chains[search_drop][rnd]
            label="search_B" if search_drop else "search_A"
            # A/B search weights are also their diagonal ticket results.
            cond="B" if search_drop else "A"
            next_path=(OUT/"masks"/f"{data['name']}_seed{seed}_{'dropout' if search_drop else 'no_dropout'}_round{rnd+1:02d}.pt")
            keep=(rnd<CONFIG["rounds"] and not next_path.exists())
            rec,final,sec,cached=train_fit(data,seed,rnd,label,masks,init,f"native-body-{body_seed}-head-{head_seed}",
                search_drop,CONFIG["lr_by_condition"][cond],keep,"native")
            fits[search_drop]=(rec,final)
            if sec: _fit_seconds.append(sec)
            # Search path results are condition A or B.
            row=record_metric(rec,data,masks,cond,rnd,extra={"search_dropout":search_drop,"ticket_dropout":search_drop,
                "body_seed":body_seed,"head_seed":head_seed,"cached":cached})
            chain_states[search_drop].append({"round":rnd,"mask":mask_hash(masks),"fit_id":rec["fit_id"]})
            if rnd<CONFIG["rounds"]:
                if next_path.exists():
                    nextmask=load_mask(next_path)
                else:
                    if final is None: raise RuntimeError("Search final weights unavailable to derive next mask")
                    trained=ConvTicket(data["n_classes"],CONFIG["dropout_p"],CONFIG["conv_dropout"],CONFIG["architecture"])
                    trained.load_state_dict(final)
                    nextmask=prune_mask(masks,trained,rates,"magnitude",seed+rnd)
                    save_mask(data["name"],seed,"dropout" if search_drop else "no_dropout",rnd+1,nextmask)
                    (OUT/"checkpoints"/f"{rec['fit_id']}_final.pt").unlink(missing_ok=True)
                if len(chains[search_drop]) == rnd+1: chains[search_drop].append(nextmask)
                else: chains[search_drop][rnd+1]=nextmask
        # Cross conditions: same search chain mask/init, opposite ticket dropout.
        for search_drop,condition in [(True,"C"),(False,"D")]:
            masks=chains[search_drop][rnd]
            ticket_drop=not search_drop
            rec,_,sec,cached=train_fit(data,seed,rnd,f"cross_{condition}",masks,init,
                f"native-body-{body_seed}-head-{head_seed}",ticket_drop,CONFIG["lr_by_condition"][condition],False,"native")
            if sec: _fit_seconds.append(sec)
            record_metric(rec,data,masks,condition,rnd,extra={"search_dropout":search_drop,"ticket_dropout":ticket_drop,
                "body_seed":body_seed,"head_seed":head_seed,"cached":cached})
        overlap=masks_jaccard(chains[False][rnd],chains[True][rnd])
        for layer,jac in overlap.items(): OVERLAPS.append({"dataset":data["name"],"seed":seed,"round":rnd,"layer":layer,"jaccard":jac})
        save_results()
        # Pairing invariant: exactly identical mask hashes and original θ0 IDs.
        rows=[r for r in RESULTS if r["dataset"]==data["name"] and r["seed"]==seed and r["round"]==rnd and r["phase"]=="native"]
        by={r["condition"]:r for r in rows}
        assert by["A"]["mask_hash"]==by["D"]["mask_hash"]
        assert by["B"]["mask_hash"]==by["C"]["mask_hash"]
        assert by["A"]["init_id"]==by["D"]["init_id"] and by["B"]["init_id"]==by["C"]["init_id"]
        if CONFIG["controls"]:
            for base in ("A","B"):
                masks=chains[base=="B"][rnd]
                for variant in ("reinit","random_prune"):
                    if variant=="reinit":
                        ctrl_init,_,_= _make_init(data,seed+10000+(base=="B")*7)
                        ctrl_masks=masks
                    else:
                        ctrl_init,_,_=_make_init(data,seed+20000+(base=="B")*11)
                        ctrl_masks=prune_mask(dense_masks(proto),proto,rates,"random",seed*99991+rnd*101+(base=="B")*13)
                    save_mask(data["name"],seed,f"control_{base}_{variant}",rnd,ctrl_masks)
                    rec,_,sec,cached=train_fit(data,seed,rnd,f"control_{base}_{variant}",ctrl_masks,ctrl_init,
                        f"control-{variant}-{seed}-{rnd}-{base}",base=="B",CONFIG["lr_by_condition"][base],False,variant)
                    if sec: _fit_seconds.append(sec)
                    record_metric(rec,data,ctrl_masks,base,rnd,phase=f"control_{variant}",extra={"control_of":base,"cached":cached})
        print(f"[{data['name']} seed {seed}] round {rnd} completed in {time.time()-round_start:.1f}s")
    # Persist exact original initialization for later cross-dataset mask transfer.
    theta_path=OUT/"masks"/f"{data['name']}_seed{seed}_theta0.pt"
    torch.save(init,theta_path)
    return {"init":init,"body_seed":body_seed,"head_seed":head_seed,"chains":chains,"rates":rates}

def _transfer_init(data,source_init,seed,source_name):
    init,body_seed,head_seed=_make_init(data,seed,head_seed=3000001+seed*65537+({"cifar100":77,"svhn":88}[data["name"]]))
    model=_model_for_rates(data); model.load_state_dict(init)
    # Transfer only the feature extractor and both hidden fully connected layers.
    for key,val in source_init.items():
        if key.startswith("features.") or key.startswith("fc1.") or key.startswith("fc2."):
            init[key]=val.clone()
    return init,body_seed,head_seed

def run_transfer(target,seed,source_bundle):
    for source_drop,source_label in [(False,"no_dropout"),(True,"dropout")]:
        source_chain=source_bundle["chains"][source_drop]
        for rnd,source_mask in enumerate(source_chain):
            transfer_start=time.time()
            print(f"[{target['name']} seed {seed}] transfer from {source_label}, round {rnd}/{CONFIG['rounds']}")
            source_init=source_bundle["init"]
            transferred_init,body_seed,head_seed=_transfer_init(target,source_init,seed,"cifar10")
            target_model=_model_for_rates(target); target_model.load_state_dict(transferred_init)
            target_masks=dense_masks(target_model)
            for k in target_masks:
                if k!="head" and k in source_mask: target_masks[k]=source_mask[k].clone()
            # Classifier remains dense by design because source/target class counts differ.
            target_masks["head"].fill_(True)
            save_mask(target["name"],seed,f"transfer_{source_label}",rnd,target_masks)
            transfer_conditions=[("C" if source_drop else "A"),
                                 ("B" if source_drop else "D")]
            for ticket_drop,cond in zip((False,True),transfer_conditions):
                rec,_,sec,cached=train_fit(target,seed,rnd,f"transfer_{source_label}_{cond}",target_masks,
                    transferred_init,f"transfer-theta0-{source_init['fc1.weight'].shape}-{head_seed}",ticket_drop,
                    CONFIG["lr_by_condition"][cond],False,"transfer")
                if sec: _fit_seconds.append(sec)
                record_metric(rec,target,target_masks,cond,rnd,phase=f"transfer_{source_label}",extra={
                    "source_dataset":"cifar10","source_chain":source_label,"source_round":rnd,
                    "search_dropout":source_drop,"ticket_dropout":ticket_drop,"body_seed":body_seed,
                    "target_head_seed":head_seed,"head_pruned":False,"cached":cached})
                # Fresh random body initialization with exactly the same transferred mask.
                reinit,_,rehead=_make_init(target,seed+500000+(source_drop*3),
                    head_seed=3000001+seed*65537+({"cifar100":77,"svhn":88}[target["name"]]))
                rec,_,sec,cached=train_fit(target,seed,rnd,f"transfer_reinit_{source_label}_{cond}",target_masks,
                    reinit,f"transfer-reinit-{seed}-{rnd}-{source_label}",ticket_drop,
                    CONFIG["lr_by_condition"][cond],False,"transfer_reinit")
                if sec: _fit_seconds.append(sec)
                record_metric(rec,target,target_masks,cond,rnd,phase=f"transfer_reinit_{source_label}",extra={
                    "source_dataset":"cifar10","source_chain":source_label,"source_round":rnd,
                    "search_dropout":source_drop,"ticket_dropout":ticket_drop,"head_pruned":False,"cached":cached})
            # Native body mask with a dense head: matched reference for transfer figure.
            if rnd < len(source_bundle["chains"][source_drop]):
                native_mask=source_bundle["chains"][source_drop][rnd]
                fixed_head_seed=3000001+seed*65537+({"cifar100":77,"svhn":88}[target["name"]])
                native_init,_bs,_hs=_make_init(target,seed,head_seed=fixed_head_seed)
                nm=dense_masks(target_model)
                for k in nm:
                    if k!="head": nm[k]=native_mask[k].clone()
                native_conditions=[("C" if source_drop else "A"),
                                   ("B" if source_drop else "D")]
                for ticket_drop,cond in zip((False,True),native_conditions):
                    rec,_,sec,cached=train_fit(target,seed,rnd,f"native_matched_{cond}",nm,native_init,
                        f"native-body-{seed}-dense-head",ticket_drop,CONFIG["lr_by_condition"][cond],False,"native_matched")
                    if sec: _fit_seconds.append(sec)
                    record_metric(rec,target,nm,cond,rnd,phase="transfer_native_reference",extra={
                        "search_dropout":source_drop,"ticket_dropout":ticket_drop,"head_pruned":False,
                        "source_chain":source_label,"cached":cached})
            print(f"[{target['name']} seed {seed}] transfer {source_label} round {rnd} finished in {time.time()-transfer_start:.1f}s")

# %% [markdown]
# ## Run or resume the study
#
# A pilot fit supplies the measured iteration speed. The total estimate assumes
# every scheduled fit is new; completed checkpoint hits are skipped automatically.
# Long defaults can take many GPU-hours, so inspect the estimate before changing
# `quick` to false. The fixed training budget, not the validation-selected epoch,
# determines magnitude masks, matching rewind-style IMP.

# %%
def estimate_fit_count():
    ndata=len(CONFIG["datasets"]); nseed=len(CONFIG["seeds"]); r=CONFIG["rounds"]+1
    native_per=2*r + 2*r  # two search/diagonal fits and two cross-condition fits
    native_per += (2*2*r if CONFIG["controls"] else 0)
    transfer_per=0
    if CONFIG["transfer"] and "cifar10" in CONFIG["datasets"]:
        ntarget=sum(d!="cifar10" for d in CONFIG["datasets"])
        transfer_per=ntarget*r*(2*2*2 + 2*2) # source chains, two ticket modes, reinit, matched native refs
    return ndata*nseed*native_per + max(0,nseed*transfer_per)

def run_experiment():
    total=estimate_fit_count()+1  # Include the timed pilot itself.
    pilot_data=load_dataset(CONFIG["datasets"][0],CONFIG["seeds"][0])
    pilot,seconds=run_pilot(pilot_data,CONFIG["seeds"][0])
    # A pilot is short in full mode; use its per-step time, with an explicit warning.
    per_iter=seconds/max(1,pilot["iterations"])
    evals=math.ceil(CONFIG["iterations"]/CONFIG["eval_every"])
    estimate=total*(CONFIG["iterations"]*per_iter + evals*0.04)
    print(f"Estimated scheduled fits: {total}; timed pilot: {seconds:.1f}s/{pilot['iterations']} iterations")
    print(f"Rough full-study training estimate: {estimate/3600:.2f} hours (data/evaluation overhead may vary).")
    source_by_seed={}
    for ds in CONFIG["datasets"]:
        for seed in CONFIG["seeds"]:
            start=time.time(); set_seed(seed)
            data=pilot_data if ds==CONFIG["datasets"][0] and seed==CONFIG["seeds"][0] else load_dataset(ds,seed)
            print(f"\n=== Dataset {ds}, seed {seed} on {DEVICE} ===")
            bundle=run_native_dataset(data,seed,pilot if ds==CONFIG["datasets"][0] and seed==CONFIG["seeds"][0] else None)
            if ds=="cifar10": source_by_seed[seed]=bundle
            print(f"=== Dataset {ds}, seed {seed} finished in {(time.time()-start)/60:.1f} min ===")
            if ds!="cifar10" and CONFIG["transfer"] and seed in source_by_seed:
                run_transfer(data,seed,source_by_seed[seed])
            save_results()
    pd.DataFrame(RESULTS).to_csv(OUT/"results.csv",index=False)
    if OVERLAPS: pd.DataFrame(OVERLAPS).to_csv(OUT/"mask_overlap.csv",index=False)
    print(f"Saved metrics to {OUT/'results.csv'}")

# %% [markdown]
# ## Figures and compact summaries
#
# Curves show mean across seeds and min–max shading; with one seed the band is
# necessarily zero-width. The dense reference is shown with a dashed line.
# Transfer curves use `Pm_body` because the target output layer is deliberately
# dense. Native full-mask curves remain available in the main experiment plot.

# %%
CONDITION_LABELS={"A":"Search off / ticket off","B":"Search on / ticket on",
                  "C":"Search on / ticket off","D":"Search off / ticket on"}
CONDITION_COLORS={"A":"#1f77b4","B":"#ff7f0e","C":"#2ca02c","D":"#9467bd"}

def _curve(ax,df,x,y,label,color,marker):
    g=df.groupby(x)[y].agg(["mean","min","max"]).reset_index().sort_values(x)
    ax.plot(g[x],g["mean"],label=label,color=color,marker=marker,linewidth=1.8,markersize=4)
    ax.fill_between(g[x].to_numpy(float),g["min"].to_numpy(float),g["max"].to_numpy(float),color=color,alpha=.16,linewidth=0)

def plot_main():
    if not RESULTS:return
    df=pd.DataFrame(RESULTS); native=df[(df.phase=="native")]
    for ds in native.dataset.unique():
        d=native[native.dataset==ds]
        fig,axs=plt.subplots(1,2,figsize=(12,4.7))
        for c in "ABCD":
            q=d[d.condition==c]
            _curve(axs[0],q,"Pm","early_stopping_iteration",CONDITION_LABELS[c],CONDITION_COLORS[c],"o")
            _curve(axs[1],q,"Pm","test_accuracy_best",CONDITION_LABELS[c],CONDITION_COLORS[c],"o")
        for ax in axs:
            ax.set_xscale("log"); ax.invert_xaxis(); ax.grid(alpha=.25); ax.set_xlabel("Fraction of weights remaining (Pm; log scale)")
        axs[0].set_ylabel("Early-stopping iteration"); axs[1].set_ylabel("Test accuracy at best validation checkpoint")
        # Dataset-specific dense condition means are the dashed reference.
        dense=d[d["round"]==0]
        for c in "ABCD":
            if len(dense[dense.condition==c]):
                y=dense[dense.condition==c].test_accuracy_best.mean()
                axs[1].axhline(y,color=CONDITION_COLORS[c],linestyle="--",alpha=.45,linewidth=1)
        fig.suptitle(f"{ds.upper()}: dropout roles and iterative magnitude pruning")
        axs[1].legend(fontsize=8,loc="best"); fig.tight_layout()
        fig.savefig(OUT/f"{ds}_main.png",dpi=300,bbox_inches="tight"); fig.savefig(OUT/f"{ds}_main.pdf",bbox_inches="tight"); plt.show(); plt.close(fig)

def plot_controls():
    if not RESULTS:return
    df=pd.DataFrame(RESULTS)
    for ds in df.dataset.unique():
        d=df[(df.dataset==ds)&df.phase.str.startswith("control_")]
        if d.empty: continue
        fig,axs=plt.subplots(2,2,figsize=(11,8))
        for i,c in enumerate(("A","B")):
            main=df[(df.dataset==ds)&(df.phase=="native")&(df.condition==c)]
            _curve(axs[i,0],main,"Pm","early_stopping_iteration",f"{c} ticket",CONDITION_COLORS[c],"o")
            _curve(axs[i,1],main,"Pm","test_accuracy_best",f"{c} ticket",CONDITION_COLORS[c],"o")
            for variant,label,color in [("reinit","Random reinitialization","#d62728"),("random_prune","Random pruning","#7f7f7f")]:
                q=d[(d.control_of==c)&(d.phase==f"control_{variant}")]
                _curve(axs[i,0],q,"Pm","early_stopping_iteration",label,color,"s")
                _curve(axs[i,1],q,"Pm","test_accuracy_best",label,color,"s")
            axs[i,0].set_ylabel(f"Condition {c}: early iteration"); axs[i,1].set_ylabel("Test accuracy")
        for ax in axs.flat: ax.set_xscale("log");ax.invert_xaxis();ax.grid(alpha=.2);ax.set_xlabel("Pm")
        axs[0,1].legend(fontsize=8);fig.suptitle(f"{ds.upper()}: ticket and pruning controls");fig.tight_layout()
        fig.savefig(OUT/f"{ds}_controls.png",dpi=300,bbox_inches="tight");fig.savefig(OUT/f"{ds}_controls.pdf",bbox_inches="tight");plt.show();plt.close(fig)

def plot_overlap():
    if not OVERLAPS:return
    df=pd.DataFrame(OVERLAPS)
    for ds in df.dataset.unique():
        q=df[df.dataset==ds]; fig,ax=plt.subplots(figsize=(8,4.5))
        for layer,g in q.groupby("layer"):
            z=g.groupby("round").jaccard.agg(["mean","min","max"]).reset_index()
            ax.plot(z["round"],z["mean"],marker="o",label=layer)
            ax.fill_between(z["round"].to_numpy(),z["min"].to_numpy(),z["max"].to_numpy(),alpha=.12)
        ax.set(xlabel="IMP round",ylabel="Per-layer Jaccard overlap",ylim=(0,1.02),title=f"{ds.upper()}: dropout vs no-dropout masks")
        ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        fig.savefig(OUT/f"{ds}_mask_overlap.png",dpi=300,bbox_inches="tight");fig.savefig(OUT/f"{ds}_mask_overlap.pdf",bbox_inches="tight");plt.show();plt.close(fig)

def plot_transfer():
    if not RESULTS:return
    df=pd.DataFrame(RESULTS)
    t=df[df.phase.str.startswith("transfer") | (df.phase=="transfer_native_reference")]
    for ds in ("cifar100","svhn"):
        q=t[t.dataset==ds]
        if q.empty:continue
        fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
        for ax,ticket in zip(axs,(False,True)):
            z=q[q.ticket_dropout==ticket]
            native_condition="D" if ticket else "A"
            native=df[(df.dataset==ds)&(df.phase=="native")&(df.condition==native_condition)]
            ref=z[z.phase=="transfer_native_reference"]
            series=[(native,"native IMP (no-dropout search)","#1f77b4"),
                    (ref[~ref.search_dropout.fillna(False)],"native matched body mask (search off)","#17becf"),
                    (ref[ref.search_dropout.fillna(False)],"native matched body mask (search on)","#bcbd22"),
                    (z[z.phase=="transfer_no_dropout"],"from CIFAR-10 no-dropout search","#2ca02c"),
                    (z[z.phase=="transfer_dropout"],"from CIFAR-10 dropout search","#ff7f0e"),
                    (z[z.phase=="transfer_reinit_no_dropout"],"no-dropout mask, reinitialized","#9467bd"),
                    (z[z.phase=="transfer_reinit_dropout"],"dropout mask, reinitialized","#d62728")]
            for r,label,color in series:
                if not r.empty:_curve(ax,r,"Pm_body","test_accuracy_best",label,color,"o")
            ax.set_xscale("log");ax.invert_xaxis();ax.grid(alpha=.2);ax.set_xlabel("Body weights remaining (Pm_body; log scale)")
            ax.set_title("Ticket dropout on" if ticket else "Ticket dropout off")
        axs[0].set_ylabel("Test accuracy at best validation checkpoint");axs[1].legend(fontsize=7,loc="best")
        fig.suptitle(f"Mask transfer to {ds.upper()} (classifier stays dense)");fig.tight_layout()
        fig.savefig(OUT/f"{ds}_transfer.png",dpi=300,bbox_inches="tight");fig.savefig(OUT/f"{ds}_transfer.pdf",bbox_inches="tight");plt.show();plt.close(fig)

def summary_table():
    if not RESULTS:return pd.DataFrame()
    df=pd.DataFrame(RESULTS); df=df[df.phase=="native"]
    rows=[]
    for (ds,c),g in df.groupby(["dataset","condition"]):
        for target in (.50,.20,.05):
            ix=(g.Pm-target).abs().groupby(g.seed).idxmin()
            vals=g.loc[ix,"test_accuracy_best"].to_numpy()
            rows.append({"dataset":ds,"condition":c,"target_Pm":target,
                         "test_accuracy_mean":float(vals.mean()),"test_accuracy_std":float(vals.std(ddof=1)) if len(vals)>1 else np.nan,
                         "n_seeds":len(vals),"actual_Pm_mean":float(g.loc[ix,"Pm"].mean())})
    out=pd.DataFrame(rows);out.to_csv(OUT/"summary_sparsity.csv",index=False)
    try: display(out)
    except NameError: print(out.to_string(index=False))
    return out

def make_all_figures():
    plot_main();plot_controls();plot_overlap();plot_transfer();summary_table()

# %% [markdown]
# ## Interpretation and limitations
#
# Compare C with B and D with A to distinguish the mask-search and ticket-training
# roles of dropout. Similar C/B results alongside D/A suggest that mask search is
# the main source of benefit; D close to B suggests a substantial regularization
# effect during ticket training. Compare CIFAR-10, CIFAR-100, and SVHN with care:
# dataset difficulty and sample counts per class differ, and accuracy levels are
# not directly comparable. Inspect per-layer mask Jaccard values and the input
# connection densities to determine whether dropout changes pruning behavior.
#
# Transfer here carries both the CIFAR-10 body mask and its original body
# initialization; reinitialization controls help separate the value of that
# pairing. This differs from Morcos et al. (NeurIPS 2019), [“One Ticket to Win
# Them All”](https://papers.neurips.cc/paper_files/paper/2019/hash/a4613e8d72a61b3b69b32d040f89ad81-Abstract.html),
# which studied broader ticket generalization across datasets and optimizers with
# larger architectures and a different pruning setup. Our study uses layer-wise
# pruning, Conv-2/Conv-4, a dropout factorial, and a small number of datasets.
#
# Limitations include one architecture family, a fixed Adam learning-rate
# schedule, at most three default seeds, and no augmentation. Equal training
# budgets differ from the paper's three-times-longer dropout runs; custom
# condition learning rates also change the optimization comparison and are saved
# in the result table. Transfer Pm uses body weights because the target head stays
# dense. Min–max seed bands are descriptive ranges, not confidence intervals.

# %%
if __name__ == "__main__":
    run_experiment()
    make_all_figures()
