# Do winning tickets transfer across datasets?

UCPH Advanced Topics in Deep Learning 2026 — Group 16, Assignment 2 (mini research report).

Building on *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*
(Frankle & Carbin, ICLR 2019), we ask whether the **sparse structure** of a winning
ticket found on one dataset is itself a useful inductive bias on a different dataset,
and **which layers** carry the transferable part.

## Hypotheses

- **H1 (transfer).** A mask found by IMP on CIFAR-10, paired with the original
  initialisation θ₀, trains better on a new dataset than a random mask with the
  same layer-wise sparsity — but worse than the target's own (native) ticket.
- **H2 (mask vs. init).** The benefit depends on the mask–init pairing:
  transferred mask + re-initialisation loses most of it.
- **H3 (layers).** Early-layer masks transfer (generic features); late-layer
  masks are task-specific, so transferring them gives little or negative gain.
- **H4 (structure).** Layer-wise overlap between source and target masks is above
  chance mainly in early layers, and the gap grows with sparsity. Overlap between two
  IMP runs on the *same* data with a different data order is the reference ceiling.

## Experimental design

For each seed one body initialisation θ₀ is shared by all datasets (only the
classifier head depends on the number of classes; it is never pruned or transferred).

| condition | mask | init | answers |
|---|---|---|---|
| `native` | IMP on the dataset itself | θ₀ | upper reference |
| `transfer` | IMP on CIFAR-10 (source) | θ₀ | H1 |
| `transfer_reinit` | IMP on source | fresh θ′ | H2 |
| `random` | random, same per-layer counts | θ₀ | lower reference |
| `native_reinit` | own IMP mask | fresh θ′ | Frankle & Carbin control |
| `native_sgd2` (source only) | IMP on source, other data order | θ₀ | overlap ceiling (H4) |
| `ablate_prefix` | source mask on layers 1…j, random elsewhere | θ₀ | H3 |
| `ablate_single` | source mask on one layer, random elsewhere | θ₀ | H3 |

Defaults (`--preset paper`) follow the paper's Conv-4 setup: Adam, lr 3·10⁻⁴,
batch 60, layer-wise magnitude pruning of 10 % (conv) / 20 % (fc) of the remaining
weights per round, 15 rounds (down to ≈5.6 % of body weights), no augmentation,
5 000 held-out training images for early stopping, 3 seeds. Targets: SVHN,
CIFAR-100, Fashion-MNIST (padded to 32×32, 3 channels).

Metrics per run: test accuracy at the best-validation epoch (main metric),
final test accuracy, NLL and ECE (useful for follow-up work on calibration).

## Usage

```bash
pip install -r requirements.txt

# 1. sanity check: synthetic data, CPU, ~1 minute
python run_transfer.py --preset smoke && python analyze.py --exp smoke

# 2. the real thing on one GPU (resumable; just re-run the same command after a crash)
python run_transfer.py --preset paper --seeds 0 1 2 --download --amp

# 3. or in parallel across GPUs: source IMP per seed, then one job per (seed, target)
bash scripts/run_parallel.sh                         # PRESET=paper SEEDS="0 1 2"
PRESET=quick SEEDS="0" GPUS="0" bash scripts/run_parallel.sh
JOBS_PER_GPU=3 bash scripts/run_parallel.sh          # Conv-4 is small; share GPUs

# 4. figures and tables
python analyze.py --exp paper
```

`--preset quick` (Conv-4, 10 rounds, 12 epochs, 2 targets, 1 seed) gives every
figure in a few GPU-hours and is the recommended first run. Every option can be
overridden on the command line, e.g. `--preset paper --targets svhn --epochs 15`.
`python run_transfer.py -h` lists all of them.

Datasets are downloaded by torchvision into `data/` (`--download`, or
`scripts/run_parallel.sh` does it once up front) and cached as uint8 tensors that
live on the GPU during training.

### Approximate cost (Conv-4, `paper` preset)

Per seed: 16 + 16 (source IMP + SGD control) + 30 source controls +
3 × (16 native + 60 control) target runs + 3 × 3 × 11 ablation runs ≈ 390 trainings of
25 epochs. Use `--cond_rounds 3 6 9 12 15` to train the controls at fewer
sparsity levels, `--ablation_rounds 10` for a single ablation level, or
`--no_sgd_control` to cut further.

## Outputs

```
results/<exp>/
  config.json
  seed<k>/masks/<dataset>_r<RR>.pt         bit-packed masks after each IMP round
  seed<k>/results/<dataset>.jsonl          one line per training run
  analysis/
    summary.csv, summary.md                mean ± std over seeds
    fig_accuracy.pdf                       accuracy vs. remaining weights, all conditions
    fig_gain.pdf                           gain over the random mask (pp)
    fig_overlap.pdf, overlap.csv           layer-wise chance-normalised mask overlap
    fig_ablation_r<RR>.pdf                 prefix / single-layer transfer ablations
```

Mask overlap is reported as `(|A∩B|/|A| − d) / (1 − d)` where `d` is the layer
density: 0 means no more overlap than two random masks, 1 means identical masks.

## Code

| file | content |
|---|---|
| `lth/models.py` | Conv-2/4/6, LeNet-300-100; shared-body / per-dataset-head initialisation |
| `lth/masking.py` | layer-wise magnitude pruning, random masks, overlap, mask enforcement |
| `lth/data.py` | dataset loading to GPU-resident uint8 tensors |
| `lth/train.py` | training loop, early-stopping bookkeeping, accuracy / NLL / ECE |
| `run_transfer.py` | experiment driver (resumable) |
| `analyze.py` | aggregation over seeds, figures, tables |

## Reference

J. Frankle and M. Carbin. *The Lottery Ticket Hypothesis: Finding Sparse, Trainable
Neural Networks.* ICLR 2019.
Related: A. Morcos et al. *One ticket to win them all: generalizing lottery ticket
initializations across datasets and optimizers.* NeurIPS 2019 (transfers
mask **and** trained-from init jointly on large models; we isolate mask vs. init and
analyse transfer per layer).
