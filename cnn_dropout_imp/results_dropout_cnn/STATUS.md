# Partial CIFAR-10 run status

The full default run was stopped at the user's request. These files preserve the completed work and the mid-fit checkpoint.

- Dataset: CIFAR-10, seed 0, Conv-4, 25,000 iterations per fit.
- Completed round 0 fits: A (no dropout), B (dropout), and C (dropout search, no-dropout ticket). Their test accuracy at the best validation checkpoint was 0.7610, 0.7890, and 0.7537, respectively. These are preliminary single-seed results, not a completed experiment.
- The D fit (no-dropout search, dropout ticket) was interrupted and has an optimizer/RNG resume checkpoint in `checkpoints/`.
- The dense mask and round-1 masks for both search chains, plus the initialization, are saved in `masks/`.
- `results.csv` includes the pilot and completed fits. The interrupted D fit has no final result row yet.

Open the notebook from this `cnn_dropout_imp/` folder and run all cells to resume the saved D fit. Keep the experiment output directory set to `results_dropout_cnn`. The original dataset root recorded in `config.json` is machine-specific; set `CONFIG["data_dir"]` to your torchvision data directory if needed.

Observed full-fit times for A/B/C were approximately 12–15 minutes each. The initial 200-iteration pilot included one-time CUDA startup overhead, so its original projected full-study time was overly conservative.
