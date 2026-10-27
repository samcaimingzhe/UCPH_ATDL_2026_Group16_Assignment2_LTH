# Dropout × IMP convolutional experiment

Open `cnn_dropout_imp.ipynb` in Jupyter and run its cells in order. Install the packages in `requirements.txt`; install a PyTorch build appropriate for your GPU from the official PyTorch selector. The default run uses a CUDA device when available and otherwise CPU.

Before a full run, set `CONFIG["quick"] = True` in the configuration cell to exercise the pipeline with a small budget. Set `CONFIG["synthetic_smoke"] = True` to validate orchestration without downloading data; those results are synthetic and not research findings. For the real experiment, leave both false. `CONFIG["data_dir"]` is the torchvision dataset root and `CONFIG["output_dir"]` is where CSVs, checkpoints, and masks are saved. CIFAR and SVHN are downloaded by torchvision when absent if `download` is true. SVHN only uses its `train` and `test` splits.

The notebook saves each completed fit and periodic optimizer/RNG checkpoints, so re-running can resume. It writes metrics to `results.csv`, mask overlaps to `mask_overlap.csv`, summary values to `summary_sparsity.csv`, and masks/initializations under `masks/`. Figure files are saved in PNG and PDF formats.

The default full configuration is compute intensive (15 IMP rounds, 3 seeds, 3 datasets, controls, and transfer). Read the pilot timing estimate before committing to a full run. The transfer experiment transfers the CIFAR-10 body mask and original body initialization, while keeping a fixed-seed target classifier dense. Native full-head tickets and dense-head native references are both recorded.

The condition definitions imply A/D share masks and B/C share masks. Transfer is applied after the target dataset's native runs. All comments and plot labels are in English.
