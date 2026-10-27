# Validation notes

- Both the notebook source and helper script pass Python compilation checks.
- The generated notebook passes `nbformat` validation (17 cells).
- A one-iteration synthetic Conv-2 IMP smoke run completed on CPU. It produced factorial result rows, per-layer mask-overlap records, saved masks, and PNG/PDF figures.
- A repeated smoke invocation reused completed fit checkpoints rather than adding fit records.
- The installed torchvision CIFAR-10 test split loaded offline. A real-data loader check formed the seeded 5,000-image validation holdout and read a batch with two workers.
- The complete CIFAR-10/CIFAR-100/SVHN experiment and the transfer study were not run; full defaults are compute intensive. The notebook estimates runtime from its own pilot fit before the study.

Synthetic smoke metrics are pipeline checks only and are not scientific results.
