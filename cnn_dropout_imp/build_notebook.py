"""Convert percent-format Python cells into a clean Jupyter notebook."""
import json
import hashlib
import re
from pathlib import Path

source = Path(__file__).with_name("cnn_dropout_imp.py")
target = source.with_suffix(".ipynb")
text = source.read_text(encoding="utf-8")
parts = re.split(r"^# %% ?(?:\[(markdown)\])?\s*$", text, flags=re.M)
cells = []
for i in range(1, len(parts), 2):
    kind = parts[i]
    body = parts[i + 1]
    lines = body.splitlines()
    if kind == "markdown":
        lines = [line[2:] if line.startswith("# ") else ("" if line == "#" else line) for line in lines]
        cell = {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines).strip() + "\n"}
    else:
        cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": body.strip() + "\n"}
    cell["id"] = hashlib.sha1((str(i) + body).encode()).hexdigest()[:8]
    cells.append(cell)
notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {target} ({len(cells)} cells)")
