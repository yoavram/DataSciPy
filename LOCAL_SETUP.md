# Local setup

Two supported routes. Both end with the same package list from
[`requirements.txt`](requirements.txt).

- **VS Code + `venv`** — the recommended route for the course. Follow
  [`README.md`](README.md).
- **Miniforge + conda/mamba** — described below. Use this if you already work in
  conda environments, or if you want a specific Python version without touching
  your system Python.

---

## Miniforge + conda/mamba

### 1. Install Miniforge

Download the installer for your platform from
<https://github.com/conda-forge/miniforge> and run it. Miniforge is a minimal
conda distribution that defaults to the `conda-forge` channel; `mamba` is the
faster solver and ships with it.

Confirm it works:

```bash
conda --version
mamba --version
```

### 2. Create the environment

```bash
mamba create -n datascipy python=3.12
mamba activate datascipy
```

Python 3.12 or 3.13. Anything older will not work with Keras 3.

### 3. Install the packages

```bash
python -m pip install -r requirements.txt
```

Install with `pip`, not `mamba`, even inside a conda environment: `jax`, `keras`
and `flowjax` release to PyPI first, and mixing solvers for these three tends to
produce a mismatched `jax` / `jaxlib` pair.

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name datascipy --display-name "Python (datascipy)"
```

Then pick **Python (datascipy)** as the kernel when you open a notebook.

### 5. Check the installation

```bash
python -c "import keras, jax; print(keras.__version__, keras.backend.backend(), jax.default_backend())"
```

Expected: a Keras version of `3.x`, backend `jax`, and a backend device of `cpu`
or `gpu`.

If the backend is not `jax`, set it once in `~/.keras/keras.json`:

```json
{"floatx": "float32", "epsilon": 1e-07, "backend": "jax", "image_data_format": "channels_last"}
```

or per-session, before importing Keras:

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'
import keras
```

---

## GPU

The course runs on CPU. Every notebook with a long training cell also ships a
`keras.models.load_model` path, so you can follow along without training
anything yourself.

If you do have an NVIDIA GPU and want to use it, replace the CPU `jax` with the
CUDA build **after** installing the requirements:

```bash
python -m pip install --upgrade "jax[cuda12]"
python -c "import jax; print(jax.devices())"
```

`requirements.txt` deliberately pins plain `jax` so that it installs on every
platform, including macOS. Monitor a GPU with
`python -m pip install gpustat && gpustat -cp -i 0.1`.

---

## Data

Small teaching datasets are committed under `data/`. The large ones and the
pretrained ImageNet weights are downloaded on demand:

```bash
python download_data.py --list     # what is available, how big, what is already here
python download_data.py            # everything the Day 1-4 notebooks need
python download_data.py cub esc50  # just these two
python download_data.py --bonus    # also the 3.2 GB hyena archive for the Day 3 bonus
```

Budget roughly 2 GB for the default set and another 3.2 GB if you want the bonus
case study. Archives are deleted after extraction; pass `--keep-archives` to keep
them. Re-running is cheap — anything already extracted is skipped.

---

## Notes for maintainers

- `requirements.txt` is the **student-facing** file. Keep it complete, and keep
  it free of `torch`, `tensorflow`, `transformers` and `tensorflow_datasets` —
  this edition of the course is JAX and Keras 3 only.
- There is no `pixi.toml` on this branch. If you add one for local development,
  it is a development convenience and does not replace `requirements.txt`.
- Notebooks are committed **with** their outputs; the figures and training logs
  are part of the teaching material. Do not strip them.
