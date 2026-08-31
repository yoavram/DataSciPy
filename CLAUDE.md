# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Teaching material for Yoav Ram's *Introduction to Deep Learning with Python* workshop
(`python.yoavram.com`). There is no application, no library, and no test suite — the
deliverable is a set of Jupyter notebooks that must run top-to-bottom in a classroom.
"Correct" here means *pedagogically clear, reproducible from a fresh kernel, and
runnable within its allotted class time*.

## Branches are course editions, not features

`master` is the general/legacy course; each branch is a different delivery of the
course with a different audience and framework choice. Current branch: `DeepLearning`
(the newest deep-learning content, Keras 3 on JAX). Other live branches include
`torch` (PyTorch port), `amat2025a`/`amat2025b`, `kla2025`, `Mobileye`, `probml`,
`lam`, plus remote-only legacy branches (`kti2018`, `kti2020`, `amat2019`, `lam2020`,
`lam2021`, `trees`, `intuit`, `landa`, `IDC2018`).

Consequences:

- Notebooks are routinely moved between editions with `git checkout <branch> -- <path>`
  rather than merged. Check whether a notebook you need already exists on another
  branch before writing it.
- Never merge branches to "sync" them. Do not port PyTorch material onto a
  Keras/JAX branch or vice versa.
- `sessions/`, `exercises/`, and `solutions/` contents differ per branch — a link in
  `index.ipynb` that resolves on `master` may be dead here.

## Framework rules on this branch

JAX for from-scratch derivations; Keras 3 on the JAX backend for applied work.
**Do not introduce `torch`, `tensorflow`, `transformers`, or `tensorflow_datasets`**
into notebooks on this branch — including indirectly via a dataset loader. Datasets
are downloaded and parsed by hand (`urllib` + `tarfile`/`zipfile` + PIL/NumPy);
`sessions/finetuning.ipynb` has the reference loader to copy.

The backend is selected globally by `~/.keras/keras.json` (`"backend": "jax"`), so
session notebooks import `keras` directly with no `KERAS_BACKEND` dance. `index.ipynb`
documents the `os.environ['KERAS_BACKEND'] = 'jax'` fallback for students whose config
is unset; use that form only if a notebook must be self-contained.

## Environment and running notebooks

`requirements.txt` is the **student-facing** file (VS Code + `venv`, per `README.md`);
keep it complete and framework-clean. A local `.venv` already exists:

```bash
.venv/bin/python --version                       # 3.12
.venv/bin/python -c "import keras, jax; print(keras.__version__, keras.backend.backend(), jax.default_backend())"
.venv/bin/python -m pip install -r requirements.txt
```

Execute a notebook end to end (this is the closest thing to a test in this repo):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace sessions/K_CNN.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --stdout sessions/K_CNN.ipynb > /dev/null  # check only, no rewrite
```

Notebooks with long training cells are meant to be *taught*, not retrained: they save
a checkpoint and then reload it, and the training cell is sometimes commented out. Keep
both paths present when editing.

Known requirements gaps to watch: `librosa` (`exercises/audio.ipynb`) and `corner`
(`sessions/mle.ipynb`) are imported but not listed.

## Notebook house style (match it exactly)

- Opening markdown cell: `![Py4Eng](../logo.png)`, `# Title`, `## Yoav Ram`, then an
  "In this session we will understand:" bullet list. (`sessions/flow.ipynb` deviates
  with `img/logo.png` and no intro cell — that is a known defect, not a pattern.)
- First code cell: `%matplotlib inline`, imports, and
  `print('Keras:', keras.__version__, 'backend:', keras.backend.backend(), jax.default_backend())`.
- Closing cells: `# References` (links and papers) then `# Colophon` with the
  CC BY-NC-SA 4.0 notice and the Python logo image — copy verbatim from a sibling
  notebook.
- Markdown narrative between every code cell; no runs of bare code.
- Fixed `SEED`; `seaborn.set(style='white', context='talk')` where plots are styled.
- Checkpoint idiom, paths always relative to the notebook (`../data/...`):
  ```python
  model.save('../data/keras_cnn_model.keras')
  with open("../data/keras_cnn_history.p", "wb") as f: pickle.dump(history, f)
  # later, the load-instead-of-train path:
  model = keras.models.load_model('../data/keras_cnn_model.keras')
  ```
- `.history` is unwrapped at the call site (`history = model.fit(...).history`) and
  plotted with the local `plot_history` helper.
- Use *validation* terminology, not *test*, in Keras notebooks (`X_validation`,
  `val_accuracy`) — this was a deliberate sweep, see commit `3c4d5e8`.

## Exercises and solutions

Every assignment in `exercises/` has a matching same-named notebook in `solutions/`.
When authoring an exercise from an existing session notebook, strip the target code to
TODOs while leaving data loading and evaluation intact, and keep the solution notebook
in sync. `index.ipynb` links both as `[assignment](...) | [solution](...)`.

## Data

`data/` holds small committed teaching datasets (`anage_data.txt`, `FordA_*.tsv`,
`heart.csv`, …). Large or downloadable artifacts are gitignored by extension
(`*.keras`, `*.h5`, `*.npz`, `*.tar.gz`, `*.zip`, `*.pt`, …) and by directory
(`data/MNIST`, `data/ESC-50-master`, `data/sign-lang`, `data/hyena`, `data/gan`).
Do not commit downloaded corpora or trained weights; route new downloads through code
in the notebook (or `download_data.py` if reintroduced) and gitignore the output.
Notebooks are committed **with** their outputs — figures are part of the teaching
material — so expect large diffs and do not strip outputs.

## index.ipynb

The table of contents and the entry point students open. Any notebook added, renamed,
or dropped must be reflected there, and every link must resolve on the current branch.

## Planning documents

Markdown plans at the repo root (`DL2026_PLAN.md`, `autoencoders-plan.md`,
`density_plan.md`, `finetuning.md`) are the authoritative specs for in-progress
restructuring, including per-session time budgets in academic hours and explicit
out-of-scope lists. Read the relevant plan before restructuring a session notebook,
and update it as work lands.
