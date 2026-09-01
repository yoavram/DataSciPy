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
are downloaded and parsed by hand (`urllib` + `tarfile`/`zipfile` + PIL/NumPy) or
fetched by `download_data.py`; `DL2026_GPU_HANDOFF.md` §2a has the reference
loader to copy.

**The backend is not configured by anything in the notebooks, and this is a live
trap.** Keras 3 hardcodes `_BACKEND = "tensorflow"`; the course does not install
TensorFlow; so with the backend unset, `import keras` dies with
`ModuleNotFoundError: No module named 'tensorflow'` in every session notebook.
Reproduce it with `env -u KERAS_BACKEND KERAS_HOME=$(mktemp -d) python -c "import keras"`.

Three things select the backend, in Keras's own order of precedence:

1. the `KERAS_BACKEND` environment variable;
2. `$KERAS_HOME/keras.json`, defaulting to `~/.keras/keras.json` — **machine-local
   state, not repo state**, which is why a machine that has run Keras before may
   work while a fresh one does not;
3. otherwise the TensorFlow default, i.e. failure.

The repo ships `.env` with `KERAS_BACKEND=jax`, which VS Code's Python extension
loads automatically (`python.envFile` is pinned in `.vscode/settings.json`). That
covers the route `README.md` tells students to use. It does **not** cover Jupyter
launched from a terminal, `nbconvert`, or a bare `python` — set `KERAS_BACKEND=jax`
in the environment for those. Keras only ever reads `$KERAS_HOME/keras.json`; a
`keras.json` in the working directory is ignored.

Do not set `KERAS_HOME` to the repo to solve this: it also relocates Keras's dataset
and pretrained-weight cache into the course folder.

If a notebook must be self-contained regardless of environment, use the
`os.environ['KERAS_BACKEND'] = 'jax'` form *before* importing Keras, as
`index.ipynb` documents.

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
(`data/MNIST`, `data/ESC-50-master`, `data/CUB_200_2011`, `data/Dataset`, `data/gan`).
Do not commit downloaded corpora or trained weights; route new downloads through code
in the notebook (or `download_data.py` if reintroduced) and gitignore the output.
Notebooks are committed **with** their outputs — figures are part of the teaching
material — so expect large diffs and do not strip outputs.

## index.ipynb

The table of contents and the entry point students open. Any notebook added, renamed,
or dropped must be reflected there, and every link must resolve on the current branch.

## Planning documents

`DL2026_PLAN.md` is the authoritative spec for the in-progress restructuring,
including per-session time budgets in academic hours, a progress log, and an
explicit out-of-scope list. `DL2026_GPU_HANDOFF.md` is the brief for the two
GPU-bound notebooks. Read the relevant one before restructuring a session
notebook, and update the plan as work lands.

`autoencoders-plan.md` is an unexecuted review of `sessions/autoencoders.ipynb`
proposing a substantial upgrade (latent-space visualization, bottleneck sweep,
convolutional autoencoder) and noting a real indexing bug in it.
`density_plan.md` has already been carried out — it is the review that turned the
old `density-estimation.ipynb` into today's `sessions/flow.ipynb`.
