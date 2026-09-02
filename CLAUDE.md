# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Teaching material for Yoav Ram's *Introduction to Deep Learning with Python* workshop
(`python.yoavram.com`). There is no application, no library, and no test suite — the
deliverable is a set of Jupyter notebooks that must run top-to-bottom in a classroom.
"Correct" here means *pedagogically clear, reproducible from a fresh kernel, and
runnable within its allotted class time*.

## Branches are course editions, not features

`master` is the general/legacy course. Every other branch is a separate *delivery* of
the course, with its own audience, duration and framework choice — some are named after
the organization they were taught for, some after the year, some after the framework
(`torch` is the PyTorch port). Run `git branch -r` to see what exists; do not assume a
branch is a feature branch or that it is meant to be merged.

Current branch: **`DL2026`** — the newest deep-learning content, Keras 3 on the JAX
backend, four days plus two more in `yoavram/nanochat`.

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
fetched by `download_data.py`. For the house pattern when a dataset arrives as a
tarball — download, `tarfile.extractall(..., filter='data')`, decode with PIL into a
preallocated array, cache to `.npy` — see `sessions/transfer.ipynb`, or
`git show dl2026-plan:DL2026_GPU_HANDOFF.md` §2a.

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

`requirements.txt` was audited against every import in every notebook and is complete;
if you add an import, add it there too. Note `keras>=3.15` is a floor, not a preference:
the checkpoints in `data/` are written by Keras 3.15.1 and earlier Keras cannot
deserialize them (see issue #6).

## Notebook house style (match it exactly)

- Opening markdown cell: `![Py4Eng](../logo.png)`, `# Title`, `## Yoav Ram`, then an
  "In this session we will understand:" bullet list. All 18 sessions use `../logo.png`
  and all 18 have a Colophon, but only 6 currently carry the intro list — add it to new
  notebooks, and to old ones you are already editing, rather than treating its absence
  as the pattern.
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

Every assignment in `exercises/` has a matching same-named notebook in `solutions/`,
and so does every *in-session* exercise — `FFN`, `K_FFN`, `functional_keras` and
`softmax_regression` all have one. Keep it that way.

When authoring an exercise from an existing session notebook, strip the target code to
TODOs while leaving data loading and evaluation intact, and keep the solution notebook
in sync. `index.ipynb` links both as `[assignment](...) | [solution](...)`.

Two session notebooks therefore **cannot** run top to bottom, by design:
`softmax_regression` (the `mygradient` stub is a `SyntaxError` until filled in) and
`K_FFN` (its exercise produces the checkpoint a later cell loads). Any automated check
should assert *no unintended errors* rather than no errors — `sessions/jax.ipynb` also
stores one deliberate `TypeError`, demonstrating why `static_argnames` is needed.

## Data

`data/` holds small committed teaching datasets (`anage_data.txt`, `FordA_*.tsv`,
`heart.csv`, …). Large or downloadable artifacts are gitignored by extension
(`*.keras`, `*.h5`, `*.npz`, `*.tar.gz`, `*.zip`, `*.pt`, …) and by directory
(`data/MNIST`, `data/ESC-50-master`, `data/CUB_200_2011`, `data/Dataset`, `data/gan`).
Do not commit downloaded corpora or trained weights; route new downloads through
`download_data.py` (`--list` shows what it fetches) or through code in the notebook, and
gitignore the output.
Notebooks are committed **with** their outputs — figures are part of the teaching
material — so expect large diffs and do not strip outputs.

## index.ipynb

The table of contents and the entry point students open. Any notebook added, renamed,
or dropped must be reflected there, and every link must resolve on the current branch.

## Project history and planning

The DL2026 restructuring is **complete** and its planning documents have been removed
from this branch, deliberately, so that students downloading the repository do not get
them. They are preserved in git and are the best available account of why this branch
looks the way it does — including per-session time budgets in academic hours, measured
results for every notebook, and a list of assumptions that measurement overturned:

```bash
git show dl2026-plan:DL2026_PLAN.md          # the full plan and progress log
git show dl2026-plan:DL2026_GPU_HANDOFF.md   # the brief for the GPU-bound notebooks
```

Read that plan before restructuring a session notebook. It records, among other things,
which notebooks cannot run top to bottom *by design* (`sessions/jax.ipynb` has a
deliberate error demonstrating `static_argnames`), why the checkpoints require
`keras>=3.15`, and what was tried and rejected.

Remaining known work is tracked as issues rather than in a plan file:

- [DataSciPy#6](https://github.com/yoavram/DataSciPy/issues/6) — whole-model `.keras`
  checkpoints are tied to the Keras version that wrote them; proposes saving weights only.
- [nanochat#1](https://github.com/yoavram/nanochat/issues/1) — the one unfinished phase,
  a Keras self-attention exercise on FordA, whose deliverable belongs in that repository.

Two older review documents may still be present in the working tree but are untracked and
fully spent: `density_plan.md` produced today's `sessions/flow.ipynb`, and
`autoencoders-plan.md` has been carried out apart from leftovers that were finished
separately.
