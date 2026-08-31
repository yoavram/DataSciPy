# DL2026 — GPU handoff: Phases 2 and 3

**For:** an agentic coding session on a machine with a CUDA GPU.
**Scope:** exactly two notebooks — `sessions/transfer.ipynb` (new) and
`sessions/audio.ipynb` (restructure). Nothing else.

This document is self-contained; you do not need the conversation it came from.
`DL2026_PLAN.md` in the repo root is the full course plan and the authority on
anything not covered here (§3 is Phase 2, §4 is Phase 3). `CLAUDE.md` has the
repository conventions. Read both before starting.

---

## 0. Why this work is remote

Both notebooks train ImageNet-scale backbones (EfficientNetV2S) on several
thousand images. Neither is feasible on the CPU-only laptop where the rest of
the branch is being assembled. Everything else in the plan is being done locally
in parallel — see §5 for the file boundaries that keep us from colliding.

---

## 1. Getting set up

```bash
git clone https://github.com/yoavram/DataSciPy.git
cd DataSciPy
git checkout DL2026            # base branch, assembled in Phase 1
git checkout -b DL2026-gpu     # do your work here; we merge it back
```

Environment (Python 3.12 or 3.13):

```bash
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install librosa      # needed by audio.ipynb, being added to requirements.txt locally
```

Confirm the backend and that the GPU is actually visible — do this before
anything else, and paste the output into your final report:

```bash
KERAS_BACKEND=jax .venv/bin/python -c "import keras, jax; print(keras.__version__, keras.backend.backend(), jax.default_backend(), jax.devices())"
```

You need `jax[cuda12]` rather than plain `jax` for GPU. `requirements.txt` is the
*student-facing* CPU-friendly file — **do not** edit it to add a CUDA wheel;
install the GPU jax in your environment and mention it in your report instead.

Reference numbers from the laptop this was assembled on: keras 3.14.0, jax 0.9.2,
backend `jax`, devices CPU.

## 2. Hard constraints

1. **No `torch`, `tensorflow`, `transformers`, or `tensorflow_datasets`** — not
   even as a dataset loader, not even in a commented-out cell. The whole course
   is JAX + Keras 3 on the JAX backend. This is the single most important rule
   here; a `tensorflow_datasets` import silently drags TensorFlow into a
   student's environment and breaks the JAX backend story.
2. Datasets are downloaded and parsed by hand: `urllib` → `tarfile`/`zipfile` →
   PIL/NumPy. **`sessions/finetuning.ipynb` contains the reference loader — copy
   its shape.** Cache the parsed arrays under `data/` and skip re-download when
   the cache is present.
3. Do not commit datasets or weights. `.gitignore` already covers `*.keras`,
   `*.h5`, `*.npz`, `*.npy`, `*.zip`, `*.tar.gz`, `*.wav`. Add a `.gitignore`
   entry for any new download directory you create under `data/`.
4. Match the notebook house style exactly (see `CLAUDE.md` → *Notebook house
   style*, and `DL2026_PLAN.md` §8). Copy the `# Colophon` cell verbatim from a
   sibling notebook. Markdown narrative between every code cell.
5. **Commit notebooks with their outputs.** Figures and training logs are the
   teaching material; this repo deliberately does not strip outputs. The trained
   checkpoints themselves stay untracked (see §6).
6. Use *validation* terminology, not *test* — `X_validation`, `val_accuracy`.
   This was a deliberate sweep across the Keras notebooks.

## 3. Phase 2 — new `sessions/transfer.ipynb`

Day 3's in-class transfer-learning session, 2 academic hours (≈100 min) of class
time. It replaces `sessions/finetuning.ipynb` (hyena re-ID) in the teaching slot,
because 256 open-set identities with ~12 images each is the wrong first example.

**Dataset: CUB-200-2011.** 11,788 images, 200 bird species, official split
5,994 train / 5,794 validation. `https://data.caltech.edu/records/65de6-vp158`
(the `CUB_200_2011.tgz` tarball; check the URL still resolves and note in the
notebook what you used). Chosen over Flowers/Oxford-Pets because it does *not*
saturate, so the probe-versus-fine-tune gap is visible. Use the official split
from `train_test_split.txt`, not a random one.

**Structure:**

1. Intro cell in house style: what transfer learning is; why a 60M-parameter
   model cannot be trained on 6,000 images.
2. Load, resize, visualize a grid of species.
3. Backbone: `keras.applications.EfficientNetV2S(weights='imagenet',
   include_top=False, pooling='avg')`.
4. **Protocol A — linear probe.** Run the frozen backbone once over the whole
   dataset, cache the 1280-d embeddings, then train a Keras `Dense` softmax head
   on the cached features. Trains in seconds, so this one runs live in class:
   keep the training cell active and show the loss curve. The head is Keras, not
   scikit-learn — it reuses Day 2's softmax model directly.
5. **Protocol B — full fine-tune.** Unfreeze the top of the backbone with
   batch-norm layers kept frozen, low LR, cosine decay with warmup, label
   smoothing. This will not run live: save a checkpoint and provide the
   `keras.models.load_model` path next to the (commented-out) training cell, the
   way `sessions/K_CNN.ipynb` does.
6. Comparison table: probe vs fine-tune, top-1 and top-5.
7. Discussion cell. Write it around the numbers you actually measure, and cover:
   - Published ImageNet-pretrained fine-tuning baselines on CUB sit near 85%
     (ResNet-50), 73–77% for older backbones; fine-tuning beats frozen feature
     extraction by roughly 2–10%.
   - Training the same architecture from scratch on CUB collapses — 10.9% vs
     57.0% with pretraining in the classic AlexNet-era comparison. This one
     number motivates the entire session.
   - **State the caveat, do not hide it:** 59 of ImageNet's 1000 classes are
     already bird categories overlapping CUB. That is *why* this transfers so
     well, and it is the honest reason the hyena case study is harder.
   - Probe vs fine-tune is not settled. Kumar et al. 2022 (arXiv:2202.10054):
     fine-tuning wins in-distribution by ~2% but loses out-of-distribution by
     ~7%, because it distorts good pretrained features under large shift;
     linear-probe-then-fine-tune does best. Note that the freeze-then-unfreeze
     staging used here *is* LP-FT.
   - **Report a range over 2 seeds, not a single number.** Run-to-run variance
     from head initialization and batch ordering is real and documented.

**References cell:** Kumar et al. 2022 (arXiv:2202.10054); Branson et al. 2014
(arXiv:1406.2952); Wah et al. 2011 (CUB-200-2011); the Keras transfer learning
guide.

## 4. Phase 3 — restructure `sessions/audio.ipynb`

Currently an EchoNet-style CNN trained from scratch on ESC-50 for 150 epochs.
This notebook was brought onto the branch from `origin/master` in Phase 1 and is
otherwise untouched. Budget: 1 academic hour of class time. Extend it into a
three-way comparison mirroring `transfer.ipynb`, on a different modality.

**New structure:**

1. ESC-50 data and mel-spectrogram preparation — existing material, keep.
2. **Baseline** — the from-scratch EchoNet CNN. Existing material; provide a
   checkpoint, do not train live.
3. **Linear probe** — feed the mel-spectrogram to a frozen ImageNet-pretrained
   `EfficientNetV2S` (or `ResNet50`) as a 3-channel image; cache embeddings;
   Keras softmax head.
4. **Fine-tune** — unfreeze the top, low LR, as in `transfer.ipynb`.
5. Comparison and discussion.

**Channel construction — the interesting detail.** Do *not* replicate one
grayscale spectrogram across three channels. Follow Palanisamy et al. and use
mel-spectrograms computed with **different window sizes and hop lengths** in each
channel, 128 mel bins, log-scaled. It is cheap to implement and it makes the
point that "make it look like an RGB image" has better and worse answers.

**Evidence for the discussion cell:** Palanisamy et al. 2020 (arXiv:2007.11154,
IJCNN) — an ensemble of ImageNet-pretrained DenseNet reaches 92.89% on ESC-50 and
87.42% on UrbanSound8K, state of the art at the time; and for a fixed
architecture, pretrained weights beat random initialization. Earlier
ImageNet-weighted ResNet work passed 91.5% on ESC-50 by splitting the
spectrogram across the frequency axis into three channels.

**No ensemble** — but keep the reason: the ensemble helped precisely because of
run-to-run variance, so report a range over seeds instead.

**Expectation to verify rather than assume.** ImageNet features are a weaker
prior for environmental audio than AudioSet features would be, and 92.89% is a
full fine-tune of an ensemble, not a frozen probe. **The frozen probe may well
land below the from-scratch CNN.** Run it and write the discussion around
whatever actually happens. "The domain gap is real and pretrained is not
automatically better" is a legitimate and more valuable lesson than a rigged
win. Record the measured numbers in the notebook itself, not just in your report.

Note ESC-50 ships with 5 official folds; if you use a single fold split rather
than proper cross-validation, say so explicitly in the notebook.

## 5. Boundaries — do not touch these

Work in parallel locally covers the rest of the plan. To keep the merge clean,
**only** create `sessions/transfer.ipynb` and edit `sessions/audio.ipynb`.

Specifically, do **not** edit:

- `index.ipynb` — rewritten locally (Phase 6). It will link your two notebooks;
  you do not need to add the links.
- `requirements.txt` — completed locally. Report anything you needed instead.
- `download_data.py` — being written locally to fetch CUB and ESC-50. If you
  write download code inside your notebooks (you should), tell us the URLs and
  cache paths you used so they can be lifted into it.
- `sessions/flow.ipynb`, `sessions/finetuning.ipynb`, `DL2026_PLAN.md`,
  `CLAUDE.md`, anything under `exercises/` or `solutions/`.

`sessions/finetuning.ipynb` has known defects (a train/validation split that
leaks by `image_id`, among others) — they are documented in `DL2026_PLAN.md` §5
and explicitly **out of scope**. Do not fix them, and do not copy its splitting
logic; copy only its download/parse loader.

## 6. Deliverables

1. Branch `DL2026-gpu` pushed, with the two notebooks committed **with outputs**.
   One commit per phase is fine.
2. The trained checkpoints under `data/` — these are gitignored, so hand them
   over out of band (the machine's path is enough if the box is reachable, or
   attach them). Name them descriptively, e.g. `data/cub_effnetv2s_probe.keras`,
   `data/cub_effnetv2s_finetune.keras`, `data/esc50_*.keras`, and make sure the
   `load_model` path in the notebook matches the name exactly.
3. A short report containing:
   - the environment line from §1 (keras/jax versions, devices);
   - **measured** top-1/top-5 for CUB probe vs fine-tune, over 2 seeds;
   - **measured** ESC-50 accuracy for from-scratch vs probe vs fine-tune, and
     specifically **whether the probe beat the from-scratch CNN**;
   - wall-clock time per notebook, end to end, and per training cell;
   - the dataset URLs and cache paths you used;
   - anything in this document that turned out to be wrong.

## 7. Acceptance checks

Run these before handing back:

```bash
# 1. no forbidden frameworks anywhere in the notebooks
grep -rl "import torch\|import tensorflow\|from tensorflow\|tensorflow_datasets" --include='*.ipynb' sessions exercises solutions
#    ^ must print nothing

# 2. both notebooks execute top to bottom from a clean kernel
.venv/bin/jupyter nbconvert --to notebook --execute --inplace sessions/transfer.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace sessions/audio.ipynb

# 3. no datasets or weights staged
git status --short
```

Also confirm by eye: opening cell in house style with an "In this session we will
understand:" list, `# References` and `# Colophon` closing cells, markdown
narrative between code cells, and every long training cell paired with a working
`load_model` path so the notebook can be *taught* without waiting.

Point 2 matters more than it looks: a notebook that only runs in the order you
happened to execute cells is the most common defect in this repo's history.
