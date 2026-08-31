# DL2026 — GPU handoff: Phases 2 and 3

**For:** an agentic coding session on a machine with a CUDA GPU.
**Scope:** exactly three notebooks — `sessions/transfer.ipynb` (new),
`sessions/audio.ipynb` (restructure) and `sessions/flow.ipynb` (extend). Nothing
else.

This document is self-contained; you do not need the conversation it came from.
`DL2026_PLAN.md` in the repo root is the full course plan and the authority on
anything not covered here (its §3 is Phase 2, §4 is Phase 3, §4b is Phase 7).
`CLAUDE.md` has the repository conventions. Read both before starting.

**Phase 7 (`flow.ipynb`) was added to this handoff on 2026-08-31**, having been
planned as local CPU work. Its spec lives in `DL2026_PLAN.md` §4b; §4b of *this*
document covers only what changes because you are running it on a GPU.

---

## 0. Why this work is remote

Two of these notebooks train ImageNet-scale backbones (EfficientNetV2S) on
several thousand images, and the third fits a masked autoregressive flow to
~200k rows in six dimensions. None is feasible on the CPU-only laptop where the
rest of the branch is being assembled — for calibration, `sessions/autoencoders.ipynb`
takes 25 minutes there, at ~50 s/epoch for a two-layer convolutional autoencoder
on MNIST. Everything else in the plan is done: see §5 for the file boundaries.

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
.venv/bin/python download_data.py --list          # what is available
.venv/bin/python download_data.py cub esc50       # ~1.9 GB, the two datasets you need
```

`requirements.txt` was completed in Phase 6 and already includes `librosa`, so you
should not need to install anything beyond the CUDA `jax`.

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
   PIL/NumPy. Cache the parsed arrays under `data/` and skip re-download when the
   cache is present. The reference loader is in §2a below.
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

## 2a. The reference loader

`sessions/finetuning.ipynb` used to hold the house loader for an image dataset
that arrives as a tarball. That notebook has been **removed from this branch**, so
the pattern is reproduced here. (If you want the original in full, it is still in
git: `git show origin/DeepLearning:sessions/finetuning.ipynb`.)

Download and extract, skipping work that is already done:

```python
from pathlib import Path
import tarfile, urllib.request

DATA_DIR = Path('../data')
ARCHIVE = DATA_DIR / 'CUB_200_2011.tgz'
DATASET_DIR = DATA_DIR / 'CUB_200_2011'

DATA_DIR.mkdir(parents=True, exist_ok=True)
if not DATASET_DIR.exists():
    if not ARCHIVE.exists():
        print(f'Downloading {url}')
        urllib.request.urlretrieve(url, ARCHIVE)
    print('Extracting archive...')
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        archive.extractall(path=DATA_DIR, filter='data')
else:
    print('Dataset already available at', DATASET_DIR)
```

Note `filter='data'` on `extractall` — it refuses absolute paths and paths that
escape the destination. The original notebook predates that argument; use it.

Parse to a NumPy array with PIL, preallocating the output:

```python
from PIL import Image
import numpy as np

IMG_SIZE = 224
images = np.empty((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, path in enumerate(paths):
    with Image.open(path) as image:
        image = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    images[i] = np.asarray(image, dtype=np.uint8)

targets = keras.utils.to_categorical(labels, n_classes).astype('float32')
```

Cache `images` and `targets` to `data/` as `.npy` or `.npz` so a kernel restart
does not re-decode 12,000 JPEGs. `.gitignore` already covers both extensions.

Also copy the `plot_history` helper the other Keras notebooks use — two panels,
accuracy and loss, train and validation on each:

```python
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    axes[0].plot(history['accuracy'], label='train')
    axes[0].plot(history['val_accuracy'], label='validation')
    axes[0].legend()
    axes[1].plot(history['loss'], label='train')
    axes[1].plot(history['val_loss'], label='validation')
    axes[1].legend()
    fig.tight_layout()
```

## 3. Phase 2 — new `sessions/transfer.ipynb`

Day 3's in-class transfer-learning session, 2 academic hours (≈100 min) of class
time. It replaces the old hyena re-identification notebook in the teaching slot,
because 256 open-set identities with ~12 images each is the wrong first example.
That notebook has been removed from the branch entirely, so `transfer.ipynb` is
now the *only* transfer-learning session — there is no fallback if it does not
land.

**Dataset: CUB-200-2011.** 11,788 images, 200 bird species, official split
5,994 train / 5,794 validation. Chosen over Flowers/Oxford-Pets because it does
*not* saturate, so the probe-versus-fine-tune gap is visible. Use the official
split from `train_test_split.txt`, not a random one.

`download_data.py` on this branch already fetches it — `python download_data.py cub`
puts it in `data/CUB_200_2011`. If you fetch it yourself, the URL is
`https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1`
(1.07 GiB, verified). **Gotcha:** that link redirects to presigned storage whose
signature covers `GET` only, so a `HEAD` probe returns `403` even though the URL
works. Do not conclude it is dead — check with a ranged `GET` instead.

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

## 4b. Phase 7 — extend `sessions/flow.ipynb` with UCI POWER

**The full specification is `DL2026_PLAN.md` §4b. Read it.** It is detailed and
current, and it supersedes an earlier conditional-AnAge design that was abandoned —
do not resurrect anything involving AnAge, Palmer penguins, autoencoder latent
spaces, or a conditional flow. This section adds only the operational detail.

Day 4 session, 1.5 academic hours. Keep the existing `make_moons` material as the
first half (~45 min) and add UCI POWER — the MAF paper's own benchmark — as the
second half (~30 min).

**The data is already wired up.** `download_data.py` fetches it:

```bash
.venv/bin/python download_data.py maf-benchmarks
```

That pulls Papamakarios's preprocessed datasets (Zenodo record 1161203, CC-BY-4.0,
a single 857 MB `data.tar.gz`) and lands them at **`data/maf/power/`** and
**`data/maf/miniboone/`**. Two things to know about it: the archive's own top-level
directory is called `data`, which would have collided with the repo's, so the
loader strips that component; and it also contains `gas`, `hepmass`, `bsds300`,
`mnist` and `cifar10`, which are filtered out to save ~650 MB. If you need one of
those for an exercise, add it to the entry's `keep` tuple and re-run — but note
that by default the archive is deleted after extraction, so pass
`--keep-archives` if you expect to want the others.

`data/maf` is gitignored.

**Three things that will bite you**, in order of likelihood:

1. **The dequantization noise is not optional.** POWER's measurements are rounded,
   so without adding small uniform noise the flow chases discrete artifacts and the
   log-likelihood diverges upward without bound — you will think you have a great
   model. Make it an explicit, commented step. It is one of the better teaching
   moments in the notebook.
2. **KDE cannot be scored naively at this scale.** POWER is ~2.05M rows, so a 10%
   test split is ~200k points, and `sklearn`'s `KernelDensity` costs
   O(n_train x n_test) — around 4e10 kernel evaluations, which will not finish.
   Score KDE on a fixed random subsample of the test set (a few thousand points)
   and say so in the notebook. The wall-time comparison is then a *lower bound* on
   KDE's real cost, which strengthens the argument rather than weakening it.
3. **Do not promise to reproduce 0.24 nats.** The published MAF(10) figure comes
   from ten layers with the paper's hyperparameter search on the full dataset.
   Report your number against the published one and let the gap be the discussion.

**Housekeeping in the same pass** (all listed in §4b of the plan, all verified):
`flow.ipynb` has no Colophon cell — it is the only session notebook missing one, so
copy it verbatim from a sibling. Its opening cell uses `img/logo.png` where the rest
use `../logo.png`; fix it here, and fix `sessions/jax.ipynb` too since it is the same
one-line change. Add the "In this session we will understand:" intro cell. Add one
sentence explaining that this notebook alone uses FlowJAX and Equinox rather than
Keras, because no Keras flow implementation is worth teaching. `flowjax` is already
in `requirements.txt`.

**GPU note.** FlowJAX runs on JAX, so it picks up the GPU with no code change once
`jax[cuda12]` is installed — but confirm with `jax.devices()` and print it in the
notebook the way the Keras sessions print their backend. If the flow trains in
seconds on your GPU, consider whether the ~200k-row subsample in the plan should be
larger; say what you chose and why.

## 5. Boundaries — do not touch these

The rest of the plan is already done locally. To keep the merge clean, **only**
create `sessions/transfer.ipynb` and edit `sessions/audio.ipynb` and
`sessions/flow.ipynb`.

Specifically, do **not** edit:

- `index.ipynb` — **already rewritten** (Phase 6). It links your two notebooks
  under Day 3, so `sessions/transfer.ipynb` is currently the one dead link on the
  branch; it resolves the moment your work merges. Do not edit the index.
- `requirements.txt` — **already complete**. If you needed something extra,
  report it rather than editing the file.
- `download_data.py` — **already written**, and it fetches CUB, ESC-50 and the MAF
  benchmarks. Use it. If your notebooks do their own downloading (fine, and closer
  to the teaching style), still tell us the URLs and cache paths you used so the
  two stay consistent. The one edit you may make here is adding a dataset to the
  `keep` tuple of the `maf-benchmarks` entry — say so in your report.
- `LOCAL_SETUP.md`, `.gitignore` — done in Phase 6. Note that `.gitignore` now
  covers `data/CUB_200_2011`, `data/ESC-50-master`, `*.keras`, `*.h5` and
  `*.tgz`, so your downloads and checkpoints stay untracked automatically.
- `DL2026_PLAN.md`, `CLAUDE.md`, `sessions/autoencoders.ipynb`, anything under
  `exercises/` or `solutions/`.

`sessions/flow.ipynb` **is now yours** (Phase 7, §4b above) — it used to be on this
list and no longer is.

One warning inherited from the notebook that used to live here: it split
`train_test_split` over *annotation* indices rather than *image* ids, so crops
from one photograph landed on both sides of the split and the reported accuracy
was optimistic. CUB has an official `train_test_split.txt` — use it, and do not
invent a random split.

## 6. Deliverables

1. Branch `DL2026-gpu` pushed, with the three notebooks committed **with outputs**.
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
   - **measured** test log-likelihood in nats on POWER for all four models -
     Gaussian, GMM, KDE and the flow - plus KDE's fit-and-score wall-time and the
     size of the test subsample you scored it on;
   - wall-clock time per notebook, end to end, and per training cell;
   - the dataset URLs and cache paths you used;
   - anything in this document that turned out to be wrong.

## 7. Acceptance checks

Run these before handing back:

```bash
# 1. no forbidden frameworks anywhere in the notebooks
grep -rl "import torch\|import tensorflow\|from tensorflow\|tensorflow_datasets" --include='*.ipynb' sessions exercises solutions
#    ^ must print nothing

# 2. all three notebooks execute top to bottom from a clean kernel
.venv/bin/jupyter nbconvert --to notebook --execute --inplace sessions/transfer.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace sessions/audio.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace sessions/flow.ipynb

# 3. no datasets or weights staged
git status --short
```

Also confirm by eye: opening cell in house style with an "In this session we will
understand:" list, `# References` and `# Colophon` closing cells, markdown
narrative between code cells, and every long training cell paired with a working
`load_model` path so the notebook can be *taught* without waiting.

Point 2 matters more than it looks: a notebook that only runs in the order you
happened to execute cells is the most common defect in this repo's history.
