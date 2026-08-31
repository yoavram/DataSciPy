# Deep Learning 2026 Plan

Implementation plan for reorganizing the *Introduction to Deep Learning* workshop
into a new `DL2026` branch of `yoavram/DataSciPy`, plus coordinated changes in
`yoavram/nanochat`.

Written for an agentic coding session. Work through the phases in order; each
phase ends at a committable state.

---

## 0a. Progress log

Maintained as work lands. Last updated 2026-08-31.

Branch `DL2026` is pushed to `origin` and tracks `origin/DL2026`.

| Phase | Status | Where | Notes |
|---|---|---|---|
| 1 (§2) Branch assembly | **DONE** | local, commit `313621a` | branch `DL2026` created off `origin/DeepLearning` |
| 2 (§3) `sessions/transfer.ipynb` | **DELEGATED** | remote GPU agent | see `DL2026_GPU_HANDOFF.md` |
| 3 (§4) `sessions/audio.ipynb` | **DELEGATED** | remote GPU agent | see `DL2026_GPU_HANDOFF.md` |
| 3b (§4b) `sessions/flow.ipynb` | TODO | local | FlowJax conditional API verified, see below |
| 4 (§5) `sessions/finetuning.ipynb` | **POSTPONED** | — | explicit decision, not this pass |
| 5 (§6) nanochat `transformer_ts` | **POSTPONED** | — | deferred by decision; source notebook stashed out of `origin/master` |
| 6 (§7) index / env / data | **DONE** | local, commit pending | one pending link: `sessions/transfer.ipynb` (Phase 2) |
| 9 (§9) Validation | PARTIAL only | local + remote | full GPU notebook runs depend on Phases 2/3 |

### Phase 1 — what actually happened

Executed as specified. `DeepLearning` was in sync with `origin/DeepLearning` at
`eb5bc51`, so the branch base is unambiguous.

- Brought in `sessions/jax.ipynb` (`origin/amat2025b`),
  `sessions/gamma_regression.ipynb` and `sessions/audio.ipynb` (`origin/master`).
- Removed `exercises/LSTM.ipynb`, `solutions/LSTM.ipynb`.
- `sessions/transformer_ts.ipynb` copied out of `origin/master` to the session
  scratch directory for Phase 5; not committed to `DL2026`.
- Verified none of the three imported notebooks touch `torch` or `tensorflow`.
- `DL2026` was created tracking `origin/DeepLearning`; upstream was unset so a
  stray `git push` cannot land on `DeepLearning`.

`sessions/audio.ipynb` is 34 MB (embedded audio and figure outputs). The blob
already exists in `master`'s history, so this costs nothing new, but it is worth
knowing before restructuring it.

### Phase 6 — what actually happened

Done locally, no GPU needed. Deliverables:

- **`index.ipynb` rewritten.** Title is now *Introduction to Deep Learning*; logo,
  author block and contact lines kept. Four days, each split into explicit
  **Sessions** / **Bonus** / **Homework** subsections so in-class time reads as
  sessions only. New **Setup** section pointing at `README.md` and
  `LOCAL_SETUP.md` plus the `download_data.py` invocation. New **Part II**
  section linking `https://github.com/yoavram/nanochat` with the Day 5–6 topics
  listed but not linked per-notebook (they live in that repo's own index). The
  dead `sessions/density-estimation.ipynb` link is gone; `sessions/flow.ipynb`
  now carries density estimation. Jupyter help / Terminal / GPU / CPU cells kept
  verbatim. 11 cells, validates under `nbformat`.
- **`requirements.txt` completed**: added `librosa`, `corner`, `pillow`,
  `ipywidgets` ordering tidied, and `keras` pinned to `keras>=3`. Verified free of
  `torch` / `tensorflow` / `transformers` / `tensorflow_datasets`.
- **`download_data.py` rewritten for Keras 3.** `import keras` with
  `KERAS_BACKEND` defaulted to `jax`; `urllib.request` instead of `requests` with
  `verify=False` (no new dependency, and TLS verification is no longer disabled);
  `tarfile.extractall(..., filter="data")`. Fetches MNIST, Fashion-MNIST,
  ResNet50 and EfficientNetV2S (no-top) weights, ESC-50, CUB-200-2011,
  SpeechEmotion, Sign-Language, and regenerates `data/penguins.csv`. The 3.2 GB
  hyena archive is `--bonus`-only. Has `--list`, named-item selection, skip-if-
  present, and `--keep-archives`. Smoke-tested: `--list`, `--help`, unknown-name
  rejection, the Keras-cache items, and one real download-and-extract cycle.
- **`LOCAL_SETUP.md` added**: Miniforge + conda/mamba route, why to install these
  packages with `pip` rather than `mamba`, kernel registration, backend check,
  GPU/`jax[cuda12]` notes, data download, and maintainer notes.
- **`.gitignore` reorganized**: added `*.pt` (see correction 3), `*.hd5`, `*.pkl`,
  `*.tgz`, `__pycache__/`, `.venv/`, and the new dataset directories
  (`data/CUB_200_2011`, `data/Dataset`, `data/FashionMNIST`,
  `data/SpeechEmotion-master`, `data/acgan`). Verified no tracked file became
  ignored.
- **`data/penguins.csv` restored** from `origin/master` (see correction 8).

Validation run at this point (§9 items 3 and 4):

- No notebook imports `torch`, `tensorflow`, `transformers`, or
  `tensorflow_datasets`. Clean.
- `index.ipynb`: 41 of 42 local links resolve. The single break is
  `sessions/transfer.ipynb`, which is Phase 2's deliverable and will resolve when
  the GPU branch merges. **This is the one known-dead link on the branch.**

### Corrections to this plan, found while executing it

1. **§4b logo claim is wrong.** `img/logo.png` is not unique to `flow.ipynb` —
   `sessions/K_FFN.ipynb` and `sessions/jax.ipynb` use it too, and
   `sessions/img/logo.png` exists, so all three resolve correctly. This is a
   consistency issue, not a broken link. Normalize all three to `../logo.png` in
   Phase 6 housekeeping rather than treating it as a `flow.ipynb` defect.
   (`sessions/gamma_regression.ipynb` uses `![](../logo.png)` with no `Py4Eng`
   alt text — same pass.)
2. **§7 `download_data.py` is not on this branch.** It exists on `master`,
   `amat2025b` and `torch`, but *not* on `DeepLearning`, so Phase 6 must
   `git checkout origin/master -- download_data.py` before rewriting it. The
   `master` version is 43 lines and opens with `from tensorflow import keras`.
3. **`.gitignore` covers `*.pth` but not `*.pt`.** Untracked PyTorch artifacts
   from the `torch` branch (`data/torch_*.pt`, `data/*_cnn.pt`) are therefore
   visible in `git status` on this branch and could be committed by accident.
   Add `*.pt` in Phase 6.
4. **§7 `requirements.txt` is closer to complete than assumed.** `flowjax` is
   already present. Genuinely missing: `librosa` (`exercises/audio.ipynb`,
   `sessions/audio.ipynb`) and `corner` (`sessions/mle.ipynb`).
5. **§6 nanochat has no `exercises/` or `solutions/` directories.** All its
   notebooks live in the repository root, so Phase 5 creates both directories.
   Repo is at `~/Work/Teaching/nanochat`, branch `main`.
6. **§4b FlowJax conditional API — verified, plan was right.** FlowJax 19.1.0:
   `masked_autoregressive_flow(key, base_dist, cond_dim=1, flow_layers=8,
   nn_width=50, nn_depth=1, transformer=None, invert=True)`;
   `flow.log_prob(x, condition=None)`;
   `flow.sample(key, sample_shape=(), condition=None)`;
   `fit_to_data(key, dist, data, ..., max_epochs=100, max_patience=5,
   batch_size=100, val_prop=0.1, return_best=True)` — for a conditional flow
   `data` is the tuple `(x, condition)`.
7. **Local environment is CPU-only** (keras 3.14.0, jax 0.9.2, backend `jax`,
   `jax.default_backend() == 'cpu'`), which is why Phases 2 and 3 were split out
   to a remote GPU machine.

8. **`data/penguins.csv` was missing and a Day 1 session depended on it.**
   `sessions/gamma_regression.ipynb`, imported from `master` in Phase 1, does
   `pd.read_csv('../data/penguins.csv')`. That file is committed on `master` and
   `amat2025b` but was absent from `DeepLearning`, so the session would have
   failed in class. Restored from `origin/master` (15 KB, 344 rows — exactly
   `palmerpenguins.load_penguins()`). This also explains why `palmerpenguins` is
   in `requirements.txt` while no notebook imports it: the CSV was pre-exported.
   `download_data.py` can now regenerate it.
9. **§1 Day 4 homework lists `exercises/GAN.ipynb`, which does not exist** — not
   on `DL2026`, `master`, `DeepLearning`, `amat2025a/b`, `torch`, `kla2025` or
   `probml`. Only `exercises/ACGAN.ipynb` exists, and the index lists that alone.
   If a plain GAN exercise is wanted it has to be written from scratch.
10. **§1 Day 4 hours do not add up.** The header says 6.5 AH but the sessions sum
    to 4.5 (autoencoders 2 + flow 1.5 + GAN 1). Days 1, 2 and 3 each sum to 6.5
    correctly. Day 4 therefore has 2 AH unaccounted for — either the budget is
    wrong or the day needs more material. **Needs a decision.**
11. **§7's dataset list for `download_data.py` is incomplete.** Auditing every
    `../data/` reference in the notebooks turns up two more downloads the course
    needs: SpeechEmotion (`exercises/audio.ipynb`) and the Sign-Language
    `Dataset.zip` (`exercises/sign-lang.ipynb`). Both are now handled.
12. **The CUB-200-2011 URL answers `403` to `HEAD`.** The Caltech DATA link
    (`https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1`)
    redirects to presigned OSN storage whose signature covers `GET` only, so a
    `HEAD` probe looks broken when the URL is fine. A ranged `GET` returns `206`
    and reports a total of 1,150,585,339 bytes (1.07 GiB). Verified working.
13. **The hyena archive is 3.2 GB** (3,441,352,255 bytes), larger than the plan
    implies for a bonus notebook. It is `--bonus`-only in `download_data.py`.
14. **`requirements.txt` was missing `pillow` too** — `sessions/finetuning.ipynb`
    imports `PIL`, which currently only arrives as a transitive `matplotlib`
    dependency. `optax` and `palmerpenguins` are imported by no notebook
    (`optax` is a FlowJax dependency, `palmerpenguins` regenerates the CSV); both
    kept, as §7 asks.

---

## 0. Context

Two repositories:

- **`yoavram/DataSciPy`** — Days 1–4. New branch `DL2026`, based on `DeepLearning`.
- **`yoavram/nanochat`** (`main`) — Days 5–6. Stays a separate repo, linked from
  the DataSciPy index. It is the single source of truth for sets, recurrent
  models, attention, and LLMs.

We keep the two repos separate, and only reorganize DataSciPy in the new DL2026 branch.

Format: four + two days, 6.5 academic hours per day, 50 minutes per academic hour
(≈5h25m contact per day). **In-class time is sessions only.** Exercises are
homework; do not budget class time for them.

Frameworks: JAX for from-scratch work, Keras 3 on the JAX backend for applied
work. Do not introduce PyTorch or TensorFlow anywhere (see §6 for the
consequences this has on dataset loading).

---

## 1. Target agenda

### Day 1 — Likelihood and generalized linear models (6.5)
| Session | AH |
|---|---|
| `sessions/mle.ipynb` | 2 |
| `sessions/linear_regression.ipynb` | 1.5 |
| `sessions/logistic_regression.ipynb` | 2 |
| `sessions/gamma_regression.ipynb` | 1 |

Bonus: `sessions/robust-regression.ipynb`
Homework: `exercises/linear-anage.ipynb`, `exercises/ridge.ipynb`

### Day 2 — From GLM to neural networks in JAX (6.5)
| Session | AH |
|---|---|
| `sessions/softmax_regression.ipynb` | 1.5 |
| `sessions/jax.ipynb` | 1.5 |
| `sessions/FFN.ipynb` | 3.5 |

Homework: generalize `feed_forward` / `back_propagation` to arbitrary depth;
`exercises/sign-lang.ipynb` Ex 1

### Day 3 — Keras, convolutions, and transfer learning (6.5)
| Session | AH |
|---|---|
| `sessions/K_FFN.ipynb` | 0.5 |
| `sessions/K_CNN.ipynb` | 2 |
| `sessions/functional_keras.ipynb` | 0.5 |
| `sessions/pretrained.ipynb` | 0.5 |
| `sessions/transfer.ipynb` **(new)** | 2 |
| `sessions/audio.ipynb` (restructured) | 1 |

Homework: `exercises/CNN.ipynb`, `exercises/sign-lang.ipynb` Ex 2,
`sessions/CNN_timeseries.ipynb`, `exercises/audio.ipynb`, `exercises/transfer_learning.ipynb`,

`autoencoders`, `finetuning` and `transfer` all build models with `keras.Input` + `keras.Model` and none use
`Sequential` so `functional_keras` is critical for these sessions.

### Day 4 — Representations and densities (6.5)
| Session | AH |
|---|---|
| `sessions/autoencoders.ipynb` | 2 |
| `sessions/flow.ipynb` | 1.5 |
| `sessions/GAN.ipynb` | 1 |

Homework: `exercises/GAN.ipynb`, `exercises/ACGAN.ipynb`

Framing to preserve in the narrative: the autoencoder learns a representation with no density; the flow learns a density you can both evaluate and sample; the GAN learns to sample without ever writing down a density. 
Do **not** label autoencoders as generative models.

### Day 5 — Sets, sequences, attention (nanochat) (6.0)
| Session | AH |
|---|---|
| `sets.ipynb` + `set-transformer.ipynb` | 2 |
| `RNN.ipynb` | 2 |
| `GRU.ipynb` | 0.5 |
| `text-transformer.ipynb` | 2 |

Homework: `exercises/LSTM.ipynb`, `exercises/transformer_ts.ipynb` **(new)**

### Day 6 — Building a chat model (nanochat) (6.5)
| Session | AH |
|---|---|
| `bpe-tokenizer.ipynb` (fast pass) | 1 |
| `nanochat.ipynb` | 2 |
| `nanochat-sft.ipynb` | 1 |
| `nanochat-grpo.ipynb` | 1 |
| `minisweagent.ipynb` | 1.5 |

---

## 2. Phase 1 — Branch assembly  ✅ DONE (commit `313621a`)

Base on `DeepLearning`. It is the newest deep-learning content (May 2026), it
already contains `flow`, `finetuning`, `autoencoders`, `functional_keras`, and
`transfer_learning`, and — conveniently — it already lacks every notebook that
now belongs to nanochat.

```bash
git fetch origin
git checkout -b DL2026 origin/DeepLearning
```

**Bring in missing sessions:**

```bash
git checkout origin/amat2025b -- sessions/jax.ipynb
git checkout origin/master    -- sessions/gamma_regression.ipynb sessions/audio.ipynb
```

**Infrastructure.** The `DeepLearning` branch has
`.gitignore`, `README.md`, `logo.png`, `requirements.txt`, and a committed
`data/` directory.

**Remove material that now lives in nanochat:**

```bash
git rm exercises/LSTM.ipynb solutions/LSTM.ipynb
```

Before deleting, copy `sessions/transformer_ts.ipynb` out of `origin/master` to a
scratch location — Phase 5 converts it into a nanochat exercise. It must not be
committed to `DL2026`.

Commit as `DL2026: assemble branch from DeepLearning + jax + gamma + audio`.

---

## 3. Phase 2 — New notebook: `sessions/transfer.ipynb`  ➡️ DELEGATED to remote GPU agent (`DL2026_GPU_HANDOFF.md`)

Day 3's transfer-learning session. Replaces the hyena notebook in the in-class
slot because hyena re-ID is the wrong task for a first transfer-learning demo
(256 classes, ~12 images each, open-set, viewpoint-split identities).

**Dataset: CUB-200-2011.** 11,788 images, 200 bird species, official split of
5,994 train / 5,794 test. Chosen because it is ecologically meaningful, has a
large published baseline literature, and — unlike Flowers or Oxford Pets — does
not saturate, so the probe-versus-fine-tune gap is actually visible.

**Loading (important):** download and extract the tarball directly, then parse
with PIL + NumPy, mirroring the loader already in `finetuning.ipynb`. Do **not**
use `tensorflow_datasets` (drags in TensorFlow) or HF `datasets` unless it can be
installed without a deep-learning framework. Cache the extracted arrays under
`data/` and skip re-download if present.

**Structure:**

1. Intro cell: what transfer learning is, why a 60M-parameter model cannot be
   trained on 6,000 images.
2. Load, resize, visualize a grid of species.
3. Backbone: `EfficientNetV2S`, `weights='imagenet'`, `include_top=False`,
   `pooling='avg'`.
4. **Protocol A — linear probe.** Run the frozen backbone once over the whole
   dataset and cache the 1280-d embeddings. Then train a Keras `Dense` softmax
   head on the cached features. This trains in seconds, so run it live with a
   loss curve. The head is Keras, not scikit-learn — it reuses Day 2's softmax
   model directly and keeps everything in one framework.
5. **Protocol B — full fine-tune.** Unfreeze the top of the backbone (batch-norm
   layers stay frozen), low LR, cosine decay with warmup, label smoothing. Ship
   a checkpoint; do not expect this to run live.
6. Comparison table: probe vs fine-tune, top-1 and top-5.
7. Discussion cell covering:
   - Expected range: published ImageNet-pretrained fine-tuning baselines on CUB
     sit near 85% (ResNet-50) and 73–77% for older backbones. Fine-tuning beats
     frozen feature extraction by roughly 2–10%.
   - Training the same architecture from scratch on CUB collapses (10.9% vs
     57.0% with pretraining in the classic AlexNet-era comparison) — this is the
     one-number motivation for the whole session.
   - **Caveat to state, not hide:** 59 of ImageNet's 1000 classes are already
     bird categories overlapping CUB, which is why this transfers so well. This
     is the honest reason the hyena case study is harder.
   - Probe vs fine-tune is not settled in favour of fine-tuning. Cite Kumar et
     al. 2022 (arXiv:2202.10054): fine-tuning wins in-distribution by ~2% but
     loses out-of-distribution by ~7%, because it distorts good pretrained
     features under large shift; linear-probe-then-fine-tune does best. Note
     that the freeze-then-unfreeze staging used here *is* LP-FT.
   - Report a range over 2 seeds, not a single number. Run-to-run variance from
     head initialization and batch ordering is real and documented.

**References to include:** Kumar et al. 2022 (arXiv:2202.10054); Branson et al.
2014 (arXiv:1406.2952); Wah et al. 2011 (CUB-200-2011); Keras transfer learning
guide.

---

## 4. Phase 3 — Restructure `sessions/audio.ipynb`  ➡️ DELEGATED to remote GPU agent (`DL2026_GPU_HANDOFF.md`)

Currently an EchoNet-style CNN trained from scratch on ESC-50 for 150 epochs.
Extend it into a three-way comparison so it mirrors `transfer` structure, on a different modality.

**New structure:**

1. ESC-50 data and mel-spectrogram preparation (existing material, keep).
2. **Baseline** — from-scratch EchoNet CNN (existing material; provide a
   checkpoint, do not train live).
3. **Linear probe** — feed the mel-spectrogram to an ImageNet-pretrained
   `EfficientNetV2S` or `ResNet50` as a 3-channel image, frozen, cached
   embeddings, Keras softmax head.
4. **Fine-tune** — unfreeze the top, low LR, as in `transfer.ipynb`.
5. Comparison and discussion.

**Channel construction:** do not replicate a single grayscale spectrogram across
three channels. Follow Palanisamy et al. and use mel-spectrograms computed with
*different window sizes and hop lengths* in each channel, 128 mel bins,
log-scaled. Cheap to implement and it makes the point that "make it look like an
RGB image" has better and worse answers.

**Evidence for the discussion cell:** Palanisamy et al. 2020
(arXiv:2007.11154, IJCNN) — an ensemble of ImageNet-pretrained DenseNet reaches
92.89% on ESC-50 and 87.42% on UrbanSound8K, state of the art at the time; and
for a fixed architecture, pretrained weights beat random initialization. Earlier
ImageNet-weighted ResNet work passed 91.5% on ESC-50 by splitting the
spectrogram across the frequency axis into three channels.

**No ensemble** — but keep the reason. The ensemble helped precisely because of
run-to-run variance, so report a range over seeds instead.

**Honest expectation to verify before finalizing:** ImageNet features are a
weaker prior for environmental audio than AudioSet features would be, and 92.89%
is a full fine-tune of an ensemble, not a frozen probe. The frozen probe may
land below the from-scratch CNN. Run it and write the discussion around whatever
actually happens — "the domain gap is real and pretrained is not automatically
better" is a legitimate and more valuable lesson than a rigged win. Record the
measured numbers in the notebook.

---

## 4b. Phase 3b — Add a real example to `sessions/flow.ipynb`  ⬜ TODO (local)

The notebook is currently toy-only: `make_moons` throughout (2,500 points), KDE
and GMM baselines, then a masked autoregressive flow with rational-quadratic
splines that trains in about two seconds. The exercises are knob-turning. Keep
all of this as the first half — it is a good, compact motivation — and add a
real conditional example as the second half.

**Dataset: AnAge** (`data/anage_data.txt`, already in the repo and already used
in `exercises/linear-anage.ipynb` on Day 1). Log body mass and log maximum
lifespan.

**The demo:** fit a *conditional* flow to p(log lifespan | log body mass) with
`cond_dim=1`, and compare it against the Day 1 linear regression on the same
data. The regression gives E[y|x] with constant Gaussian scatter. The flow gives
the whole predictive distribution, which for life-history data is
heteroscedastic and right-skewed, and may be multimodal at a given body mass —
bats and rodents sit close together on the x-axis and far apart on the y-axis.

Plot conditional density slices at three or four body masses, overlaid with the
regression's fitted normal at the same points. That contrast is the point of the
session.

This makes AnAge the third treatment of one dataset across the workshop —
regression on Day 1, robust regression as the Day 1 bonus that handles outliers
by widening the tails, and a flow on Day 4 that stops assuming a shape at all.
Preserve that thread explicitly in the narrative; it is worth more than a new
dataset would be.

**Verify before building:** confirm the conditional API against the installed
FlowJAX version — `masked_autoregressive_flow(..., cond_dim=1)`,
`flow.log_prob(x, condition)`, `flow.sample(key, shape, condition=...)`. This is
the one part of the design that should be checked rather than assumed.

**Literature to cite,** so students see this is a standard use of flows and not
an improvisation:

- Trippe & Turner 2018 (arXiv:1802.04908), *Conditional Density Estimation with
  Bayesian Normalising Flows* — flows as a conditional likelihood model,
  benchmarked on small UCI **regression** datasets against mixture density
  networks and Bayesian neural nets with homoscedastic Gaussian likelihoods.
  The closest published analogue to this demo.
- Winkler et al. 2019, *Learning Likelihoods with Conditional Normalizing Flows*
  — canonical conditional-flow reference.
- Rothfuss et al., *Conditional density estimation with neural networks: best
  practices and benchmarks*.
- Papamakarios et al. 2021, JMLR 22(57) — review, includes conditional flows.

**Caveats to state in the notebook, not hide:**

1. Flows do not automatically win here. On small tabular data with a 1D
   response, mixture density networks are strong competitors, and Trippe &
   Turner reach state of the art on *some* of their six benchmarks, not all. If
   time allows, add a small MDN as a third comparison. Report the measured
   result whichever way it falls.
2. Species are not independent samples. AnAge rows share phylogeny, so this is a
   density over extant species as sampled, not over a biological population. The
   Day 1 regression already makes this assumption silently; naming it here is a
   free correction.
3. With a few thousand points a spline flow can overfit. Keep the validation
   curve visible and let `max_patience` do its job — watching a flow overfit is
   itself instructive.

**Housekeeping in the same pass:**

- Logo path is `img/logo.png`; every other session uses `../logo.png`. Fix.
- No "In this session we will understand:" intro cell. Add one.
- No Colophon cell — this is the only session notebook missing the CC BY-SA
  block. Add it, copied verbatim from a sibling notebook.
- Add one sentence acknowledging that this notebook uses FlowJAX and Equinox
  rather than Keras or raw JAX. It is the only one that does, and students will
  otherwise wonder why the API changed. The justification: no Keras flow
  implementation is worth teaching.
- Replace at least two of the four knob-turning exercises with questions about
  the conditional model (e.g. predict the conditional median and a 90% interval
  at a given body mass; compare against the regression's interval).

Session budget is unchanged at 1.5 AH: roughly 45 minutes for the existing
two-moons material and 30 for the AnAge conditional example.

Do **not** add a Palmer penguins example.

---

## 5. Phase 4 — POSTPONED, DO NOT DO THIS NOW - Rework `sessions/finetuning.ipynb` as a bonus case study

The hyena notebook stops being the Day 3 session and becomes a Day 3 bonus /
case study whose job is to motivate metric learning. Current recorded results:
3,129 images, 256 identities, 2,816 train / 313 val; head-only 5 epochs → 29.7%
val; +50 epochs fine-tuning → 54.0% val (loss 2.06); ~45 min total runtime.

**Fixes required:**

1. **The split leaks.** `train_test_split` runs over annotation indices, not
   image ids. Multiple annotations share an `image_id`, so crops from the same
   photograph can land on both sides. Split by `image_id` at minimum, and by
   encounter or date if the metadata supports it — same-day sightings are
   near-duplicates. The 54% is optimistic by an unknown amount. Keep this
   visible in the narrative as a teaching point; it is the failure mode that
   makes wildlife re-ID papers wrong.
2. **Metrics.** 313 validation images across 256 classes is ~1.2 per class, so
   top-1 has a standard error near 2.8% and many classes have no validation
   example at all. Report rank-1 and rank-5; a CMC curve is standard in re-ID
   and more informative.
3. **Viewpoint.** Check whether `metadata['annotations'][0]` carries a
   `viewpoint` field (Wild-Me COCO exports usually do). If present, add a
   discussion cell: a hyena's left and right flanks carry different spot
   patterns, so one label covers two visual classes and a 256-way softmax cannot
   represent that. Report accuracy split by viewpoint.
4. **No horizontal flip** in any augmentation here — it manufactures a fake
   opposite flank. Worth an explicit cell: the default augmentation pipeline is
   actively wrong for this data.
5. **Note it is undertrained, not overfit** — validation loss was still falling
   monotonically at epoch 50.
6. Keep the "Moving forward" list. Add a marker that the metric-learning
   follow-up (ArcFace + cosine nearest-neighbour retrieval, open-set split with
   held-out individuals) is planned but **out of scope for this branch**.

Do not implement metric learning. Do not implement the cosine-NN retrieval
evaluation. Those are deferred.

---

## 6. Phase 5 — nanochat changes  ⏸️ POSTPONED (independent of Phases 2–3; do later)

1. **New `exercises/transformer_ts.ipynb` + `solutions/transformer_ts.ipynb`.**
   Port `sessions/transformer_ts.ipynb` from `DataSciPy@master` (Keras, FordA
   time-series classifier with self-attention). Convert from a walkthrough
   session into an assignment: strip the completed self-attention and classifier
   construction into TODOs, keep data loading and evaluation intact, write a
   separate solution notebook. This is genuine authoring work, not a file move.
   Placing it after `text-transformer.ipynb` is deliberate — same architecture,
   different data, Keras rather than JAX.

---

## 7. Phase 6 — Index, environment, and data  ✅ DONE

### `index.ipynb` (DataSciPy `DL2026`)
Rewrite to the agenda in §1. Requirements:

- Title: *Introduction to Deep Learning*. Keep the logo cell, author block, and
  `python.yoavram.com` / email lines.
- Four days of sessions, each day with a **Bonus** subsection and a **Homework**
  subsection, clearly separated — in-class time is sessions only.
- A Part II section linking to `https://github.com/yoavram/nanochat`.
- Delete the dead link to `sessions/density-estimation.ipynb`, which does not
  exist on this branch. Normalizing flows are covered by `sessions/flow.ipynb`.
- Keep the Jupyter help, terminal, GPU and CPU monitoring cells.
- Add a Setup section (see below).

### Environment
Students use miniforge + pip/mamba; `pixi.toml` is for local development only.
Keep `requirements.txt` as the student-facing file and make sure it is complete.

- Add `flowjax` (needed by `flow.ipynb`; present in the `DeepLearning`
  `requirements.txt`, absent from every `pixi.toml`).
- Add `librosa` for `audio.ipynb`.
- Verify `keras>=3`, `jax`, `optax`, `statsmodels`, `palmerpenguins`,
  `scikit-image`, `imageio` are all present.
- Do **not** add `torch`, `tensorflow`, `transformers`, or `tensorflow_datasets`.
- Write the miniforge + pip instructions into `LOCAL_SETUP.md` and summarize in
  the index.

### Data
`download_data.py` was inherited from `master` and opens with
`from tensorflow import keras`. Rewrite for Keras 3 (`import keras`, with the
JAX backend set via `KERAS_BACKEND`). Extend it to fetch:

- MNIST / Fashion-MNIST, ResNet50 and EfficientNetV2S weights
- ESC-50
- CUB-200-2011
- the hyena COCO archive (bonus notebook)

The `DeepLearning` branch commits `data/` into the repo. Keep the existing small
committed files, but route everything new through `download_data.py` and
`.gitignore` it. Do not commit CUB or ESC-50.

---

## 8. Notebook conventions

Match the existing house style exactly:

- Opening cell: `![Py4Eng](../logo.png)`, `# Title`, `## Yoav Ram`, then an
  "In this session we will understand:" bullet list.
- Closing cells: `# References` (links, papers) then `# Colophon` with the
  CC BY-SA 4.0 notice and the Python logo image, copied verbatim from an
  existing notebook.
- `%matplotlib inline`; fixed `SEED`; print Keras version and backend at import.
- Keep training loops and plots explicit; reuse the existing `plot_history`
  helper.
- Markdown narrative between every code cell — no bare code runs.
- Save checkpoints under `data/` with descriptive names.

---

## 9. Validation  ⚠️ only partially doable locally — items 1, 2 and 5 need the GPU work

1. Run every Day 1–4 notebook end to end on GPU. Record wall-clock time per
   notebook in a scratch file.
2. Flag any notebook whose in-class runtime exceeds its allotted AH. Every
   long-running training cell must have a provided checkpoint and a documented
   load path so the notebook can be taught without waiting.
3. Confirm no notebook imports `torch`, `tensorflow`, or `transformers`.
4. Check every link in `index.ipynb` resolves to a file that exists on the
   branch.
5. Report the measured accuracy numbers for `transfer.ipynb`, the restructured
   `audio.ipynb`, and the fixed `finetuning.ipynb` — especially whether the
   audio probe beats the from-scratch CNN, and how much the hyena number drops
   once the split leak is fixed. Both may change what gets said in class.

---

## 10. Out of scope

- Metric learning (ArcFace, contrastive, cosine-NN retrieval) — planned, not now.
- Any PyTorch migration. The `torch` branch stays where it is.
- `sessions/augmentation.ipynb` on the `torch` branch is a 2-cell stub; ignore it.
- KerasHub / Whisper audio embeddings — a possible future bonus, not this pass.
- The legacy branches (`kti2018`, `kti2020`, `amat2019`, `trees`, `lam2020`,
  `lam2021`, `intuit`, `landa`, `IDC2018`) and their `reinforcement.ipynb` /
  `FFN_GenModel.ipynb` / `TF_CNN.ipynb` material.
