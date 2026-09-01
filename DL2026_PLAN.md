# Deep Learning 2026 Plan

Implementation plan for reorganizing the *Introduction to Deep Learning* workshop
into a new `DL2026` branch of `yoavram/DataSciPy`, plus coordinated changes in
`yoavram/nanochat`.

Written for an agentic coding session. Work through the phases in order; each
phase ends at a committable state.

---

## 0a. Progress log

Maintained as work lands. Last updated 2026-09-01.

Branch `DL2026` is pushed to `origin` and tracks `origin/DL2026`.

**Phase numbering.** The old "Phase 3b" is now **Phase 7**, and the autoencoders
work added below is **Phase 8**. Section numbers (§2, §3, §4b, §12, …) are frozen
document anchors and do *not* track phase numbers — `DL2026_GPU_HANDOFF.md` cites
them, and the Phase 1 and Phase 6 commit messages are already in the history.
Read the status table as the index of what to do.

| Phase | Status | Where | Notes |
|---|---|---|---|
| 1 (§2) Branch assembly | **DONE** | local, `313621a` | branch `DL2026` created off `origin/DeepLearning` |
| 2 (§3) `sessions/transfer.ipynb` | **DONE** | GPU box, `DL2026-gpu` `20da46a`+`1ae6a4d` | probe 67.4-67.9%, fine-tune 76.6-76.8% top-1; see below |
| 3 (§4) `sessions/audio.ipynb` | **IN PROGRESS** | GPU box, `DL2026-gpu` | not started; four pre-existing defects surveyed, see below |
| 4 (§5) `sessions/finetuning.ipynb` | **DROPPED** | — | notebook deleted from the branch; §5 is void |
| 5 (§6) nanochat `transformer_ts` | **POSTPONED** | — | deferred by decision; source notebook stashed out of `origin/master` |
| 6 (§7) index / env / data | **DONE** | local, `e568f74` | the `sessions/transfer.ipynb` link now resolves |
| 7 (§4b) `sessions/flow.ipynb` | **DONE** | GPU box, `DL2026-gpu` `c044755` | UCI POWER added; flow +0.350 nats against a −7.742 Gaussian, see below |
| 8 (§12) `sessions/autoencoders.ipynb` | **DONE** | local | grew past the original spec: label-efficiency sweep, see §12 |
| 9 (§9) Validation | PARTIAL only | local + remote | local checks pass; Phases 2 and 7 verified on GPU, Phase 3 outstanding |

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
  the GPU branch merges. ~~**This is the one known-dead link on the branch.**~~
  **Resolved 2026-09-01:** `sessions/transfer.ipynb` exists on `DL2026-gpu`.

### Cleanup pass, 2026-08-31

Decided after Phase 6 and applied on top of it:

- **`sessions/finetuning.ipynb` deleted** and Day 3's Bonus subsection removed
  from the index. §5 is void; see that section. `download_data.py` lost the hyena
  entry and, with it, the `--bonus` flag, which had no other members. The loader
  pattern the notebook carried is preserved in `DL2026_GPU_HANDOFF.md` §2a so the
  Phase 2 work does not lose it. 6.4 GB of hyena data (`data/hyena` plus the
  `hyena.coco.tar.gz` tarball) is still on the local disk and can be deleted.
- **`exercises/GAN.ipynb` removed from the §1 Day 4 homework list** — it never
  existed (correction 9).
- **Removed from the working tree:** the empty `finetuning_torch.ipynb`, and
  `data/poker-hand-testing.data` / `data/poker-hand-training-true.data`, which
  belong to nanochat (verified byte-identical to the copies there before
  deleting).
- **`autoencoders-plan.md` and `density_plan.md` reviewed. Both are largely
  spent.** `density_plan.md` is what turned the old `density-estimation.ipynb`
  into today's `sessions/flow.ipynb`; §4b here is the next step past it.
  `autoencoders-plan.md` has *also* mostly been carried out already, against what
  its own text implies: `sessions/autoencoders.ipynb` now runs intro → dense
  baseline → bottleneck sweep → 2D latent space with a labelled scatter →
  convolutional autoencoder → structured denoising (σ, warm-started from the conv
  model) → exercises → references → colophon, 28 cells. The indexing bug the plan
  flags is already fixed (`X_test.shape[0]`), and the unused `pickle` import and
  one-hot labels it complains about are gone.

  What is genuinely left from that plan is small: **latent interpolation**
  (encode two digits, decode along the line between them) and **reconstruction
  diagnostics** (best and worst reconstructions by per-image loss), plus its
  optional extensions — anomaly detection via reconstruction error, encoder
  features for a downstream classifier, or a VAE teaser. That is perhaps 0.5 AH
  of material, so **it does not on its own close Day 4's 2 AH gap** (correction
  10). Both `autoencoders-plan.md` and `density_plan.md` can be archived or
  deleted; only these leftovers are worth carrying forward.

### Phase 2 — what actually happened (2026-09-01, GPU box)

Done on branch `DL2026-gpu` off `DL2026`, commits `20da46a` (the notebook) and
`1ae6a4d` (a correction to its discussion). Executed end to end from a clean
kernel in 8m30s and committed with outputs; 44 cells, 4 figures, 1.8 MB.

**Measured, official 5,994/5,794 split, seeds 23 and 24:**

| protocol | top-1 | top-5 |
|---|---|---|
| linear probe | 67.36-67.88% | 90.94-91.03% |
| fine-tune | 76.58-76.80% | 92.79-92.82% |

The +8.7 to +9.4 point gain sits at the top of §3's predicted 2-10% range, and
the seed spread is ~0.5 points, so the gap is unambiguous. Probe trains in 9s on
cached 1280-d embeddings, so it runs live in class as §3 wanted; the fine-tune is
~18s/epoch on an RTX A4000, ~3 min per seed, with a checkpoint and `load_model`
path. Built as LP-FT: head copied from the trained probe, only `block6*`/`top_*`
unfrozen, batch-norm frozen, one epoch of warmup into cosine decay at peak 1e-4,
label smoothing 0.1.

**§3's discussion checklist is covered**, including the 59-overlapping-ImageNet-
bird-classes caveat and the Kumar et al. LP-FT argument. Two claims in it were
measured rather than quoted, and one of those overturned an assumption:

- A horizontal flip is worth **+0.4 points** (76.96% vs 76.56%), so augmentation
  is not where the missing accuracy lives. Left out of the notebook; it is an
  exercise.
- **Unfreeze depth saturates.** `top_*` alone 68.62%, `block6*`+`top_*` 76.80%,
  `block5*` onward 78.60%, `block4*` onward 78.84%, everything 79.60%. So a full
  unfreeze buys 2.8 points over the notebook's choice and still lands ~5 points
  short of the published ~85%. The first draft of the discussion asserted the gap
  was "mostly about unfreezing more of the backbone"; that was wrong and is fixed
  in `1ae6a4d`. The notebook now carries this table and names resolution
  (224 vs 384), schedule length and augmentation recipe as *unmeasured*
  candidates, explicitly labelled as such.

Worth keeping: unfreezing `top_*` alone barely beats the frozen probe (68.62% vs
67.36%). The final 1x1 convolution contributes almost nothing on its own.

**Environment on the GPU box** — three deviations from `DL2026_GPU_HANDOFF.md` §1:

- There was no `.venv`; system Python is 3.10/3.11, so one was built with `uv`
  at 3.12.13, then `requirements.txt` + `jax[cuda12]`. `requirements.txt` not edited.
- **`~/.keras/keras.json` was set to `"backend": "tensorflow"`**, not `jax` as
  `CLAUDE.md` claims. Keras would not import at all (no TensorFlow installed).
  Flipped to `jax`; old file kept at `~/.keras/keras.json.bak-tensorflow`.
- Versions run ahead of the handoff's reference: keras 3.15.1, jax 0.11.1,
  backend `jax`, `gpu`, `[CudaDevice(id=0), CudaDevice(id=1)]` (2x RTX A4000
  16 GB). No code changes were needed for either bump.

**Open items from this phase:**

1. **`data/attributes.txt` — fixed.** The CUB tarball carries it at its root
   *next to* `CUB_200_2011/`, so extraction dropped it into `data/`, where it was
   untracked and not gitignored. Nothing in the course uses CUB attribute data.
   `download_data.py`'s `cub` entry now sets `keep=("CUB_200_2011",)`, using the
   filtering machinery the `maf-benchmarks` entry already relied on; with
   `strip=0` the stray root file fails the prefix test and the dataset tree
   extracts unchanged. Verified against a synthetic tarball mirroring the real
   layout. This edit goes beyond `DL2026_GPU_HANDOFF.md` §5, which permits editing
   only the `maf-benchmarks` `keep` tuple — **done on Yoav's explicit
   instruction**, so §5 should be read as superseded on this point.
   Note that the *already extracted* `data/CUB_200_2011/attributes/` (70 MB) is
   untouched by this and is also unused; it can be deleted freely.
2. **Disk on the GPU box.** The two regenerated CUB caches
   (`cub_images_224.npy` 1.7 GB, `cub_effnetv2s_embeddings.npy` 58 MB) were
   **deleted** to reclaim space; the notebook rebuilds both in ~60s (38s decode,
   21s embed). `data/CUB_200_2011` (1.2 GB) is kept, since re-fetching it is a
   1.1 GB download. That leaves 27 GB free, against an 857 MB MAF download for
   Phase 7.
3. Checkpoints are gitignored and live at `~/Work/Teaching/DataSciPy/data/` on the
   GPU box: `cub_effnetv2s_probe.keras` (3.1 MB),
   `cub_effnetv2s_finetune.keras` (196 MB), plus both `_history.p` files.

### Phase 3 — survey before starting (2026-09-01)

Not started. Reading `sessions/audio.ipynb` as it stands turned up four defects
that predate the restructure and that §4 does not mention. The first one changes
what the from-scratch baseline number means:

1. **The train/validation split leaks.** `model.fit(..., validation_split=0.1)`
   splits over *segments*, but segments are ~1s overlapping windows cut from the
   same 5s clip, so windows of one recording land on both sides. The reported
   ~55% is optimistic. ESC-50 ships 5 official folds; the restructure should split
   on `fold` and say which it used. This is the same defect the handoff's §5
   warns about for the deleted hyena notebook.
2. **The history plot is already broken on Keras 3.** It reads `history['acc']`
   and `history['val_acc']`; these have been `accuracy`/`val_accuracy` since
   Keras 2.3, so the cell raises `KeyError` as committed.
3. Saves to `../data/keras_esc50_model.h5` — legacy format; the branch idiom is
   `.keras`.
4. Uses *test* terminology throughout, against the branch-wide sweep to
   *validation*.

None of this changes §4's spec, but fixing (1) will move the from-scratch
baseline, which is the number the probe gets compared against.

### Phase 7 — what actually happened (2026-09-01, GPU box)

Done on `DL2026-gpu`, commit `c044755`. `sessions/flow.ipynb` went from 21 to 49
cells and executes top to bottom from a clean kernel in **2m59s** on an RTX A4000,
committed with outputs (2.1 MB, 10 figures). The `make_moons` material is kept
intact as the first half; UCI POWER is the second.

**Measured test log-likelihood on POWER**, all four models on the same 200,000-row
training subsample:

| model | nats | scored on |
|---|---|---|
| Gaussian | **−7.742** | 204,928 points |
| GMM, 50 components | **−0.123** | 204,928 points |
| KDE, bandwidth 0.08 | **−0.783** | 5,000-point subsample |
| MAF, 8 spline layers | **+0.350** | 204,928 points |

The Gaussian reproduces the MAF paper's published −7.74 exactly, which is the
check that the Appendix D preprocessing is right. KDE fit-and-score cost **32.6 s**
for 5,000 test points; the full test set would take ~22 minutes, so it is scored
on a subsample and the notebook says so.

Timings: flow training 36 s (38 epochs, early-stopped), flow scoring of all
204,928 test points 7.7 s, GMM sweep over 5/10/20/50 components 48 s. Nothing
needs a checkpoint — the whole second half runs live in class.

**§4b's warning 3 inverted, and this is now the discussion's centrepiece.** We
did not fall short of the published MAF(10) 0.24 nats, we exceeded it at 0.350.
The reason is legitimate rather than a fluke: the notebook's transformer is a
*rational-quadratic spline* (Durkan et al. 2019), two years newer and strictly
more expressive than the affine transformer MAF used, so this is a later model
run with less data and no hyperparameter search. It lands between MAF (0.24) and
NAF (0.62). The notebook says exactly this instead of claiming a reproduction.

**Everything asserted as a mechanism is measured**, per the Phase 2 convention:

- **The 200k subsample is a real handicap, and it is imposed for KDE's benefit.**
  The identical flow on all 1,659,917 training rows scores **0.477 nats in 244 s**.
  We kept 200k anyway so all four models see the same data. Stated with both
  numbers in the notebook.
- **KDE's cost is linear in the training set**, as the O(n_train × n_test) claim
  predicts: 4.9 / 9.2 / 16.9 / 33.0 s to fit and score 5,000 test points at
  25k / 50k / 100k / 200k training rows. Quoted in the discussion; turned into
  exercise 4.
- **A 100-component GMM reaches +0.014 nats in 94 s** — still climbing, still
  behind the flow, and too slow to run live, so it is quoted rather than fitted
  in the notebook.
- Two consecutive runs of the committed notebook reproduced 38 epochs and +0.350
  exactly; an earlier variant stopped at 46 epochs and scored +0.368. The
  notebook warns that the second decimal can move and why.

**Exercises.** Three of the four knob-turning exercises were replaced, one more
than §4b asked for: (3) drop the dequantization noise and watch the likelihood
diverge, (4) measure the KDE-versus-flow cost crossover, (5) MINIBOONE, 36,488
rows in 43 dimensions. The `flow_layers`/`nn_width`/`knots` and KDE-bandwidth
exercises are kept.

**Housekeeping**, all of §4b's list: house-style opening cell with the "In this
session we will understand:" bullets, the sentence explaining why this is the one
notebook on FlowJAX and Equinox rather than Keras, the missing Colophon, and the
logo path. The logo fix went to all four notebooks correction 1 identified —
`flow.ipynb`, `jax.ipynb`, `CNN_timeseries.ipynb` (which had no logo at all) and
`gamma_regression.ipynb` (which had no alt text). References gained Durkan et al.
2019, Papamakarios et al. 2021 JMLR, Rezende & Mohamed 2015 and the Zenodo record.

**Data.** `download_data.py maf-benchmarks` worked exactly as written — 857 MB
from Zenodo record 1161203 in ~20 minutes, filtered down to 138 MB at
`data/maf/power/` and `data/maf/miniboone/`, correctly gitignored. `download_data.py`
was **not** edited; the `keep` tuple did not need changing. The archive's `power`
entry is `data.npy`, 2,049,280 × 8 float64; the eight columns and the two the
paper drops are documented in a table in the notebook.

**Not done, and outside this phase:** Day 4 is still 2 AH short (correction 10).
Phase 7 did not change that — the flow session stays at its budgeted 1.5 AH.

### Corrections to this plan, found while executing it

1. **§4b's logo claim needed refining.** In the *opening* cell, `img/logo.png` is
   used by `sessions/flow.ipynb` and `sessions/jax.ipynb`; `CNN_timeseries.ipynb`
   has no logo image at all; everything else uses `../logo.png`.
   `sessions/K_FFN.ipynb` matches a naive grep only because of a Keras logo in its
   body. `sessions/img/logo.png` exists, so nothing is actually broken — this is a
   consistency issue, handled in the Phase 7 housekeeping list.
   (`sessions/gamma_regression.ipynb` uses `![](../logo.png)` with no `Py4Eng` alt
   text — same pass.)
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
9. **§1 Day 4 homework listed `exercises/GAN.ipynb`, which does not exist** — not
   on `DL2026`, `master`, `DeepLearning`, `amat2025a/b`, `torch`, `kla2025` or
   `probml`. Only `exercises/ACGAN.ipynb` exists. **Resolved:** §1 and the index
   now list ACGAN alone. If a plain GAN exercise is ever wanted it has to be
   written from scratch.
10. **§1 Day 4 hours do not add up.** The header says 6.5 AH but the sessions sum
    to 4.5 (autoencoders 2 + flow 1.5 + GAN 1). Days 1, 2 and 3 each sum to 6.5
    correctly. Day 4 therefore has 2 AH unaccounted for — either the budget is
    wrong or the day needs more material. **Needs a decision**, and note that the
    leftovers in `autoencoders-plan.md` amount to only about 0.5 AH, so they do
    not close the gap by themselves.
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

15. **§4b's AnAge callback story was wrong in three ways.** Checked against the
    notebooks: (a) AnAge appears only in `exercises/linear-anage.ipynb` and its
    solution, which is **Day 1 homework, not a session** — the apparent hit in
    `sessions/linear_regression.ipynb` is base64 image data, not a reference;
    (b) `sessions/robust-regression.ipynb` uses `data/outliers.csv`, **not**
    AnAge, so there is no AnAge treatment in the Day 1 bonus and the "third
    treatment of one dataset" thread does not exist; (c) the Day 1 exercise is
    Kleiber's law — `Metabolic rate (W)` on `Body mass (g)` — so it does not share
    a response variable with a lifespan model. Also, `Body mass (g)` has only 627
    non-null rows, so the demo would have needed `Adult weight (g)` (2,951 non-null,
    2,560 complete with `Maximum longevity (yrs)`) rather than the column the plan
    named. **Superseded:** §4b was subsequently respecified from scratch around UCI
    POWER, and AnAge is now explicitly out of scope for `flow.ipynb`. This entry is
    kept because it is the reason the AnAge design was abandoned.
16. **`autoencoders-plan.md` is nearly spent, contrary to what its text implies.**
    The bottleneck sweep, the 2D latent scatter, **latent interpolation**, the
    convolutional autoencoder and the structured denoising section are all already
    in `sessions/autoencoders.ipynb`, the indexing bug is fixed, and the unused
    imports are gone. Only reconstruction diagnostics (currently just exercise 5)
    and one optional extension remain — see §12 / Phase 8. This also
    supersedes the note in an earlier progress entry that called the document
    unexecuted.
17. **§4b's warning "do not promise to reproduce 0.24 nats" pointed the wrong
    way.** The concern was that eight layers on a 200k subsample with no
    hyperparameter search would fall short of MAF(10)'s published 0.24 nats on
    POWER. It measured **+0.350**. The warning missed that the notebook's
    transformer is a rational-quadratic spline from Durkan et al. 2019, which
    postdates the MAF paper and is more expressive than the affine transformer
    MAF used — so the comparison was never like-for-like in the direction
    assumed. The advice itself was still right: do not frame it as a
    reproduction. See the Phase 7 entry above.
18. **§5 of `DL2026_GPU_HANDOFF.md` forbids editing this plan from the GPU
    branch**, which is why Phases 2 and 7 were recorded here only after the fact.
    Updated on Yoav's explicit instruction (2026-09-01); treat §5 as superseded
    on this point, as it already is for `download_data.py` (Phase 2, open item 1).

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

`autoencoders` and `transfer` both build models with `keras.Input` + `keras.Model` and neither uses
`Sequential`, so `functional_keras` is critical for these sessions.

### Day 4 — Representations and densities (6.5)
| Session | AH |
|---|---|
| `sessions/autoencoders.ipynb` | 2 |
| `sessions/flow.ipynb` | 1.5 |
| `sessions/GAN.ipynb` | 1 |

Homework: `exercises/ACGAN.ipynb`

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
already contains `flow`, `autoencoders`, `functional_keras`, and
`transfer_learning` (and `finetuning`, since deleted), and — conveniently — it already lacks every notebook that
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
with PIL + NumPy, mirroring the loader reproduced in `DL2026_GPU_HANDOFF.md`
§2a (originally from `finetuning.ipynb`, now deleted). Do **not**
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

## 4b. Phase 7 — Add a real density-estimation example to `sessions/flow.ipynb`  ✅ DONE (2026-09-01, GPU box, `c044755`)

*(This was "Phase 3b". Section number kept as a document anchor.)*

**This section was replaced wholesale on 2026-08-31.** The previous design fitted a
*conditional* flow to AnAge life-history data. It is dropped: `p(lifespan | mass)`,
the AnAge callback story, Palmer penguins, autoencoder latent spaces, and
conditioning of any kind are all **out of scope** for this notebook. Correction 15
records why the AnAge framing did not survive contact with the notebooks. The
replacement below is the current spec.

### Context

Day 4 session, 1.5 academic hours (50 minutes each). The notebook currently uses
`make_moons` throughout: KDE and GMM baselines, then a FlowJAX masked
autoregressive flow with rational-quadratic splines, a density comparison and a
sample comparison. It trains in about two seconds. **Keep all of it as the first
half (~45 min)** and add a real dataset as the second half (~30 min).

Do not use Palmer penguins, AnAge, or an autoencoder latent space. Do not make the
flow conditional.

### What to add: UCI POWER

The MAF paper (Papamakarios, Pavlakou & Murray 2017, arXiv:1705.07057) introduced
POWER, GAS, HEPMASS, MINIBOONE and BSDS300 to the density-estimation literature.
Every subsequent flow paper — RealNVP comparisons, NAF, Block-NAF, Neural Spline
Flows, FFJORD, Glow — reports the same benchmarks. This is the textbook use of a
normalizing flow, and it fits the notebook's existing
fit-baselines-then-fit-flow-then-compare structure with no redesign.

POWER is six dimensions of household electricity measurements, so pairwise
marginals remain plottable and the existing density and sample plots keep working.

### Data acquisition

Use Papamakarios's preprocessed version, which is what the whole comparison
literature uses — locate it via the `gpapamak/maf` repository. Fall back to raw UCI
"Individual household electric power consumption" plus the preprocessing in
`maf/datasets/power.py` only if the preprocessed archive is unavailable.

Preprocessing, per the paper's Appendix D: drop date and time, drop
discrete-valued attributes and attributes with near-perfect Pearson correlation,
add small uniform noise to dequantize the rounded measurements, then subtract the
sample mean and divide by the sample standard deviation. Split 10% test, then 10%
of the remainder as validation.

**The dequantization noise is not optional.** The raw measurements are rounded, so
without it the flow chases discrete artifacts and the log-likelihood diverges
upward without bound. Make this an explicit, commented step with a one-line
explanation — it is a real gotcha and a good teaching moment about likelihoods on
quantized data.

Route the download through `download_data.py`, cache under `data/`, and
`.gitignore` it.

### Notebook structure for the new section

1. Markdown intro: why move past two moons. In 2D, KDE is genuinely competitive —
   the first half of this notebook understates the baselines. The case for flows is
   dimension and sample size: KDE's cost grows with the training set and its
   quality degrades with dimension, while a flow is a fixed-size model giving exact
   likelihoods either way.
2. Load POWER, describe the six variables, show pairwise marginals.
3. Subsample to roughly 200k training rows so training stays under a couple of
   minutes. State explicitly that this is a subsample and that it changes the
   achievable number.
4. Baselines on the same data: a Gaussian fitted to the training set (the MAF
   paper's own baseline), a GMM, and KDE. Report test log-likelihood in nats for
   each, and report KDE's fit and scoring wall-time — the cost is part of the
   argument.
5. Fit the MAF, reusing the existing FlowJAX setup with more `flow_layers` and a
   wider `nn_width`. Keep the train/validation loss curve visible.
6. Compare test log-likelihood in nats across all four models. Compare flow samples
   against the data marginals.
7. Discussion cell with the published numbers: MAF(10) reaches 0.24 ± 0.01 nats on
   POWER against RealNVP(10) at 0.17 ± 0.01, and NAF later reached 0.62 ± 0.01.
   Across the five benchmarks, MAF was best on three and MADE MoG on the other two.

**Do not promise to reproduce 0.24 nats.** That is ten layers with the paper's
hyperparameter search on the full dataset. Frame it as "our number against the
published one" and let the gap be the discussion.

### Exercises

Replace at least two of the four existing knob-turning exercises:

- Fit the same flow to MINIBOONE — 43 dimensions, only 36k rows, the smallest
  download of the five — and compare against the Gaussian and GMM baselines. This
  is where the baselines stop being competitive at all.
- Compare KDE fit-and-score time against flow training-and-score time as a
  function of training-set size.

Keep the existing exercises on `flow_layers` / `nn_width` / spline knots.

### Housekeeping in the same pass

- Logo path is `img/logo.png`; every other session uses `../logo.png`. **Verified:**
  in the *opening* cell only `flow.ipynb` and `jax.ipynb` use `img/logo.png`;
  `CNN_timeseries.ipynb` has no logo image at all. Fix `flow.ipynb` here, and fix
  `jax.ipynb` and `CNN_timeseries.ipynb` in the same pass since it is one line each.
  (`K_FFN.ipynb` also matches a grep for `img/logo.png`, but that is a Keras logo in
  its body, not the course logo — leave it.)
- Add the missing "In this session we will understand:" intro cell. **Note:** only
  4 of 17 session notebooks currently have this opening (`GAN`, `K_CNN`, `K_FFN`,
  `audio`), so §8's house style is aspirational rather than universal. Adding it
  here is still right; just do not expect the other notebooks to match.
- Add the missing Colophon cell with the CC BY-SA 4.0 block, copied verbatim from a
  sibling session notebook. **Verified:** `flow.ipynb` really is the only session
  notebook without one.
- Add one sentence acknowledging that this notebook uses FlowJAX and Equinox rather
  than Keras or raw JAX — it is the only one that does, and students will otherwise
  wonder why the API changed. The reason: no Keras flow implementation is worth
  teaching.
- Confirm `flowjax` is in `requirements.txt`. **Verified present** (line 20), added
  in Phase 6.
- Keep the framing that autoencoders learn a representation with no density, flows
  learn a density you can evaluate and sample, and GANs learn to sample without
  writing down a density. Do not call autoencoders generative models.

### References to add

- Papamakarios, Pavlakou & Murray 2017, *Masked Autoregressive Flow for Density
  Estimation*, arXiv:1705.07057 — already cited; add the benchmark table as the
  source of the reported numbers.
- Durkan et al. 2019, *Neural Spline Flows*, arXiv:1906.04032 — the
  rational-quadratic splines the notebook already uses.
- Papamakarios et al. 2021, *Normalizing Flows for Probabilistic Modeling and
  Inference*, JMLR 22(57).
- One sentence, no demo: Rezende & Mohamed 2015 named normalizing flows and used
  them for variational inference, which is the other classical application.

### Acceptance criteria

- Notebook runs top to bottom on GPU; record wall-clock time and confirm the new
  section fits in ~30 minutes of class time, with a checkpoint provided if training
  exceeds that.
- Report the measured test log-likelihood in nats for Gaussian, GMM, KDE and flow on
  POWER, so the discussion cell can be written around real numbers rather than
  expected ones.
- No new dependencies beyond `flowjax`; no `torch`, `tensorflow`, or
  `tensorflow_datasets`.
- Total session budget unchanged at 1.5 AH.

### Two practical warnings before starting

1. **KDE cannot be scored naively at this scale.** POWER has ~2.05M rows, so a 10%
   test split is ~200k points. `sklearn`'s `KernelDensity` costs
   O(n_train x n_test), which at 200k x 200k is ~4e10 kernel evaluations and will
   not finish. Score KDE on a fixed random subsample of the test set (a few
   thousand points) and say so in the notebook — and note that the wall-time
   comparison in step 4 is then a *lower bound* on KDE's real cost, which
   strengthens rather than weakens the argument.
2. **Resolved: Phase 7 moved to the GPU handoff** (2026-08-31). The acceptance
   criteria call for GPU, and a 6-dimensional MAF with more layers over 200k rows is
   a different proposition from the two-moons design this phase started as. It is now
   `DL2026_GPU_HANDOFF.md` §4b, alongside Phases 2 and 3. The housekeeping items in
   this section moved with it, including the `sessions/jax.ipynb` logo fix.

3. **Resolved: the data is wired up.** `download_data.py maf-benchmarks` fetches
   Papamakarios's preprocessed datasets — Zenodo record 1161203, CC-BY-4.0, a single
   857 MB `data.tar.gz` — and lands them at `data/maf/power/` and
   `data/maf/miniboone/`. The archive's own top-level directory is called `data`, so
   the loader strips that component to avoid colliding with the repo's `data/`, and it
   filters out `gas`, `hepmass`, `bsds300`, `mnist` and `cifar10` to save ~650 MB.
   `data/maf` is gitignored. The strip-and-filter logic was tested against a synthetic
   archive mirroring the real layout.

## 5. Phase 4 — ❌ VOID. `sessions/finetuning.ipynb` has been deleted

**Decision, 2026-08-31: the hyena notebook is removed from the branch entirely**,
not reworked and not kept as a bonus. The index link and the file are both gone,
`download_data.py` no longer fetches the hyena archive, and Day 3 has no Bonus
subsection. `sessions/transfer.ipynb` (Phase 2) is now the only transfer-learning
material in the course.

The notebook survives in git — `git show origin/DeepLearning:sessions/finetuning.ipynb`
— if the metric-learning case study is ever revived. The rest of this section is
kept only as a record of what was wrong with it.

<details>
<summary>Original Phase 4 plan (no longer to be executed)</summary>

### Original: Rework `sessions/finetuning.ipynb` as a bonus case study

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

</details>

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
5. Report the measured accuracy numbers for `transfer.ipynb` and the restructured
   `audio.ipynb` — especially whether the audio probe beats the from-scratch CNN.
   This may change what gets said in class.

---

## 10. Out of scope

- The hyena re-identification notebook, in any form — **deleted from the branch**
  (see §5), along with the metric-learning follow-up it was meant to motivate.
- Metric learning (ArcFace, contrastive, cosine-NN retrieval).
- Any PyTorch migration. The `torch` branch stays where it is.
- `sessions/augmentation.ipynb` on the `torch` branch is a 2-cell stub; ignore it.
- KerasHub / Whisper audio embeddings — a possible future bonus, not this pass.
- The legacy branches (`kti2018`, `kti2020`, `amat2019`, `trees`, `lam2020`,
  `lam2021`, `intuit`, `landa`, `IDC2018`) and their `reinforcement.ipynb` /
  `FFN_GenModel.ipynb` / `TF_CNN.ipynb` material.

---

## 12. Phase 8 — Finish `sessions/autoencoders.ipynb`  ✅ DONE

Closes out `autoencoders-plan.md`. **This is a small stage** — most of that
document has already been carried out (correction 16). Verified present in the
notebook today, 28 cells: intro and framing, dense baseline, bottleneck sweep
over `latent_dims`, a 2D latent space with a labelled scatter, **latent
interpolation** (cell 15 encodes a `1` and an `8` and decodes along the path),
convolutional autoencoder with a dense-versus-conv comparison, structured
denoising at fixed σ warm-started from the conv model, exercises, references,
colophon. The `X_test.shape[1]` indexing bug is fixed, the unused `pickle` import
is gone, and `Y_test` is genuinely used (cell 15 selects digits by label).

Two things are left.

**1. Reconstruction diagnostics, in the notebook body.** Currently this exists
only as exercise 5 ("Plot the test images with the largest reconstruction
error"). Promote it to a worked cell:

- per-image MSE over the validation set, then show the best and worst
  reconstructions side by side;
- mean reconstruction error broken down by digit class, as a bar plot;
- one or two sentences on the failure modes — which digits blur, and whether the
  model hedges on ambiguous ones.

The teaching point is inspecting a trained model rather than only training it.
Replace exercise 5 with something that is not now redundant.

**2. One extension, not several.** `autoencoders-plan.md` offers three; pick the
second:

- **Encoder features for a downstream classifier** ← recommended. Freeze the
  trained encoder, take the latent vectors as features, and fit a small Keras
  softmax head on them; compare against the same head on raw pixels. This is the
  *same protocol* as the linear probe in Day 3's `sessions/transfer.ipynb`, with
  the representation coming from an autoencoder trained on the data itself rather
  than from ImageNet. Making that parallel explicit is worth more than a new
  technique, and it ties Day 4 back to Day 3.
- Anomaly detection via reconstruction error — already named in the intro's
  "typical uses" list but never shown. Cheap, but a weaker link to the rest.
- A VAE teaser. Keep to a short "what comes next" markdown cell if wanted; do not
  implement a VAE here.

**Budget.** About 0.5 AH, taking the session from 2 to 2.5 AH. That does not on
its own close Day 4's 2 AH gap (correction 10) — the gap needs a separate
decision.

**Housekeeping in the same pass:** check whether `Y_train` is still unused (only
`Y_test` is needed for the label-based selections) and drop it if so.

After this lands, `autoencoders-plan.md` and `density_plan.md` are both fully
spent and can be deleted.

### Phase 8 — what actually happened

Both planned items landed, and the second one was **respecified by decision** into
something considerably better than "one extension".

**1. Reconstruction diagnostics** (new section after the dense-vs-conv figure).
Per-image MSE over the 10,000 validation images with a histogram, an
easiest-versus-hardest grid of 5+5 images labelled with their own MSE, and mean
error per digit class. Measured: mean 0.0033, median 0.0028, worst single image
0.0351 = **12.5x the median**; easiest digit `1` at 0.0011, hardest digit `8` at
0.0049, a **4.4x** spread. The mean sitting above the median makes the
"averages hide the distribution" point on real numbers.

**2. Label-efficiency sweep, not a linear probe.** The original spec was a linear
probe on the latents versus raw pixels. That was replaced with the question that
actually matters: does unsupervised pretraining pay off when labels are scarce?
Three models at five label budgets:

| labeled images | end-to-end CNN | AE + MLP head | AE + linear head |
|---|---|---|---|
| 100 | 0.7659 | **0.8105** | 0.7530 |
| 250 | 0.8353 | **0.8743** | 0.8458 |
| 1,000 | 0.8979 | **0.9300** | 0.8930 |
| 2,000 | 0.9411 | **0.9487** | 0.8957 |
| 60,000 | **0.9856** | 0.9837 | 0.9321 |

With all labels, end-to-end supervision wins (by 0.2 points, i.e. a tie, and our
CNN is small and trained for 10 epochs). As labels get scarcer the ordering flips
and the margin grows monotonically: +0.8, +3.2, +3.9, **+4.5** points. Crossover
between 2,000 and 60,000.

**A methodological trap worth recording**, because the first attempt fell into it.
A linear head on 100 labels initially scored **0.1547** — near chance — and it would
have been easy to write that up as a fact about representations. It was not. Two
causes: `batch_size=min(128, n)` meant 100 labels x 60 epochs was only **60
gradient updates**, and the encoder's final layer is `activation="linear"` and
unconstrained so the latents are off-scale (measured mean |z| 2.11, max 15.16). The
notebook now standardizes the latents with training-set statistics and equalizes
*gradient updates* rather than epochs, and says why in the narrative. Anyone
touching this section should keep both guards.

Also worth carrying forward: the linear head caps at 0.9321 with all 60,000 labels
while an MLP head on the *same frozen features* reaches 0.9837. A linear probe is a
comparable **lower bound** on representation quality, not a measurement of it —
which is the right way to read Day 3's `transfer.ipynb` probe numbers too.

**Housekeeping done in the same pass:**

- `SEED = 23` with `keras.utils.set_random_seed`, replacing a commented-out config
  stub. The notebook previously had unseeded `np.random.choice` throughout, so the
  "hardest images" figure would have changed on every run.
- The 2D latent scatter plotted all 10,000 validation points and was **2.74 MB on
  its own**, the largest single object in the notebook. Now a seeded 3,000-point
  subsample, with the count in the title. Colormap changed `Set1` -> `tab10`:
  `Set1` has only 9 colors for 10 digit classes, in the one figure where color *is*
  the data.
- `conv_history` was fitted with `verbose=1`, producing **1.06 MB** of ANSI
  progress-bar spam. Now `verbose=2`, matching the denoising cell.
- Net effect: the notebook is **3.0 MB, down from 4.2 MB**, despite 10 new cells.
- The `Y_train` housekeeping item is void — the sweep uses `Y_train`, and `ncats`,
  both of which were previously dead.
- Exercises reworked from 5 to 8, dropping the one the diagnostics section made
  redundant and adding the crossover hunt and the Fashion-MNIST repeat.
- References gained Erhan et al. 2010 (the classic study of exactly this
  experiment) and SimCLR (the modern version, with label-efficiency curves).

**Validation.** Runs top to bottom from a clean kernel via `nbconvert --execute`,
zero cell errors, **25 minutes wall-clock on CPU**. That discharges §9 item 2 for
this notebook. Note the runtime for classroom purposes: the conv autoencoder is
~50 s/epoch x 15 and the denoising model ~50 s/epoch x 10 on CPU, so this notebook
cannot be trained live in class — the session budget assumes the stored outputs are
used, and unlike the Keras sessions there is **no `load_model` checkpoint path in
this notebook**. Adding one would be a sensible follow-up.

`autoencoders-plan.md` is now fully spent and can be deleted.
