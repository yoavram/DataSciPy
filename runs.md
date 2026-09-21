# Runs log — `sessions/reid.ipynb` (issue #12)

Every computation done for the re-identification session: what was run, why, what came
back, and whether it is still load-bearing. Working notes, not teaching material —
untracked, and safe to delete once the session is finished.

Branch `metric-learning`. Machine: 2x RTX A4000 (16 GB), 96 CPU cores.
Course env `.venv` (Keras 3.15.1 / JAX). PyTorch work runs in
`~/Work/Research/AnimalCLEF26/.pixi/envs/default`, never in the course env.

---

## 0. Data

**`scripts/reid_data.py`** — once, ~2 min (GPU for the embeddings).

SeaTurtleIDHeads via `wildlife-datasets` (Kaggle): 7,582 head crops, 400 loggerhead
individuals, 2010-2021. Outputs `turtle_catalogue.csv`, `turtle_images_224.npy`,
`turtle_effnetv2s_embeddings.npy`, `turtle_clip_vitb16_images.npy`.

Individuals split three ways, fixed once: **train 300** (fitted and reported),
**tune 50** (every hyperparameter, never reported), **unknown 50** (never trained,
never in the database — the open-set queries and the clustering target).

Two arms, differing in one rule only. Time: the later half of each individual's
observation *dates* becomes the query set (`TimeProportionSplit`, Čermák et al.).
Random: the **same number** of query images per individual, drawn at random.

| | database | query | same-day leak |
|---|---|---|---|
| random | 3,838 | 3,744 | **94.7%** |
| time | 3,838 | 3,744 | **0.0%** |

The leak is over known-identity queries only; unknowns have no database entry by
construction. 172 of 400 individuals have a single observation date and so contribute
no query image to either arm.

---

## 1. Probe grid, first pass — **superseded**

`reid_arcface.py`, 18 runs (2 arms x 3 heads x 3 seeds), CPU, ~5 min.
Fixed `margin=0.3`, `k=3`.

Result: dynamic margin looked harmful, sub-centers looked neutral.
**Discarded**: the margin was fixed across the comparison, so the arms differed in a
hyperparameter as well as in the thing under test. This is the confound the notebook
warns about, committed while writing the notebook that warns about it.

---

## 2. Margin / sub-center sweep — `scripts/reid_sweep.py`

Every configuration gets its *own* margin and `k`, chosen on the tune individuals.
Sharded 6 ways (one process per arm x head), CPU.

- **v1**: margins 0.0–0.5, k ∈ {1,2,3,4}, 2 seeds = 216 runs, ~1 h.
  **Discarded**: selected `m = 0.5` and `k = 4`, both its own largest values. A search
  that stops at the edge of its range has found a wall, not an optimum.
- **v2**: margins extended to 0.8, k to 8 → **468 runs** total (resumes, skips done).
  Winners now interior. `data/turtle_arcface_sweep.csv`.

Selected (tune mAP@R):

| arm | head | margin | k |
|---|---|---|---|
| random | plain | 0.2 | 1 |
| random | sub-center | 0.3 | 4 |
| random | dynamic | 0.3 | 4 |
| time | plain | 0.5 | 1 |
| time | sub-center | 0.5 | 4 |
| time | dynamic | 0.4 | 4 |

Left free, the sweep picks `k = 1` for the sub-centered head in the random arm — it
declines to use sub-centers at all. Sub-centered heads are therefore restricted to
`k >= 2` so the comparison is between genuinely different models.

Margin profile is a gentle hill in the random arm (peak ~0.2) and nearly flat in the
time arm; seed sd across the sweep is 0.004–0.006, comparable to most differences in it.

---

## 3. Selected probes — reported

`reid_arcface.py --selected`, 18 runs, CPU, ~6 min. Backbone frozen; a 512-d trunk and
the ArcFace head fitted on cached features (1,072,640 trainable params against the
backbone's 20,331,360, which are untouched). Epoch budget chosen per run on `tune`,
peak weights restored.

Reported individuals, mean of 3 seeds:

| arm | head | recall@1 | mAP@R |
|---|---|---|---|
| random | plain | **0.5981** ± .005 | **0.2137** ± .004 |
| random | sub-center | 0.5763 | 0.2002 ± .011 |
| random | dynamic | 0.5648 | 0.1775 ± .016 |
| time | plain | **0.2671** ± .003 | **0.1016** ± .001 |
| time | sub-center | 0.2580 | 0.0954 |
| time | dynamic | 0.2450 | 0.0822 |

Neither extension pays for itself at this scale, even with its own hyperparameters;
the dynamic margin costs more than the seed spread. Plain head is the reported model.

---

## 4. Batch composition — answering the companion notebook's deferred question

`reid_arcface.py --selected --head plain --sampler {2,4}`, 12 runs, CPU.
m-per-class batches (`MPerClassSampler`) against plain shuffling.

| arm | batches | recall@1 | mAP@R |
|---|---|---|---|
| random | shuffled | **0.5981** | **0.2137** |
| random | m = 2 | 0.5914 | 0.1973 |
| random | m = 4 | 0.5825 | 0.1848 |
| time | shuffled | **0.2671** | **0.1016** |
| time | m = 2 | 0.2352 | 0.0718 |
| time | m = 4 | 0.2400 | 0.0727 |

Shuffling wins. A proxy loss is indifferent to whether a batch contains positive pairs,
and constraining composition *hurts*: a shuffled batch of 128 touches ~128 proxies, an
m = 4 batch touches 32. Consistent with the lab's own turtle script, which uses
`MPerClassSampler(m=1)`.

---

## 5. Fine-tuning the backbone

Recipe is **not tuned here**. It is `~/Work/Research/AnimalCLEF26/scripts/finetune_turtles.py`
("same recipe as CzechLynx v2") transplanted onto EfficientNetV2S, and it agrees with
Musgrave et al. (arXiv:2003.08505) §3.1 on separate learning rates:

- backbone **5e-6**, projection and ArcFace **1e-4** (20x; supplied in Keras by the
  `w = raw * mult` reparameterization, since Keras has no per-layer learning rate)
- AdamW, cosine annealing to 1e-6 over the budget, batch 32
- ArcFace **s = 64, m = 0.5, one centre**, 512-d embedding behind Linear+BatchNorm
- last block (`block6a`+) unfrozen, **every BatchNorm frozen**
- augmentation: hflip (turtles are bilaterally symmetric), brightness, contrast

Deviations from the lab script, deliberate: 224 px not 440, EfficientNetV2S not the
MiewID backbone, no RandomErasing/GaussianBlur.

### 5a. 50 epochs — **discarded**

6 runs (2 arms x 3 seeds), ~20 s/epoch.

| arm | recall@1 | mAP@R | AUC |
|---|---|---|---|
| random | 0.5689 | 0.1549 | 0.7134 |
| time | 0.2705 | 0.0922 | 0.5777 |

Worse than the frozen probe — i.e. "fine-tuning doesn't help". **Every run peaked at
epoch 50 of 50.** The conclusion was an artifact of the budget, not a result.

### 5b. 150 epochs — 6 runs

Best tune mAP@R: random 0.3189 / 0.3254 / 0.3159 (peaks 148, 149, 149);
time 0.1957 / 0.1866 / 0.1815 (peaks 146, 145, 147).

Reported individuals, mean of 3 seeds:

| arm | model | recall@1 | mAP@R | AUC |
|---|---|---|---|---|
| random | probe | 0.5981 | 0.2137 | 0.7511 |
| random | **fine-tuned** | **0.7282** ± .004 | **0.4680** ± .005 | **0.8154** |
| time | probe | 0.2671 | 0.1016 | 0.6408 |
| time | **fine-tuned** | **0.4178** ± .006 | **0.2612** ± .002 | **0.6720** |

### 5c. 250 epochs — **reported**

Peak-at-cap means something different under cosine annealing: the LR goes to ~0 at the
end of the budget, so the best epoch lands near the end *by construction*. The test
that means something is whether a longer budget still improves the result.

6 runs at 250 epochs. All peaks now **interior**: random 247 / 187 / 225,
time 214 / 153 / 222.

Tune mAP@R, mean of 3 seeds:

| arm | 150 ep | 250 ep | seed sd (250) | gain in sd |
|---|---|---|---|---|
| time | 0.1879 | 0.1948 | 0.0055 | ~1 (converged) |
| random | 0.3200 | 0.3367 | 0.0026 | ~6 (still improving) |

Time arm converged; random arm still improving at 250, so its fine-tuned numbers are
an underestimate. Stopped here because the peaks are interior and no conclusion turns
on the difference — stated that way in the notebook rather than claiming convergence.

**Reported** (250 epochs, 3 seeds, per-arm models):

| arm | model | recall@1 | recall@5 | mAP@R | AUC |
|---|---|---|---|---|---|
| random | probe | 0.5981 | 0.7344 | 0.2137 | 0.7511 |
| random | **fine-tuned** | **0.7653** ± .001 | 0.8344 | **0.6089** ± .013 | **0.8399** |
| time | probe | 0.2671 | 0.4188 | 0.1016 | 0.6408 |
| time | **fine-tuned** | **0.4580** ± .003 | 0.5570 | **0.3287** ± .010 | **0.7012** |

Fine-tuning is the largest effect in the session: top-1 0.27 → 0.46 and mAP@R more
than tripled on the honest split. Clustering ARI (unknown individuals, time arm)
0.331 → 0.483 oracle-k, 0.346 → 0.517 estimated-k.

Per-arm inflation: probe 2.24, fine-tuned 1.67. Holding one embedding fixed across
both protocols (the diagnostic table in the notebook): frozen 2.87, probe 2.45,
fine-tuned 1.83, MiewID 1.02.

Throughput note: 4 concurrent runs (2 per GPU, `XLA_PYTHON_CLIENT_MEM_FRACTION=.45`,
~15 GB/card) gave **13.4 s/epoch each** vs ~18 s/epoch for a lone job — packing is a
real ~5x win in wall-clock, not just time-slicing.

---

## 6. PyTorch-side precomputation — `scripts/reid_precompute_torch.py`

Run in the research env, never in the course env. Output is `.npy`/`.npz` only.

- **MiewID-msv3** embeddings, all 7,582 crops, 2,152-d, **CPU** (~50 min) to avoid
  contending for GPUs. Sanity check before trusting it: same-ID mean cosine 0.716 vs
  diff-ID 0.34, top-1 0.953 on an easy subset — matches the lab's own reported
  behaviour, so the `meta parameter` load warnings are benign.
- **ALIKED** features for the 7,341 images that appear in a query or a top-10
  shortlist, 12 CPU shards, ~45 min, 2.8 GB cache in `data/turtle_aliked/`.
- **LightGlue** match counts for 37,440 (query, shortlisted candidate) pairs, 12 CPU
  shards, ~25 min. A full matrix would be 14.4M pairs ≈ 500 h — which is the cost that
  motivates the hybrid, so a shortlist is what gets shipped.
- **Figures**: `sessions/img/reid_lightglue_matches.png` — a clear match (201
  correspondences), a thin one (35), and a **false** match between different
  individuals (100).

### Contamination check — decisive

Both fine-tuned turtle MiewID checkpoints in the research repo are unusable as a
comparison: `miewid_turtle_ext_finetuned.pt` was fine-tuned on SeaTurtleIDHeads itself,
and 100% of SeaTurtleIDHeads images are head crops of photographs in the AnimalCLEF
SeaTurtleID2022 set. There is no turtle image here it has not trained on. **Base
MiewID-msv3 only**, with the caveat that its Wildbook corpus may include this archive.

---

## 7. Evaluation results (three metric families)

Rejection: threshold chosen on tune individuals, half of them removed from the tune
database to create held-out unknowns.

Clustering: agglomerative, on the 50 unknown individuals; linkage and distance
threshold chosen on tune, **separately for each mode**.

| | frozen | probe (time) | MiewID |
|---|---|---|---|
| ARI (oracle k) | 0.136 | 0.331 | **0.918** |
| ARI (estimated k) | 0.157 | 0.346 | 0.916 |
| k estimated (true 50) | 135 | 232 | 69 |

Estimated k beats oracle k despite being wrong by 3–5x: ARI punishes impure merges far
more than over-segmentation. An early version reported oracle-k ARI of 0.014 purely
because the linkage had been chosen for the other mode — a tenfold swing from an
unreported decision.

**Split sensitivity** (one embedding, two protocols):

| model | random | time | inflation |
|---|---|---|---|
| frozen EffNetV2S | 0.384 | 0.134 | **2.87** |
| probe | 0.646 | 0.263 | 2.45 |
| fine-tuned (250 ep) | 0.845 | 0.461 | 1.83 |
| MiewID-msv3 | 0.948 | 0.928 | **1.02** |

Monotone: the better the embedding, the less the split matters. The encounter shortcut
is worth most to the model that can least do the task. (Caveat stated in the notebook:
a model that trained on these photographs would show the same signature.)

**Hybrid re-ranking** (exercise; 2,578 reported queries, top-10 shortlist):
embedding top-1 0.250 → local re-rank **0.369**, ceiling 0.476, so ~53% of the
available headroom. Fusion weight chosen on tune lands at 0.9 — local score dominates.
Correct pairs average 114 correspondences, wrong pairs 39. Gain as a *fraction of
headroom* is 0.89 for individuals with ≤3 database photos vs ~0.5 for the rest; the raw
gain says the opposite, because the groups differ in headroom.

**Few-shot** (nearest-class-mean, unknown individuals): support drawn anywhere
0.302 → 0.563 from 1 to 5 shots; support from the earliest encounter with queries on
later days 0.200 → 0.288. Five photographs from one encounter are worth far less than
five spread over time.

---

## 8. Notebook

`sessions/reid.ipynb` — 99 cells, 0 errors, **3 min 23 s** end-to-end on CPU.
`solutions/reid.ipynb` — 13 cells, the hybrid exercise worked.
Built by a generator in the session scratchpad (`build_reid_nb.py` + `part_*.py` +
`emit.py`) so the whole notebook can be regenerated and re-executed.

Artifacts the notebook loads total ~352 MB; only the MiewID embeddings and LightGlue
scores (66 MB) cannot be rebuilt in the course env. `download_data.py reid-arrays`
prints rebuild instructions until `REID_ARRAYS_URL` is set to a hosted tarball.
