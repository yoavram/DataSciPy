# Next session: Phase 3 — `sessions/audio.ipynb`

Branch `DL2026-gpu` (tracks `origin/DL2026-gpu`). Phases 2 and 7 are done and
committed with outputs. Phase 3 is the last GPU-bound item in the handoff.

## Read, in this order

1. `DL2026_PLAN.md` §4 — the spec for this notebook.
2. `DL2026_PLAN.md` §0a "Phase 3 — survey before starting" — **four pre-existing
   defects that §4 does not mention.** The first one changes what the
   from-scratch baseline number means.
3. `DL2026_GPU_HANDOFF.md` §3 and §5–7 — operational detail, boundaries,
   deliverables, acceptance checks.
4. `CLAUDE.md` — house style.

## Start here

```bash
.venv/bin/python download_data.py esc50   # check whether it is already present
ls data/ESC-50-master
```

The four defects to fix while restructuring, from the survey:

1. `validation_split=0.1` splits over ~1s overlapping *segments* cut from the
   same 5s clip, so windows of one recording land on both sides and the reported
   ~55% is optimistic. ESC-50 ships 5 official folds — split on `fold`.
2. The history plot reads `history['acc']` / `history['val_acc']`; Keras 3 uses
   `accuracy` / `val_accuracy`, so the cell raises `KeyError` as committed.
3. Saves to `../data/keras_esc50_model.h5`; the branch idiom is `.keras`.
4. Uses *test* terminology against the branch-wide sweep to *validation*.

Note `sessions/audio.ipynb` is 34 MB (embedded audio and figure outputs).

## Verified for you

- Environment: keras 3.15.1, jax 0.11.1, backend `jax`, `gpu`, 2x RTX A4000.
  `.venv` exists at 3.12.13.
- Disk was at 97% / 27 GB free before Phase 7; `data/maf` now holds 138 MB.
  `data/CUB_200_2011/attributes/` (70 MB) is unused and can be deleted.
- Notebooks execute cleanly with
  `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <nb>`.

## Convention to follow

Anything stated as a mechanism in a discussion cell must be measured, or
labelled unmeasured. GPU runs here are minutes. See the unfreeze-depth table in
`sessions/transfer.ipynb` and the model-comparison table in `sessions/flow.ipynb`
for the pattern.

## Not yet done, outside the GPU handoff

- `DL2026_PLAN.md` §0a still lists Phase 7 as DELEGATED; it is done. §5 of the
  handoff forbids editing the plan from this branch, so whoever merges should
  update the status table and record the Phase 7 numbers.
- Day 4 is still 2 AH short (correction 10). Phase 7 did not change that: the
  flow session remains 1.5 AH.
