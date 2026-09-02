# Next session: no notebook work is outstanding

Branch `DL2026-gpu`, pushed, tracks `origin/DL2026-gpu`. **Phases 2, 3 and 7 are
all done and committed with outputs.** `DL2026_GPU_HANDOFF.md` has no remaining
items — Phase 3 (`sessions/audio.ipynb`) was the last one.

## What is actually left

**One decision, not more notebook work: Day 4 is 2 academic hours short**
(`DL2026_PLAN.md` correction 10). Phases 7 and 8 did not close it — the flow
session stays at 1.5 AH and the autoencoder leftovers are worth about 0.5 AH.
This needs Yoav to choose what fills the gap, or to shorten the day.

## Phase 3, for the record

`sessions/audio.ipynb`: 68 cells, 2.8 MB, executes top to bottom from a clean
kernel. Measured on the official ESC-50 folds (1-4 train, 5 validation),
clip level, seeds 23 and 24:

| protocol | top-1 | top-5 |
|---|---|---|
| EchoNet from scratch | 55.25-57.00% | 84.50-85.50% |
| frozen `EfficientNetV2S` probe | 66.25-67.00% | 90.00-90.25% |
| fine-tune | 75.00-75.50% | 93.00-93.50% |

The probe **did** beat the from-scratch CNN, by about 10 points. Full write-up,
including the three-way channel-construction null and the 34-point measurement
of what a genuinely leaky split buys, is in `DL2026_PLAN.md` §0a
"Phase 3 — what actually happened".

**Three claims in `DL2026_GPU_HANDOFF.md` §4 did not survive execution** — the
"fatal" librosa blocker, the leaking-split mechanism, and the "~55%" baseline it
was to be compared against. See `DL2026_PLAN.md` corrections 20-22. The handoff
itself was left unedited, per its §5.

## Known gaps, unchanged by Phase 3

- `requirements.txt` still lacks `librosa` and `corner` (correction 4). Not edited
  here, per §5 of the handoff.
- Checkpoints and caches are gitignored and live at
  `~/Work/Teaching/DataSciPy/data/` on the GPU box. Disk there is at 98%; the
  three `esc50_image_*.npy` caches (331 MB each) are regenerable and are the
  first thing to delete.
