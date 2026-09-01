# Next session: Phase 7 — `sessions/flow.ipynb`

Branch `DL2026-gpu` (pushed, tracks `origin/DL2026-gpu`). Phase 2 is done; Phase 3
(`audio.ipynb`) is deliberately **not** next.

## Read, in this order

1. `DL2026_PLAN.md` §4b — the authoritative spec for this notebook. Detailed and current.
2. `DL2026_GPU_HANDOFF.md` §4b — operational deltas only (data paths, three traps).
3. `CLAUDE.md` — house style.
4. `DL2026_PLAN.md` §0a "Phase 2 — what actually happened" — the environment notes apply to you.

Do **not** resurrect AnAge, penguins, or a conditional flow; §4b was respecified
around UCI POWER and supersedes all of that.

## Start here

```bash
.venv/bin/python download_data.py maf-benchmarks   # 857 MB; lands in data/maf/{power,miniboone}
```
Not downloaded yet — kick it off before reading, it is the long pole.

Then: keep the existing `make_moons` material as the first half (~45 min) and add
UCI POWER as the second (~30 min). 1.5 academic hours total.

## Verified for you

- Environment is ready: keras 3.15.1, jax 0.11.1, backend `jax`, `gpu`,
  2x RTX A4000. `.venv` exists.
- **flowjax 19.1.1 + equinox 0.13.8 run fine on jax 0.11.1, on GPU** — a
  `masked_autoregressive_flow` builds and scores. No version risk.
- `data/maf` is gitignored. Disk is at 97%, 27 GB free — enough, but not roomy.
- `sessions/flow.ipynb` is 21 cells and all three housekeeping defects in §4b are
  real: it uses `img/logo.png`, has no Colophon, and has no "In this session we
  will understand:" intro. `sessions/jax.ipynb` needs the same one-line logo fix.

## The three traps (detail in §4b)

1. Dequantization noise is **not** optional — without it the log-likelihood
   diverges upward and the model looks great. Make it an explicit, commented step.
2. Do not score KDE naively; it is O(n_train x n_test) on ~200k test rows. Score a
   few-thousand-point subsample and say so.
3. Do not promise to reproduce the published 0.24 nats.

## Convention to follow, learned the hard way in Phase 2

Anything stated as a mechanism in a discussion cell must be measured, or labelled
unmeasured. GPU runs here are minutes. See the unfreeze-depth table in
`sessions/transfer.ipynb` for the pattern.

## Deliverable

Notebook executes top to bottom from a clean kernel, committed **with outputs**.
Report measured test log-likelihood in nats for all four models (Gaussian, GMM,
KDE, flow), plus KDE wall-time and the subsample size.
