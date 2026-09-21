"""Choose the angular margin and the number of sub-centers, on held-out individuals.

Keras 3 on the JAX backend. Run from the repository root, after scripts/reid_data.py:

    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python scripts/reid_sweep.py

Why this exists
---------------
The first pass compared a plain head at a fixed margin of 0.3 against a dynamic-margin
head at a fixed range of [0.15, 0.45], and the dynamic margin lost. That comparison is
worthless: neither value was chosen, so the arms differ in a hyperparameter as well as
in the thing under test, and a hyperparameter held fixed across a comparison is
measuring itself rather than the effect.

So every configuration gets its *own* margin and its *own* number of sub-centers,
selected on the 50 tune individuals -- never trained on, never reported -- and only
then are the configurations compared on the individuals the notebook reports.

The grid is run under both arms, because the arms have different amounts of signal and
there is no reason to assume they want the same margin.

Writes data/turtle_arcface_sweep.csv: one row per (arm, head, margin, k, seed).
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

import keras

import reid_arcface as ra

OUT = os.path.join(ra.DATA, "turtle_arcface_sweep.csv")
# The sweep is hundreds of small fits, which JAX on a CPU parallelizes badly: one
# process per (arm, head) with a modest thread budget finishes far sooner than one
# process trying to use the whole machine. `--out` keeps the shards apart.
SWEEP_SEEDS = (ra.SEED, ra.SEED + 1)


def run(arm, head_kind, margin, subcenters, seed):
    """One fit; returns the peak tune score and the epoch it happened at."""
    keras.utils.set_random_seed(seed)
    df, features, fit_rows, labels, counts, identities = ra.load_split(arm)

    tune = df[df.role == "tune"]
    tune_labels = pd.factorize(tune.identity)[0]
    is_query = (tune[f"arm_{arm}"] == "query").to_numpy()
    tune_rows = tune.index.to_numpy()

    if head_kind == "dynamic":
        # The dynamic head's margin grid is a *range* centred on the swept value, so
        # that "dynamic margin" and "margin" are not two names for one knob.
        margins = ra.margins_from_counts(counts, span=(margin / 2, margin * 1.5))
    else:
        margins = margin

    trunk = ra.build_trunk(seed, features.shape[-1])
    head = ra.ArcFaceHead(len(identities), scale=ra.SCALE, margins=margins, subcenters=subcenters)
    model = ra.build_model(trunk, head)
    monitor = ra.RetrievalMonitor(
        trunk,
        features[tune_rows[is_query]], tune_labels[is_query],
        features[tune_rows[~is_query]], tune_labels[~is_query],
    )
    monitor.on_epoch_end = _quiet(monitor.on_epoch_end)
    model.fit({"features": features[fit_rows], "label": labels}, labels,
              batch_size=ra.BATCH_SIZE, epochs=ra.MAX_EPOCHS, verbose=0, callbacks=[monitor])
    peak = int(np.argmax([s["mAP@R"] for s in monitor.history]))
    return monitor.history[peak], peak + 1


def _quiet(fn):
    """The sweep is hundreds of fits; the per-epoch line would bury the result."""
    import contextlib
    import io

    def wrapped(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    return wrapped


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ra.ARMS, action="append")
    parser.add_argument("--head", choices=ra.HEADS, action="append")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    # Resume: a rerun after extending the grid should only compute what is new.
    rows = []
    if os.path.exists(args.out):
        rows = pd.read_csv(args.out).to_dict("records")
        print(f"    resuming from {args.out} ({len(rows)} rows)")
    done = {(r["arm"], r["head"], r["margin"], r["subcenters"], r["seed"]) for r in rows}

    for arm in args.arm or ra.ARMS:
        for head_kind in args.head or ra.HEADS:
            subcenter_grid = (1,) if head_kind == "plain" else ra.SUBCENTER_GRID
            for margin in ra.MARGIN_GRID:
                for subcenters in subcenter_grid:
                    for seed in SWEEP_SEEDS:
                        if (arm, head_kind, margin, subcenters, seed) in done:
                            continue
                        scores, peak = run(arm, head_kind, margin, subcenters, seed)
                        rows.append(dict(arm=arm, head=head_kind, margin=margin,
                                         subcenters=subcenters, seed=seed,
                                         peak_epoch=peak, **scores))
                        r = rows[-1]
                        print(f"    {arm:6s} {head_kind:9s} m={margin:.1f} k={subcenters} "
                              f"seed {seed}: tune mAP@R {r['mAP@R']:.4f} @ epoch {peak}")
                        pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"    wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
