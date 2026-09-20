"""How much is letting the backbone move worth?

sessions/metric_learning.ipynb trains on *frozen* EfficientNetV2S features throughout,
and concludes that the backbone dominates everything done on top of it. The obvious
objection is that frozen features give a head nothing to reshape, so the notebook's
headroom claim might be an artifact of the setup. This script is the measurement behind
the fine-tuning table in that notebook's discussion: the same trunk, the same protocol,
but with the last stage of the backbone unfrozen. The answer is that unfreezing is worth
about nine points of R@1 and seven of mAP@R, which is the notebook's point rather than a
qualification of it. One seed per invocation; `--seed 23`, `24` and `25` are the three
the notebook's table quotes:

    seed 23  14 epochs  R@1 61.95%  mAP@R 23.02%
    seed 24  12 epochs  R@1 61.92%  mAP@R 22.60%
    seed 25  15 epochs  R@1 61.41%  mAP@R 22.46%

Re-running seed 23 reproduces the budget and R@1 exactly and lands at mAP@R 22.96%, so
GPU nondeterminism is worth about 0.06 points here -- an order below the seed spread, and
two orders below the effect being measured.

MAX_EPOCHS had to be raised from 12 to 30 to get those numbers. At the old cap the arm
selected 7 epochs at seed 23 and 12 -- its own cap, a wall rather than a peak -- at seed
24. The learning-rate schedule is defined over MAX_EPOCHS, so raising the cap changed
every number in the table by about a point. A budget nobody questioned was setting the
result, which is the trap the notebook is built around, here one level down.

Keras 3 on the JAX backend, and a GPU -- about twelve minutes per seed on an RTX A4000,
and roughly two orders of magnitude slower on a CPU. Run from the repository root, after
scripts/metric_learning_features.py has written the caches:

    KERAS_BACKEND=jax python scripts/metric_learning_finetune.py

The recipe is the one sessions/transfer.ipynb arrives at, including **LP-FT**
(Kumar et al. 2022): the head is trained first on the frozen features -- which is free,
they are already cached -- and only then is the backbone unfrozen, so that the large
gradients from a randomly initialized head never reach the pretrained weights. Beyond
that, only `block6*` and `top_*` are trainable, every BatchNormalization layer keeps its
ImageNet statistics, and the learning rate warms up for one epoch and then decays on a
cosine.

The epoch budget is selected inside this regime rather than imported from a
frozen-feature run: fit on species 1-80, score retrieval on the held-out species 81-100
after every epoch, keep the peak, then refit on all 100 species for that long and report
on the 100 species nobody has touched. The optimum moves when the regime does -- the
budget here is about half the frozen-feature one -- so carrying it over would be the same
mistake the notebook warns about, one level up.

The cosine schedule is defined over MAX_EPOCHS in *both* phases rather than over the
length of each run, so that epoch e sits at the same learning rate whether it is a search
epoch or a refit epoch. Sizing it to the run length instead would choose the budget under
one schedule and spend it under another, which is the same class of mistake as sharing a
budget across configurations.

There is no head comparison here. The notebook reports one embedding model, the plain
softmax baseline, and this script measures that model in one more regime; the normalized
cosine head belongs to the notebook's exercises, not to its results.
"""

import argparse
import os
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

import keras

DATA = "data"
DATASET_DIR = os.path.join(DATA, "CUB_200_2011")
IMAGES_CACHE = os.path.join(DATA, "cub_images_224.npy")

SEED = 23
IMG_SIZE = 224
NSEEN = 100                 # species 1..100 are trained on, 101..200 are never seen
NFIT = 80                   # species 1..80 fit, 81..100 choose the epoch budget
EMBEDDING_DIM = 512
MAX_EPOCHS = 30             # the search range for the per-configuration budget. A run whose
                            # best epoch equals this has found a wall, not a peak: seed 24
                            # selected 12 of 12 under the previous cap.
BATCH_SIZE = 32
PEAK_LEARNING_RATE = 1e-4
PROBE_EPOCHS = 20           # head warm-up on the cached frozen features (LP-FT)
PROBE_LEARNING_RATE = 3e-4
FEATURES_CACHE = os.path.join(DATA, "cub_effnetv2s_embeddings.npy")


def normalize(V):
    V = np.asarray(V, dtype="float32")
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def retrieval_scores(embeddings, y, ks=(1, 5, 10)):
    """Recall@K and mAP@R, identical to the implementation in the notebook."""
    E = normalize(embeddings)
    similarity = E @ E.T
    np.fill_diagonal(similarity, -np.inf)
    ranking = np.argsort(-similarity, axis=1)
    hit = y[ranking] == y[:, None]

    scores = {"R@{}".format(k): hit[:, :k].any(axis=1).mean() for k in ks}

    R = np.bincount(y)[y] - 1
    depth = int(R.max())
    relevant = hit[:, :depth]
    precision = np.cumsum(relevant, axis=1) / np.arange(1, depth + 1)
    within_R = np.arange(depth)[None, :] < R[:, None]
    scores["mAP@R"] = ((precision * relevant * within_R).sum(axis=1) / np.maximum(R, 1)).mean()
    return scores


def load_split():
    images_df = pd.read_csv(
        os.path.join(DATASET_DIR, "images.txt"), sep=" ", names=["image_id", "filename"]
    )
    labels_df = pd.read_csv(
        os.path.join(DATASET_DIR, "image_class_labels.txt"),
        sep=" ",
        names=["image_id", "class_id"],
    )
    labels = images_df.merge(labels_df, on="image_id").class_id.to_numpy() - 1
    if not os.path.exists(IMAGES_CACHE):
        raise SystemExit(
            f"{IMAGES_CACHE} is missing; run the decoding cell in sessions/transfer.ipynb "
            "or sessions/metric_learning.ipynb first."
        )
    return np.load(IMAGES_CACHE, mmap_mode="r"), labels


def build_probe(num_classes, feature_dim):
    """Trunk plus head on the cached frozen features: the LP half of LP-FT."""
    keras.utils.set_random_seed(SEED)
    features = keras.Input(shape=(feature_dim,), name="features")
    embedding = keras.layers.Dense(EMBEDDING_DIM, use_bias=False, name="embedding")(features)
    embedding = keras.layers.BatchNormalization(name="embedding_bn")(embedding)
    out = keras.layers.Dense(num_classes, activation="softmax", name="species")(embedding)
    model = keras.Model(features, out)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(PROBE_LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def build(num_classes, steps_per_epoch, probe=None):
    keras.utils.set_random_seed(SEED)
    backbone = keras.applications.EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = True
    for layer in backbone.layers:
        # only the last stage and the final convolution are adapted
        if not (layer.name.startswith("block6") or layer.name.startswith("top")):
            layer.trainable = False
        # batch-norm statistics come from ImageNet and stay there
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    embedding = keras.layers.Dense(EMBEDDING_DIM, use_bias=False, name="embedding")(backbone.output)
    embedding = keras.layers.BatchNormalization(name="embedding_bn")(embedding)

    out = keras.layers.Dense(num_classes, activation="softmax", name="species")(embedding)
    model = keras.Model(backbone.input, out)

    # LP-FT: start the trunk and head where the frozen-feature probe finished
    if probe is not None:
        for name in ("embedding", "embedding_bn", "species"):
            model.get_layer(name).set_weights(probe.get_layer(name).get_weights())
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        # over MAX_EPOCHS, not over the run length: the search and the refit must
        # put epoch e at the same learning rate, or the budget is chosen under one
        # schedule and spent under another
        decay_steps=steps_per_epoch * MAX_EPOCHS,
        warmup_target=PEAK_LEARNING_RATE,
        warmup_steps=steps_per_epoch,       # one epoch of warmup
        alpha=0.0,
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(schedule),
        metrics=["accuracy"],
    )
    return model


class RetrievalMonitor(keras.callbacks.Callback):
    """Score retrieval on the held-out species 81-100 after every epoch."""

    def __init__(self, images, y):
        super().__init__()
        self.images, self.y = images, y
        self.scores = []

    def on_epoch_end(self, epoch, logs=None):
        encoder = keras.Model(self.model.input, self.model.get_layer("embedding_bn").output)
        embeddings = encoder.predict(self.images, batch_size=64, verbose=0)
        self.scores.append(retrieval_scores(embeddings, self.y)["mAP@R"])


def main():
    images, labels = load_split()
    train_idx = np.flatnonzero(labels < NSEEN)
    fit_idx = np.flatnonzero(labels < NFIT)
    tune_idx = np.flatnonzero((labels >= NFIT) & (labels < NSEEN))
    open_idx = np.flatnonzero(labels >= NSEEN)
    y_open = labels[open_idx]

    if not os.path.exists(FEATURES_CACHE):
        raise SystemExit(
            f"{FEATURES_CACHE} is missing; run scripts/metric_learning_features.py first."
        )
    Z = np.load(FEATURES_CACHE)

    X_train = np.asarray(images[train_idx])
    Y_train = keras.utils.to_categorical(labels[train_idx], NSEEN).astype("float32")
    X_fit = np.asarray(images[fit_idx])
    Y_fit = keras.utils.to_categorical(labels[fit_idx], NFIT).astype("float32")
    X_tune = np.asarray(images[tune_idx])
    X_open = np.asarray(images[open_idx])

    print(f"{len(fit_idx):,} images of species 1-{NFIT} to fit, "
          f"{len(tune_idx):,} of species {NFIT + 1}-{NSEEN} to choose the budget, "
          f"{len(train_idx):,} of species 1-{NSEEN} to refit, "
          f"{len(open_idx):,} of species {NSEEN + 1}-200 to report")

    # 1. search: fit on species 1-80 and score, after every epoch, on the held-out
    #    species 81-100. Nothing here sees species 101-200.
    t0 = time.time()
    fit_probe = build_probe(NFIT, Z.shape[1])
    fit_probe.fit(x=Z[fit_idx], y=Y_fit, batch_size=128, epochs=PROBE_EPOCHS, verbose=0)
    monitor = RetrievalMonitor(X_tune, labels[tune_idx])
    search = build(NFIT, int(np.ceil(len(fit_idx) / BATCH_SIZE)), probe=fit_probe)
    search.fit(x=X_fit, y=Y_fit, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
               callbacks=[monitor], verbose=0)
    epochs = int(np.argmax(monitor.scores)) + 1
    print(f"    search  tune mAP@R {max(monitor.scores):.2%} at epoch {epochs:2d} "
          f"of {MAX_EPOCHS}  ({time.time() - t0:.0f}s)", flush=True)
    if epochs == MAX_EPOCHS:
        print("    WARNING: the budget is at the cap, which is a wall and not a peak; "
              "raise MAX_EPOCHS and re-run every seed", flush=True)
    del search, fit_probe

    # 2. refit on all 100 training species for that long, and report once
    probe = build_probe(NSEEN, Z.shape[1])
    probe.fit(x=Z[train_idx], y=Y_train, batch_size=128, epochs=PROBE_EPOCHS, verbose=0)
    model = build(NSEEN, int(np.ceil(len(train_idx) / BATCH_SIZE)), probe=probe)
    history = model.fit(x=X_train, y=Y_train, batch_size=BATCH_SIZE,
                        epochs=epochs, verbose=0).history
    encoder = keras.Model(model.input, model.get_layer("embedding_bn").output)
    scores = retrieval_scores(encoder.predict(X_open, batch_size=64, verbose=0), y_open)
    print()
    print(f"LP-FT softmax  seed {SEED}  {epochs:2d} epochs  "
          f"train acc {history['accuracy'][-1]:.2%}  "
          + "  ".join(f"{k} {v:.2%}" for k, v in scores.items()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNetV2S on CUB and report open-set retrieval.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="random seed for every model built in this run (default 23)")
    args = parser.parse_args()
    SEED = args.seed
    print(f"seed {SEED}\n")
    main()
