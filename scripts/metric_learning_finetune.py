"""Does the angular margin pay off once the backbone is allowed to move?

sessions/metric_learning.ipynb trains its heads on frozen EfficientNetV2S features,
on frozen EfficientNetV2S features. The obvious objection is that frozen features give
a head nothing to reshape. This script is the measurement behind the table in that
notebook's discussion: the same trunk and the same two heads, but with the last stage
of the backbone unfrozen.

Keras 3 on the JAX backend, and a GPU -- each configuration is about two minutes on an
RTX A4000 and roughly two orders of magnitude slower on a CPU. Run from the repository
root, after sessions/transfer.ipynb or scripts/metric_learning_features.py has written
data/cub_images_224.npy:

    KERAS_BACKEND=jax python scripts/metric_learning_finetune.py

The recipe is the one sessions/transfer.ipynb arrives at, including **LP-FT**
(Kumar et al. 2022): the head is trained first on the frozen features -- which is free,
they are already cached -- and only then is the backbone unfrozen, so that the large
gradients from a randomly initialized head never reach the pretrained weights. Beyond
that: only `block6*` and `top_*` are trainable, every BatchNormalization layer keeps its
ImageNet statistics, and the learning rate warms up for one epoch and then decays on a
cosine.

Each configuration gets its *own* epoch budget, chosen the same way the notebook chooses
everything else: fit on species 1-80, score retrieval on the held-out species 81-100
after every epoch, and keep the epoch where that peaks. Then refit on all 100 species
for that many epochs and report on the 100 species nobody has touched. Sharing one fixed
budget across configurations is not fair to the margin -- it moves the optimum earlier
and steepens the decay after it, so a budget chosen on one head penalizes the others.

`scale` and `margin` are re-selected here too, on the same held-out species, rather than
carried over from the frozen-feature sweep -- the frozen sweep's own lesson is that the
optimum in both moves when the regime does, so importing it would be the same mistake one
level up. The remaining caveat is that this is a single seed per configuration.
"""

import os
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

import keras
from keras import ops

DATA = "data"
DATASET_DIR = os.path.join(DATA, "CUB_200_2011")
IMAGES_CACHE = os.path.join(DATA, "cub_images_224.npy")

SEED = 23
IMG_SIZE = 224
NSEEN = 100                 # species 1..100 are trained on, 101..200 are never seen
NFIT = 80                   # species 1..80 fit, 81..100 choose the epoch budget
EMBEDDING_DIM = 512
MAX_EPOCHS = 12             # the search range for the per-configuration budget
BATCH_SIZE = 32
PEAK_LEARNING_RATE = 1e-4
PROBE_EPOCHS = 20           # head warm-up on the cached frozen features (LP-FT)
PROBE_LEARNING_RATE = 3e-4
FEATURES_CACHE = os.path.join(DATA, "cub_effnetv2s_embeddings.npy")

# The grid searched on the held-out species. `None` marks the plain softmax head;
# every other candidate is the notebook's cosine head at that scale.
SCALES = (8.0, 16.0)
CANDIDATES = [(None, None)] + [(s, 0.0) for s in SCALES]


class CosineHead(keras.layers.Layer):
    """Cosine similarity between an embedding and one learned proxy per class, scaled by `s`."""

    def __init__(self, num_classes, scale=30.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale

    def build(self, input_shape):
        self.proxies = self.add_weight(
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_uniform",
            name="proxies",
        )

    def call(self, embeddings):
        embeddings = embeddings / (ops.norm(embeddings, axis=-1, keepdims=True) + 1e-12)
        proxies = self.proxies / (ops.norm(self.proxies, axis=0, keepdims=True) + 1e-12)
        return self.scale * ops.matmul(embeddings, proxies)


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


def add_head(embedding, scale, margin, num_classes):
    """The 512-d trunk is already built; put a softmax or a cosine head on it."""
    if margin is None:
        return keras.layers.Dense(num_classes, activation="softmax", name="species")(
            embedding
        ), "categorical_crossentropy"
    return (
        CosineHead(num_classes, scale=scale, name="cosine")(embedding),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    )


def build_probe(scale, margin, num_classes, feature_dim):
    """Trunk plus head on the cached frozen features: the LP half of LP-FT."""
    keras.utils.set_random_seed(SEED)
    features = keras.Input(shape=(feature_dim,), name="features")
    embedding = keras.layers.Dense(EMBEDDING_DIM, use_bias=False, name="embedding")(features)
    embedding = keras.layers.BatchNormalization(name="embedding_bn")(embedding)
    out, loss = add_head(embedding, scale, margin, num_classes)
    model = keras.Model(features, out)
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(PROBE_LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def build(scale, margin, num_classes, epochs, steps_per_epoch, probe=None):
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

    out, loss = add_head(embedding, scale, margin, num_classes)
    model = keras.Model(backbone.input, out)

    # LP-FT: start the trunk and head where the frozen-feature probe finished
    if probe is not None:
        for name in ("embedding", "embedding_bn", "species" if margin is None else "cosine"):
            model.get_layer(name).set_weights(probe.get_layer(name).get_weights())
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=steps_per_epoch * epochs,
        warmup_target=PEAK_LEARNING_RATE,
        warmup_steps=steps_per_epoch,       # one epoch of warmup
        alpha=0.0,
    )
    model.compile(
        loss=loss, optimizer=keras.optimizers.Adam(schedule), metrics=["accuracy"]
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


def describe(scale, margin):
    return "softmax" if margin is None else f"s={scale:.0f}"


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

    # 1. search: every candidate is fitted on species 1-80 and scored, after every
    #    epoch, on the held-out species 81-100. Nothing here sees species 101-200.
    searched = {}
    for scale, margin in CANDIDATES:
        t0 = time.time()
        fit_probe = build_probe(scale, margin, NFIT, Z.shape[1])
        fit_probe.fit(x=Z[fit_idx], y=Y_fit, batch_size=128, epochs=PROBE_EPOCHS, verbose=0)
        monitor = RetrievalMonitor(X_tune, labels[tune_idx])
        search = build(scale, margin, NFIT, MAX_EPOCHS,
                       int(np.ceil(len(fit_idx) / BATCH_SIZE)), probe=fit_probe)
        search.fit(x=X_fit, y=Y_fit, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
                   callbacks=[monitor], verbose=0)
        searched[(scale, margin)] = (max(monitor.scores), int(np.argmax(monitor.scores)) + 1)
        print(f"    search {describe(scale, margin):30s} "
              f"tune mAP@R {searched[(scale, margin)][0]:.2%} "
              f"at epoch {searched[(scale, margin)][1]:2d}  ({time.time() - t0:.0f}s)", flush=True)
        del search, fit_probe

    # 2. the three heads we report: plain softmax, the best without a margin, the best with
    chosen = [
        ("softmax", (None, None)),
        ("cosine head", max((k for k in searched if k[1] is not None),
                            key=lambda k: searched[k][0])),
    ]
    print()
    for name, (scale, margin) in chosen:
        epochs = searched[(scale, margin)][1]
        probe = build_probe(scale, margin, NSEEN, Z.shape[1])
        probe.fit(x=Z[train_idx], y=Y_train, batch_size=128, epochs=PROBE_EPOCHS, verbose=0)
        model = build(scale, margin, NSEEN, epochs,
                      int(np.ceil(len(train_idx) / BATCH_SIZE)), probe=probe)
        history = model.fit(x=X_train, y=Y_train, batch_size=BATCH_SIZE,
                            epochs=epochs, verbose=0).history
        encoder = keras.Model(model.input, model.get_layer("embedding_bn").output)
        scores = retrieval_scores(encoder.predict(X_open, batch_size=64, verbose=0), y_open)
        print(f"{name:16s} {describe(scale, margin):22s} {epochs:2d} epochs  "
              f"train acc {history['accuracy'][-1]:.2%}  "
              + "  ".join(f"{k} {v:.2%}" for k, v in scores.items()), flush=True)
        del model, encoder, probe


if __name__ == "__main__":
    main()
