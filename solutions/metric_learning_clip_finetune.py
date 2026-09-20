"""Fine-tune CLIP's vision encoder on CUB, and compare the same three heads.

sessions/metric_learning.ipynb ends with CLIP winning twice: its image encoder is a
better retrieval space than anything we trained on EfficientNetV2S, and heads trained
on its features are better still. This script asks the remaining question -- what
happens when the good backbone is also allowed to move.

Keras 3 on the JAX backend, and a GPU. Run from the repository root, after
scripts/metric_learning_features.py has written the caches:

    KERAS_BACKEND=jax python scripts/metric_learning_clip_finetune.py

The protocol is the one the notebook uses everywhere else, and nothing is carried
over between regimes:

  * LP-FT -- the head is trained first on the cached CLIP features, which is free,
    and only then is the backbone unfrozen;
  * only the last `UNFROZEN_BLOCKS` transformer blocks, the final layer norm and the
    projection are trainable;
  * `scale`, `margin` and the epoch budget are all chosen on species 81-100, held out
    from a fit on species 1-80;
  * the winner is refitted on all 100 training species and reported on the 100 species
    nobody has touched.

What it prints, on an RTX A4000, in about twenty-five minutes::

    softmax      softmax   18 epochs   R@1 71.02%   mAP@R 31.33%
    cosine head  s=8        2 epochs   R@1 73.02%   mAP@R 34.90%

against 69.0% / 29.6% and 70.2% / 31.9% for the same two heads on *frozen* CLIP features
in the notebook. So fine-tuning is worth two to three points of R@1 on top of the best
frozen-feature result, and the ordering of the heads is unchanged -- the cosine head
stays ahead, by rather more than it was ahead by before. 34.90% mAP@R is the best number
anywhere in the session.

Two things are worth noticing beyond the totals. **The selected scale moves**: the
notebook picks s=4 on frozen CLIP features and this search picks s=8, which is the whole
reason the search is re-run here rather than carried over. And **the budgets collapse**:
the softmax head wants 18 epochs, the cosine head peaks after **two** and is past its
best by the third. Whatever the normalization does to the optimization, it arrives almost
immediately.

Two differences from scripts/metric_learning_finetune.py are worth noting.Two differences from scripts/metric_learning_finetune.py are worth noting.Two differences from scripts/metric_learning_finetune.py are worth noting. A ViT has
**no BatchNormalization** -- LayerNorm keeps no running statistics -- so the "freeze
batch-norm" rule that matters so much for EfficientNetV2 has nothing to act on here.
And `vision_projection` does not normalize its output, while the cached features the
notebook trains on are L2-normalized, so the model applies `UnitNormalization` to keep
the two paths identical.
"""

import math
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
FEATURES_CACHE = os.path.join(DATA, "cub_clip_vitb16_images.npy")
CLIP_PRESET = "clip_vit_base_patch16"

SEED = 23
NSEEN = 100
NFIT = 80
EMBEDDING_DIM = 512
UNFROZEN_BLOCKS = 4          # of the twelve; the rest keep their pretrained weights
MAX_EPOCHS = 20          # generous on purpose: a search that selects its own last
                        # epoch has not found a peak, it has hit a wall
BATCH_SIZE = 32
PEAK_LEARNING_RATE = 1e-5    # an order below the EfficientNet fine-tune: 86M pretrained
PROBE_EPOCHS = 20            # parameters and 5,864 images is a delicate combination
PROBE_LEARNING_RATE = 3e-4

SCALES = (4.0, 8.0)
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
    hit = y[np.argsort(-similarity, axis=1)] == y[:, None]
    scores = {"R@{}".format(k): hit[:, :k].any(axis=1).mean() for k in ks}
    R = np.bincount(y)[y] - 1
    depth = int(R.max())
    relevant = hit[:, :depth]
    precision = np.cumsum(relevant, axis=1) / np.arange(1, depth + 1)
    within_R = np.arange(depth)[None, :] < R[:, None]
    scores["mAP@R"] = ((precision * relevant * within_R).sum(axis=1) / np.maximum(R, 1)).mean()
    return scores


class RetrievalMonitor(keras.callbacks.Callback):
    """Score retrieval on the held-out species 81-100 after every epoch."""

    def __init__(self, images, y):
        super().__init__()
        self.images, self.y = images, y
        self.scores = []
        self.encoder = None

    def on_train_begin(self, logs=None):
        self.encoder = keras.Model(self.model.input, self.model.get_layer("embedding_bn").output)

    def on_epoch_end(self, epoch, logs=None):
        embeddings = self.encoder.predict(self.images, batch_size=64, verbose=0)
        self.scores.append(retrieval_scores(embeddings, self.y)["mAP@R"])


def describe(scale, margin):
    return "softmax" if margin is None else f"s={scale:.0f}"


def add_head(embedding, scale, margin, num_classes):
    if margin is None:
        return (
            keras.layers.Dense(num_classes, activation="softmax", name="species")(embedding),
            "categorical_crossentropy",
        )
    return (
        CosineHead(num_classes, scale=scale, name="cosine")(embedding),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    )


def build_probe(scale, margin, num_classes, feature_dim):
    """Trunk plus head on the cached CLIP features: the LP half of LP-FT."""
    keras.utils.set_random_seed(SEED)
    features = keras.Input(shape=(feature_dim,), name="features")
    embedding = keras.layers.Dense(EMBEDDING_DIM, use_bias=False, name="embedding")(features)
    embedding = keras.layers.BatchNormalization(name="embedding_bn")(embedding)
    out, loss = add_head(embedding, scale, margin, num_classes)
    model = keras.Model(features, out)
    model.compile(
        loss=loss, optimizer=keras.optimizers.Adam(PROBE_LEARNING_RATE), metrics=["accuracy"]
    )
    return model


def build(scale, margin, num_classes, epochs, steps_per_epoch, probe=None):
    from keras_hub.layers import CLIPImageConverter
    from keras_hub.models import CLIPBackbone

    keras.utils.set_random_seed(SEED)
    clip = CLIPBackbone.from_preset(CLIP_PRESET)
    vision = clip.vision_encoder

    # only the last few transformer blocks, the final norm and the projection move.
    # Note there is no BatchNormalization anywhere in a ViT, so unlike the
    # EfficientNetV2 fine-tune there are no running statistics to protect.
    trainable_blocks = {
        f"clip_vision_encoder_encoder_block_{i}"
        for i in range(12 - UNFROZEN_BLOCKS, 12)
    }
    for layer in vision.layers:
        layer.trainable = (
            layer.name in trainable_blocks or layer.name == "clip_vision_encoder_layer_norm"
        )

    images = keras.Input(shape=(224, 224, 3), name="images")
    converted = CLIPImageConverter.from_preset(CLIP_PRESET)(images)
    pooled = clip.vision_pooler(vision({"images": converted}))
    projected = clip.vision_projection(pooled)
    # the cached features the notebook trains on are L2-normalized; match them
    features = keras.layers.UnitNormalization(name="clip_embedding")(projected)

    embedding = keras.layers.Dense(EMBEDDING_DIM, use_bias=False, name="embedding")(features)
    embedding = keras.layers.BatchNormalization(name="embedding_bn")(embedding)
    out, loss = add_head(embedding, scale, margin, num_classes)
    model = keras.Model(images, out)

    if probe is not None:
        for name in ("embedding", "embedding_bn", "species" if margin is None else "cosine"):
            model.get_layer(name).set_weights(probe.get_layer(name).get_weights())

    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=steps_per_epoch * epochs,
        warmup_target=PEAK_LEARNING_RATE,
        warmup_steps=steps_per_epoch,
        alpha=0.0,
    )
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(schedule), metrics=["accuracy"])
    return model


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
    for path in (IMAGES_CACHE, FEATURES_CACHE):
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing; run scripts/metric_learning_features.py first.")
    return np.load(IMAGES_CACHE, mmap_mode="r"), labels


def main():
    images, labels = load_split()
    Z = np.load(FEATURES_CACHE)

    train_idx = np.flatnonzero(labels < NSEEN)
    fit_idx = np.flatnonzero(labels < NFIT)
    tune_idx = np.flatnonzero((labels >= NFIT) & (labels < NSEEN))
    open_idx = np.flatnonzero(labels >= NSEEN)
    y_open = labels[open_idx]

    X_train, X_fit = np.asarray(images[train_idx]), np.asarray(images[fit_idx])
    X_tune, X_open = np.asarray(images[tune_idx]), np.asarray(images[open_idx])
    Y_train = keras.utils.to_categorical(labels[train_idx], NSEEN).astype("float32")
    Y_fit = keras.utils.to_categorical(labels[fit_idx], NFIT).astype("float32")

    print(f"CLIP {CLIP_PRESET}, last {UNFROZEN_BLOCKS} of 12 blocks trainable")
    print(f"{len(fit_idx):,} images to fit, {len(tune_idx):,} to choose on, "
          f"{len(train_idx):,} to refit, {len(open_idx):,} to report\n")

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
        print(f"    search {describe(scale, margin):16s} tune mAP@R {searched[(scale, margin)][0]:.2%} "
              f"at epoch {searched[(scale, margin)][1]:2d}  ({time.time() - t0:.0f}s)", flush=True)
        del search, fit_probe

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
        print(f"{name:16s} {describe(scale, margin):14s} {epochs:2d} epochs  "
              f"train acc {history['accuracy'][-1]:.2%}  "
              + "  ".join(f"{k} {v:.2%}" for k, v in scores.items()), flush=True)
        del model, encoder, probe


if __name__ == "__main__":
    main()
