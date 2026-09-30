"""Train the sub-center ArcFace re-identification models for sessions/reid.ipynb.

Keras 3 on the JAX backend. Run from the repository root, after scripts/reid_data.py:

    KERAS_BACKEND=jax python scripts/reid_arcface.py            # the whole grid
    KERAS_BACKEND=jax python scripts/reid_arcface.py --arm time --head subcenter

Every configuration is trained under *both* splits built by scripts/reid_data.py --
the random arm and the time-aware arm -- because the notebook's headline result is
those two numbers side by side. Nothing here is allowed to differ between the arms
except the split itself.

What is trained
---------------
A 512-d embedding fitted on the *frozen* EfficientNetV2S features cached by
scripts/reid_data.py -- the backbone is never fine-tuned here, exactly as in the
companion notebook, so every run costs seconds on a CPU -- under one of three
heads:

  plain       cosine head with an angular margin, one proxy per individual
  subcenter   k = 3 proxies per individual, the cosine taken as the max over them
  dynamic     sub-centers, plus a per-individual margin set by how many photographs
              that individual has

Outputs per configuration, under data/ and gitignored:

  turtle_arcface_{arm}_{head}_s{seed}.keras            the trained model
  turtle_arcface_{arm}_{head}_s{seed}_history.p        the training history
  turtle_arcface_{arm}_{head}_s{seed}_embeddings.npy   (7582, 512) float32,
                                                       L2-normalized, every image

Choosing the epoch budget
-------------------------
The budget is a hyperparameter like any other, so it is chosen on the 50 `tune`
individuals -- never trained on, never reported -- by scoring open-set retrieval
after every epoch and taking the peak. `MAX_EPOCHS` is deliberately well above
every peak observed: a run whose best epoch is its own cap has not found an
optimum, it has hit a wall.

Frozen BN
---------
`build_finetune_trunk` is not used by the grid above. It exists because the notebook
documents what unfreezing a block of the backbone would require, and a claim that is
only asserted is worth less than one that runs: a BatchNormalization layer with
`trainable=False` in Keras also runs in *inference* mode, using the moving statistics
it already has rather than the batch's. That is not a side effect to work around, it
is the behaviour re-identification practice wants -- fine-tuning batches are small and
dominated by a handful of individuals, so batch statistics are a poor estimate of the
population's, and letting them drift is the usual reason "fine-tuning made it worse".
Note the ordering trap: setting `backbone.trainable = True` re-enables *every* layer,
so the BatchNormalization layers have to be switched off again afterwards.

Two learning rates
------------------
The head owns learnable proxies, and the standard protocols do not let them share
an optimizer with the backbone (Musgrave et al., arXiv:2003.08505, §3.1). Keras
optimizers have no per-layer learning rate, so the head uses the reparameterization
`w = raw * PROXY_LR_MULT`: Adam moves `raw` by roughly `lr` per step whatever the
gradient scale, so the effective rate on `w` is `lr * PROXY_LR_MULT`.
"""

import argparse
import os
import pickle
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

import keras
from keras import ops

DATA = "data"
CATALOGUE = os.path.join(DATA, "turtle_catalogue.csv")
IMAGES_CACHE = os.path.join(DATA, "turtle_images_224.npy")
FEATURES_CACHE = os.path.join(DATA, "turtle_effnetv2s_embeddings.npy")
IMG_SIZE = 224
EMBEDDING_DIM = 512
SEED = 42
SEEDS = (SEED, SEED + 1, SEED + 2)
ARMS = ("random", "time")
HEADS = ("plain", "subcenter", "dynamic")
# Swept, not assumed. A margin held fixed across the comparison would be measuring
# itself: see the sweep in scripts/reid_sweep.py and the discussion in the notebook.
# Both grids were extended once already: the first pass selected margin 0.5 and
# k = 4, which were its own largest values, and a search that stops at the edge of
# its range has found a wall rather than an optimum.
MARGIN_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SUBCENTER_GRID = (1, 2, 3, 4, 6, 8)

BATCH_SIZE = 128
TRUNK_LR = 3e-4
PROXY_LR_MULT = 8.0
MAX_EPOCHS = 64
SCALE = 30.0
MARGIN = 0.3
MARGIN_RANGE = (0.15, 0.45)   # dynamic margins: the tail gets the wide one
MARGIN_LAMBDA = 0.25
SUBCENTERS = 3
UNFREEZE_FROM = "block6a"     # the last block of EfficientNetV2S

# These are not tuned here. They are the recipe the lab already uses to fine-tune
# MiewID on this species -- `scripts/finetune_turtles.py` in the AnimalCLEF-2026 work,
# itself "the same recipe as CzechLynx v2" -- transplanted onto our backbone:
#
#   backbone 5e-6, projection and ArcFace 1e-4   (separate rates, as Musgrave
#     arXiv:2003.08505 sec. 3.1 prescribes for a loss with its own learnable weights)
#   AdamW, cosine annealing to 1e-6 over 50 epochs, batch 32
#   ArcFace s = 64, m = 0.5, one centre, 512-d embedding behind a Linear+BatchNorm
#   MPerClassSampler(m = 1), which is shuffling -- see the batch-composition section
#
# Keras has no per-layer learning rate, so the 20x that the projection and the head
# get over the backbone is supplied by the `w = raw * mult` reparameterization the
# notebook explains: Adam moves `raw` by about lr per step whatever the gradient.
FINETUNE_BACKBONE_LR = 5e-6
FINETUNE_HEAD_LR = 1e-4
FINETUNE_LR_MULT = FINETUNE_HEAD_LR / FINETUNE_BACKBONE_LR     # 20
FINETUNE_MAX_EPOCHS = 50
FINETUNE_BATCH = 32
FINETUNE_MIN_LR = 1e-6
FINETUNE_SCALE = 64.0
FINETUNE_MARGIN = 0.5
FINETUNE_LR = FINETUNE_BACKBONE_LR


# --------------------------------------------------------------------------- head


@keras.saving.register_keras_serializable(package="reid")
class ArcFaceHead(keras.layers.Layer):
    """Sub-center ArcFace: a cosine classifier with an additive angular margin.

    The weight tensor holds `k` proxies per individual, `(d, C, k)`. An embedding is
    scored against every proxy and only the closest one counts, so an individual
    whose photographs fall into several clusters -- a bad crop, an odd pose, an
    occlusion -- can own a proxy for each instead of being forced into one mean.

    `margins` may be a scalar or one margin per individual.
    """

    def __init__(self, num_classes, scale=SCALE, margins=MARGIN, subcenters=1,
                 proxy_lr_mult=PROXY_LR_MULT, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margins = margins
        self.subcenters = subcenters
        self.proxy_lr_mult = proxy_lr_mult

    def build(self, input_shape):
        # w = raw * mult is how this layer gets its own learning rate: Adam moves
        # raw by about lr per step, so w moves by about lr * mult.
        self.raw = self.add_weight(
            shape=(input_shape[-1], self.num_classes, self.subcenters),
            initializer=keras.initializers.GlorotUniform(),
            name="proxies",
        )
        margins = np.broadcast_to(np.asarray(self.margins, dtype="float32"), (self.num_classes,))
        self.margin_vector = self.add_weight(
            shape=(self.num_classes,),
            initializer=keras.initializers.Constant(margins),
            trainable=False,
            name="margins",
        )

    def cosine(self, embeddings):
        """cos of the angle to the nearest sub-center of every individual."""
        x = embeddings / (ops.norm(embeddings, axis=-1, keepdims=True) + 1e-12)
        w = self.raw * self.proxy_lr_mult
        w = w / (ops.norm(w, axis=0, keepdims=True) + 1e-12)
        cos = ops.einsum("bd,dck->bck", x, w)
        return ops.max(cos, axis=-1)

    def call(self, inputs, labels=None, training=False):
        # Whether the margin is applied is decided by the presence of labels, not by
        # the `training` flag: Keras sets that flag itself inside a functional graph,
        # and the margin needs a true class to be applied to.
        cos = self.cosine(inputs)
        if labels is None:
            return self.scale * cos

        # sqrt(1 - cos^2) is the whole reason this clip exists: cos drifting to
        # exactly +/-1 in float32 makes the gradient of the square root infinite.
        cos = ops.clip(cos, -1.0 + 1e-7, 1.0 - 1e-7)
        sin = ops.sqrt(1.0 - ops.square(cos))
        m = ops.take(self.margin_vector, ops.cast(labels, "int32"))[:, None]
        cos_m, sin_m = ops.cos(m), ops.sin(m)
        target = cos * cos_m - sin * sin_m
        # Beyond theta > pi - m the rotated cosine starts increasing again; keep the
        # penalty monotone there, as the ArcFace paper does.
        target = ops.where(cos > ops.cos(np.pi - m), target, cos - m * ops.sin(m))
        onehot = ops.one_hot(ops.cast(labels, "int32"), self.num_classes)
        return self.scale * ops.where(ops.cast(onehot, "bool"), target, cos)

    def get_config(self):
        config = super().get_config()
        margins = self.margins
        config.update(
            num_classes=self.num_classes,
            scale=self.scale,
            margins=np.asarray(margins).tolist() if np.ndim(margins) else float(margins),
            subcenters=self.subcenters,
            proxy_lr_mult=self.proxy_lr_mult,
        )
        return config


def build_trunk(seed, input_dim, embedding_dim=EMBEDDING_DIM):
    """Frozen features -> a 512-d embedding. The backbone is not touched.

    The same shape as the companion notebook's trunk, so that the only thing that
    differs between the two sessions is what sits on top of it.
    """
    initializer = keras.initializers.GlorotUniform(seed=seed)
    features = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(embedding_dim, activation="relu", kernel_initializer=initializer)(features)
    x = keras.layers.BatchNormalization()(x)
    embeddings = keras.layers.Dense(
        embedding_dim, use_bias=False, kernel_initializer=initializer, name="embedding"
    )(x)
    return keras.Model(features, embeddings, name="trunk")


@keras.saving.register_keras_serializable(package="reid")
class ScaledDense(keras.layers.Layer):
    """A Dense layer with a learning rate of its own, by the same trick as the head.

    The recipe gives the projection 1e-4 while the backbone gets 5e-6. Keras optimizers
    apply one rate to everything, so the kernel is stored as `raw` and used as
    `raw * lr_mult`.
    """

    def __init__(self, units, lr_mult=FINETUNE_LR_MULT, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.lr_mult = lr_mult
        self.seed = seed

    def build(self, input_shape):
        self.raw = self.add_weight(
            shape=(input_shape[-1], self.units), name="kernel",
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.raw * self.lr_mult)

    def get_config(self):
        return {**super().get_config(), "units": self.units,
                "lr_mult": self.lr_mult, "seed": self.seed}


def build_finetune_trunk(seed, unfreeze_from=UNFREEZE_FROM, lr_mult=1.0):
    """What fine-tuning the backbone would look like, with BatchNorm kept frozen.

    Not used by the grid -- see "Frozen BN" in the module docstring. The order of
    the two `trainable` assignments is the whole point: `backbone.trainable = True`
    re-enables every layer it contains, so the BatchNormalization layers must be
    switched off *after* it, not before.
    """
    backbone = keras.applications.EfficientNetV2S(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = True
    unfrozen = False
    for layer in backbone.layers:
        if layer.name.startswith(unfreeze_from):
            unfrozen = True
        layer.trainable = unfrozen
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    initializer = keras.initializers.GlorotUniform(seed=seed)
    images = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    features = backbone(images)
    if lr_mult == 1.0:
        embeddings = keras.layers.Dense(
            EMBEDDING_DIM, use_bias=False, kernel_initializer=initializer, name="embedding"
        )(features)
    else:
        # The recipe's projection: Linear then BatchNorm, at the head's learning rate.
        embeddings = ScaledDense(EMBEDDING_DIM, lr_mult=lr_mult, seed=seed,
                                 name="embedding")(features)
        embeddings = keras.layers.BatchNormalization(name="embedding_bn")(embeddings)
    return keras.Model(images, embeddings, name="finetune_trunk")


def augmentation(seed):
    """The recipe's augmentation, in the layers this course already uses.

    Horizontal flip is safe here and the lab's script says why: sea turtles are
    bilaterally symmetric, so a mirrored head crop is still a plausible photograph of
    the same individual. `RandomTranslation` is deliberately absent -- on this backend
    it costs far more than the rest of the model put together.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomBrightness(0.3, value_range=(0, 255), seed=seed),
        keras.layers.RandomContrast(0.3, seed=seed),
    ], name="augmentation")


def build_model(trunk, head, learning_rate=TRUNK_LR):
    """Trunk plus head, as a two-input model.

    The angular margin is applied to the true class, so the head needs the label as
    an *input* rather than only as a target. Training therefore runs on a model of
    two inputs; the embedding model used at inference is the trunk alone, and the
    head -- proxies and all -- is thrown away, exactly as in the companion notebook.
    """
    labels = keras.Input(shape=(), dtype="int32", name="label")
    logits = head(trunk.output, labels=labels, training=True)
    model = keras.Model([trunk.input, labels], logits, name="arcface")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def margins_from_counts(counts, span=MARGIN_RANGE, lam=MARGIN_LAMBDA):
    """A wider angular margin for individuals with fewer photographs.

    The long tail is a property of the *label distribution*, not of ArcFace: an
    individual with four photographs gets four gradient steps' worth of proxy
    updates per epoch, and its proxy stays where initialization left it. Asking for
    a larger angular separation on those classes is the standard remedy (Ha et al.,
    Google Landmark 2020): m_c = a * n_c^-lambda + b, with a and b fixed by the
    ends of the requested range.
    """
    lo, hi = span
    scaled = np.asarray(counts, dtype="float64") ** -lam
    a = (hi - lo) / (scaled.max() - scaled.min())
    b = hi - a * scaled.max()
    return (a * scaled + b).astype("float32")


# --------------------------------------------------------------------------- data


def load_split(arm):
    """Catalogue, frozen features and the label vector for one arm."""
    df = pd.read_csv(CATALOGUE)
    features = np.load(FEATURES_CACHE)
    column = f"arm_{arm}"

    fit = df[(df.role == "train") & (df[column] == "database")]
    identities = np.sort(fit.identity.unique())
    lookup = {identity: i for i, identity in enumerate(identities)}
    labels = fit.identity.map(lookup).to_numpy("int32")
    counts = fit.identity.value_counts().reindex(identities).to_numpy()
    return df, features, fit.index.to_numpy(), labels, counts, identities


def retrieval_scores(embeddings, labels, ks=(1, 5)):
    """Recall@K and mAP@R, on unit-normalized embeddings, leave-one-out."""
    E = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-12)
    similarity = E @ E.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)
    hits = labels[order] == labels[:, None]
    scores = {f"recall@{k}": float(hits[:, :k].any(axis=1).mean()) for k in ks}

    relevant = np.bincount(labels)[labels] - 1        # R, per query
    average_precisions = []
    for row, R in zip(hits, relevant):
        if R <= 0:
            continue
        top = row[:R]
        precision = np.cumsum(top) / np.arange(1, R + 1)
        average_precisions.append(float((precision * top).sum() / R))
    scores["mAP@R"] = float(np.mean(average_precisions))
    return scores


def gallery_scores(query, query_labels, database, database_labels, ks=(1, 5)):
    """Recall@K and mAP@R for queries searched against a separate database.

    Leave-one-out scoring over a pooled set would quietly undo the split: a time-arm
    query would be allowed to retrieve its own encounter-mates. Queries and database
    are kept apart here for exactly the reason the two arms exist.
    """
    Q = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-12)
    D = database / (np.linalg.norm(database, axis=-1, keepdims=True) + 1e-12)
    order = np.argsort(-(Q @ D.T), axis=1)
    hits = database_labels[order] == query_labels[:, None]
    scores = {f"recall@{k}": float(hits[:, :k].any(axis=1).mean()) for k in ks}

    available = np.bincount(database_labels, minlength=int(database_labels.max()) + 1)
    relevant = available[query_labels]           # R, per query
    average_precisions = []
    for row, R in zip(hits, relevant):
        if R <= 0:
            continue
        top = row[:R]
        precision = np.cumsum(top) / np.arange(1, R + 1)
        average_precisions.append(float((precision * top).sum() / R))
    scores["mAP@R"] = float(np.mean(average_precisions))
    return scores


class RetrievalMonitor(keras.callbacks.Callback):
    """Score open-set retrieval on the tune individuals after every epoch.

    These 50 individuals are never trained on and never reported. They exist so
    that the epoch budget -- and every threshold the notebook needs later -- is
    chosen without touching the numbers the notebook publishes.

    The scoring here obeys the *arm's own* database/query assignment. An epoch
    budget chosen under a random split and then spent on the time-aware arm would
    be a hyperparameter measuring itself, which is the confound this whole session
    is about.
    """

    def __init__(self, trunk, query, query_labels, database, database_labels, batch_size=512):
        super().__init__()
        self.trunk = trunk
        self.query = query
        self.query_labels = query_labels
        self.database = database
        self.database_labels = database_labels
        self.batch_size = batch_size
        self.history = []
        self.best = -np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        embed = lambda X: self.trunk.predict(X, batch_size=self.batch_size, verbose=0)
        scores = gallery_scores(
            embed(self.query), self.query_labels, embed(self.database), self.database_labels
        )
        self.history.append(scores)
        # The budget chosen on the tune individuals is the budget that has to be
        # spent: keep the weights from the peak rather than whatever the last epoch
        # happens to leave behind.
        if scores["mAP@R"] > self.best:
            self.best = scores["mAP@R"]
            self.best_weights = [np.array(w) for w in self.trunk.get_weights()]
        (logs or {}).update({f"tune_{k}": v for k, v in scores.items()})
        # flush: these runs are long and usually watched through a redirected log,
        # where the default block buffering hides progress for tens of minutes.
        print(f"      epoch {epoch + 1:2d}  tune mAP@R {scores['mAP@R']:.4f}  "
              f"recall@1 {scores['recall@1']:.4f}", flush=True)


# ----------------------------------------------------------------------- training


def selected_settings(arm, head_kind, sweep=os.path.join(DATA, "turtle_arcface_sweep.csv")):
    """The margin and sub-center count the sweep chose for this configuration.

    Chosen on the tune individuals, per configuration. Sharing one margin between
    the arms of a comparison would make the margin part of what is being compared.
    """
    rows = pd.read_csv(sweep)
    rows = rows[(rows.arm == arm) & (rows["head"] == head_kind)]
    if head_kind != "plain":
        # k = 1 is the plain head under another name; the comparison this feeds is
        # "the best sub-centered configuration against the best plain one".
        rows = rows[rows.subcenters >= 2]
    means = rows.groupby(["margin", "subcenters"])["mAP@R"].mean()
    margin, subcenters = means.idxmax()
    if margin in (min(MARGIN_GRID), max(MARGIN_GRID)) or subcenters == max(SUBCENTER_GRID):
        print(f"    WARNING: {arm}/{head_kind} selected an edge of the grid "
              f"(margin {margin}, k {subcenters}); extend it")
    return float(margin), int(subcenters)


def m_per_class_batches(labels, batch_size, m, generator):
    """Batch indices holding m images each of batch_size // m individuals.

    The sampler `pytorch-metric-learning` calls MPerClassSampler, and the question the
    companion session deferred: does a proxy loss care how a batch is composed? A pair
    or triplet loss certainly does -- a batch with no positive pair contributes nothing.
    A proxy loss compares each embedding against the proxies rather than against the
    other images, so in principle it does not. The long tail is where that argument is
    worth testing rather than repeating.
    """
    by_class = {c: np.flatnonzero(labels == c) for c in np.unique(labels)}
    classes_per_batch = max(1, batch_size // m)
    order = []
    for _ in range(len(labels) // batch_size):
        chosen = generator.choice(list(by_class), size=classes_per_batch, replace=False)
        batch = [generator.choice(by_class[c], size=m, replace=len(by_class[c]) < m)
                 for c in chosen]
        order.append(np.concatenate(batch))
    return np.concatenate(order) if order else np.arange(len(labels))


def train_one(arm, head_kind, seed, max_epochs=MAX_EPOCHS, margin=None, subcenters=None,
              tag="", sampler=None):
    stem = os.path.join(DATA, f"turtle_arcface_{arm}_{head_kind}{tag}_s{seed}")
    if os.path.exists(stem + "_embeddings.npy"):
        print(f"    already present: {stem}_embeddings.npy")
        return

    keras.utils.set_random_seed(seed)
    df, features, fit_rows, labels, counts, identities = load_split(arm)
    X = features[fit_rows]

    # The tune individuals, split by this arm's own rule.
    tune = df[df.role == "tune"]
    tune_labels = pd.factorize(tune.identity)[0]
    is_query = (tune[f"arm_{arm}"] == "query").to_numpy()
    tune_rows = tune.index.to_numpy()

    if margin is None:
        margin = MARGIN
    if subcenters is None:
        subcenters = 1 if head_kind == "plain" else SUBCENTERS
    margins = margin if head_kind != "dynamic" else margins_from_counts(
        counts, span=(margin / 2, margin * 1.5))
    subcenters = 1 if head_kind == "plain" else subcenters
    trunk = build_trunk(seed, features.shape[-1])
    head = ArcFaceHead(len(identities), scale=SCALE, margins=margins, subcenters=subcenters)
    model = build_model(trunk, head)

    monitor = RetrievalMonitor(
        trunk,
        features[tune_rows[is_query]], tune_labels[is_query],
        features[tune_rows[~is_query]], tune_labels[~is_query],
    )
    t0 = time.time()
    print(f"    {arm}/{head_kind}/seed {seed}: {len(X):,} images, {len(identities)} individuals")
    if sampler is None:
        history = model.fit(
            {"features": X, "label": labels}, labels,
            batch_size=BATCH_SIZE, epochs=max_epochs, verbose=0, callbacks=[monitor],
        ).history
    else:
        # Keras reshuffles every epoch, so an m-per-class order has to be rebuilt each
        # epoch and shuffling switched off.
        generator = np.random.default_rng(seed)
        history = {}
        for epoch in range(max_epochs):
            order = m_per_class_batches(labels, BATCH_SIZE, sampler, generator)
            epoch_history = model.fit(
                {"features": X[order], "label": labels[order]}, labels[order],
                batch_size=BATCH_SIZE, epochs=1, verbose=0, shuffle=False,
                callbacks=[monitor],
            ).history
            for k, v in epoch_history.items():
                history.setdefault(k, []).extend(v)

    peak = int(np.argmax([s["mAP@R"] for s in monitor.history]))
    history["tune"] = monitor.history
    history["peak_epoch"] = peak + 1
    if peak + 1 == max_epochs:
        print(f"    WARNING: peak epoch equals the cap ({max_epochs}); raise MAX_EPOCHS")

    trunk.set_weights(monitor.best_weights)
    embeddings = trunk.predict(features, batch_size=512, verbose=0)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-12

    model.save(stem + ".keras")
    np.save(stem + "_embeddings.npy", embeddings.astype("float32"))
    with open(stem + "_history.p", "wb") as f:
        pickle.dump(history, f)
    print(f"    wrote {stem}.* -- peak epoch {peak + 1}, "
          f"tune mAP@R {monitor.history[peak]['mAP@R']:.4f}, {time.time() - t0:.0f}s")


# Fine-tuning settings, taken from the two sources that have a right to an opinion
# rather than chosen here.
#
# Musgrave, Belongie & Lim (arXiv:2003.08505) sec. 3.1: Adam, batch 32, BatchNorm frozen,
# and the trunk at a *constant* 1e-6, with the learning rate of a loss that owns
# learnable weights -- ArcFace does -- left as a separately tuned hyperparameter. That
# 1e-6 is two orders of magnitude below what re-identification codebases typically use
# on a backbone, so it is the bottom of the grid below rather than an assumption.
#
# MiewID-msv3 (conservationxlabs/miewid-msv3): sub-center ArcFace with k = 3,
# scale 51.5, dynamic margin starting at 0.5. Those are the MIEWID_* values.


def train_finetune(arm, seed, margin=FINETUNE_MARGIN, max_epochs=FINETUNE_MAX_EPOCHS,
                   learning_rate=FINETUNE_BACKBONE_LR, unfreeze_from=UNFREEZE_FROM,
                   tag="", scale=FINETUNE_SCALE, subcenters=1, augment=True):
    """Fine-tune the last block of EfficientNetV2S, with BatchNorm kept frozen.

    Same head, same three-way split of individuals and same held-out epoch budget as
    the frozen-feature runs; the only difference is that gradients now reach the
    backbone. The images go in as uint8 0-255, because EfficientNetV2 carries its own
    rescaling layer.
    """
    stem = os.path.join(DATA, f"turtle_finetune_{arm}{tag}_s{seed}")
    if os.path.exists(stem + "_embeddings.npy"):
        print(f"    already present: {stem}_embeddings.npy")
        return

    keras.utils.set_random_seed(seed)
    df = pd.read_csv(CATALOGUE)
    images = np.load(IMAGES_CACHE, mmap_mode="r")
    column = f"arm_{arm}"

    fit = df[(df.role == "train") & (df[column] == "database")]
    identities = np.sort(fit.identity.unique())
    labels = fit.identity.map({c: i for i, c in enumerate(identities)}).to_numpy("int32")
    X = np.asarray(images[fit.index.to_numpy()])

    tune = df[df.role == "tune"]
    tune_labels = pd.factorize(tune.identity)[0]
    is_query = (tune[f"arm_{arm}"] == "query").to_numpy()
    tune_rows = tune.index.to_numpy()

    trunk = build_finetune_trunk(seed, unfreeze_from=unfreeze_from,
                                 lr_mult=FINETUNE_LR_MULT)
    head = ArcFaceHead(len(identities), scale=scale, margins=margin,
                       subcenters=subcenters, proxy_lr_mult=FINETUNE_LR_MULT)

    # AdamW with cosine annealing to 1e-6, as the recipe specifies. The schedule is in
    # steps, so the epoch count has to be turned into one.
    steps = max(1, len(X) // FINETUNE_BATCH) * max_epochs
    schedule = keras.optimizers.schedules.CosineDecay(
        learning_rate, decay_steps=steps, alpha=FINETUNE_MIN_LR / learning_rate)
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = augmentation(seed)(inputs) if augment else inputs
    labels_in = keras.Input(shape=(), dtype="int32", name="label")
    model = keras.Model([inputs, labels_in],
                        head(trunk(x), labels=labels_in), name="arcface_finetune")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    monitor = RetrievalMonitor(
        trunk,
        np.asarray(images[tune_rows[is_query]]), tune_labels[is_query],
        np.asarray(images[tune_rows[~is_query]]), tune_labels[~is_query],
        batch_size=64,
    )
    t0 = time.time()
    print(f"    finetune {arm}/seed {seed}: {len(X):,} images, backbone lr {learning_rate:g}, "
          f"head lr {learning_rate * FINETUNE_LR_MULT:g}, s {scale}, m {margin}, "
          f"k {subcenters}, from {unfreeze_from}, augment {augment}")
    history = model.fit(
        {"image": X, "label": labels}, labels,
        batch_size=FINETUNE_BATCH, epochs=max_epochs, verbose=0, callbacks=[monitor],
    ).history

    peak = int(np.argmax([s["mAP@R"] for s in monitor.history]))
    history["tune"] = monitor.history
    history["peak_epoch"] = peak + 1
    if peak + 1 == max_epochs:
        print(f"    WARNING: peak epoch equals the cap ({max_epochs}); raise it")

    trunk.set_weights(monitor.best_weights)
    embeddings = trunk.predict(np.asarray(images), batch_size=64, verbose=0)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-12

    model.save(stem + ".keras")
    np.save(stem + "_embeddings.npy", embeddings.astype("float32"))
    with open(stem + "_history.p", "wb") as f:
        pickle.dump(history, f)
    print(f"    wrote {stem}.* -- peak epoch {peak + 1}, "
          f"tune mAP@R {monitor.history[peak]['mAP@R']:.4f}, {time.time() - t0:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, action="append")
    parser.add_argument("--head", choices=HEADS, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--selected", action="store_true",
                        help="use the margin and k chosen by scripts/reid_sweep.py")
    parser.add_argument("--sampler", type=int, default=None,
                        help="m images per individual per batch, instead of shuffling")
    parser.add_argument("--finetune", action="store_true",
                        help="fine-tune the backbone instead of fitting on frozen features")
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--unfreeze-from", default=UNFREEZE_FROM)
    parser.add_argument("--tag", default="")
    parser.add_argument("--lr", type=float, default=FINETUNE_BACKBONE_LR)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    if args.finetune:
        for arm in args.arm or ARMS:
            for seed in args.seed or SEEDS:
                train_finetune(arm, seed,
                               margin=args.margin if args.margin is not None
                               else FINETUNE_MARGIN,
                               learning_rate=args.lr,
                               max_epochs=args.max_epochs if args.max_epochs != MAX_EPOCHS
                               else FINETUNE_MAX_EPOCHS,
                               unfreeze_from=args.unfreeze_from, tag=args.tag,
                               augment=not args.no_augment)
        return

    for arm in args.arm or ARMS:
        for head_kind in args.head or HEADS:
            margin = subcenters = None
            tag = ""
            if args.selected:
                margin, subcenters = selected_settings(arm, head_kind)
                tag = "_selected"
                print(f"    {arm}/{head_kind}: margin {margin}, k {subcenters}")
            if args.sampler:
                tag += f"_m{args.sampler}"
            for seed in args.seed or SEEDS:
                train_one(arm, head_kind, seed, max_epochs=args.max_epochs,
                          margin=margin, subcenters=subcenters, tag=tag,
                          sampler=args.sampler)


if __name__ == "__main__":
    main()
