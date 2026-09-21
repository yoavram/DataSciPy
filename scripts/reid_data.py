"""Build the catalogue, the two splits, and the cached arrays sessions/reid.ipynb loads.

Keras 3 on the JAX backend. Run once from the repository root:

    KERAS_BACKEND=jax python scripts/reid_data.py

The dataset is SeaTurtleIDHeads (Adam et al., WACV 2024): 7,582 head crops of 400
loggerhead sea turtles photographed between 2010 and 2021, fetched through the
`wildlife-datasets` package, which needs a Kaggle API token -- see
`python download_data.py --list`.

Outputs, all under data/ and all gitignored:

  turtle_catalogue.csv            7,582 rows: identity, date, orientation, the role
                                  of the individual, and the arm-by-arm assignment
  turtle_images_224.npy           (7582, 224, 224, 3) uint8
  turtle_effnetv2s_embeddings.npy (7582, 1280) float32 - frozen ImageNet features
  turtle_clip_vitb16_images.npy   (7582,  512) float32 - CLIP image embeddings,
                                  L2-normalized, for the zero-shot opening

Why two splits
--------------
The notebook's central lesson is that a random split of a re-identification dataset
leaks: photographs from one encounter -- same animal, same day, same water, same
light -- land on both sides of the split, so the model is rewarded for recognising
the encounter rather than the individual. The two arms built here differ in exactly
one thing, the rule that assigns an individual's photographs to the database or to
the query set:

  time    the later half of an individual's observation *dates* becomes the query
          set (Cermak et al., arXiv:2211.10307, as implemented by
          wildlife_datasets.splits.TimeProportionSplit)
  random  the same *number* of query images per individual, drawn at random

Matching the per-individual counts is deliberate. It leaves gallery size, query
size and the label distribution identical across the arms, so the difference in the
reported numbers cannot be attributed to any of them. Measured on this catalogue,
95% of random-arm query images have a same-individual same-day photograph sitting
in the database; under the time arm, none do.

Individuals with a single observation date cannot contribute a query image to the
time arm, and so contribute none to either arm. That is a real cost of an honest
protocol -- 172 of the 400 individuals here -- and the notebook says so.

Roles of individuals
--------------------
  train    (300) the model is fitted on their database images
  tune      (50) never trained on; every threshold and epoch budget is chosen here
  unknown   (50) never trained on, never in the database: the open-set queries that
                 the model is supposed to *reject*, and the identities the
                 clustering metric has to recover without supervision
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import time

import numpy as np
import pandas as pd
from PIL import Image

import keras

DATA = "data"
DATASET_DIR = os.path.join(DATA, "SeaTurtleIDHeads")
CATALOGUE = os.path.join(DATA, "turtle_catalogue.csv")
IMAGES_CACHE = os.path.join(DATA, "turtle_images_224.npy")
IMG_SIZE = 224
SEED = 42
N_TUNE = 50
N_UNKNOWN = 50
CLIP_PRESET = "clip_vit_base_patch16"


def load_catalogue():
    """The wildlife-datasets catalogue, sorted so that row order is reproducible."""
    from wildlife_datasets.datasets import SeaTurtleIDHeads

    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(
            f"{DATASET_DIR} is missing. Run `python download_data.py turtles` first."
        )
    df = SeaTurtleIDHeads(DATASET_DIR).df
    df = df.sort_values(["identity", "date", "path"]).reset_index(drop=True)
    df["image_id"] = np.arange(len(df))
    return df


def assign_roles(df, rng):
    """Split the individuals three ways: trained on, tuned on, held out as unknown."""
    identities = np.sort(df.identity.unique())
    shuffled = rng.permutation(identities)
    unknown = set(shuffled[:N_UNKNOWN])
    tune = set(shuffled[N_UNKNOWN : N_UNKNOWN + N_TUNE])
    role = np.where(
        df.identity.isin(unknown), "unknown", np.where(df.identity.isin(tune), "tune", "train")
    )
    return pd.Series(role, index=df.index, name="role")


def query_counts(df):
    """How many query images each individual contributes, from the time-aware rule.

    `TimeProportionSplit` puts the later half of an individual's *distinct dates*
    into the test set and ignores individuals observed on a single date. Both arms
    inherit these counts, so the arms differ only in *which* images are queries.
    """
    from wildlife_datasets import splits

    idx_train, idx_test = splits.TimeProportionSplit(0.5).split(df)[0]
    query = pd.Series(False, index=df.index)
    query.loc[df.index.isin(idx_test)] = True
    return query


def assign_arms(df, time_query, rng):
    """database / query / unused, under each arm.

    Unknown individuals are query-only in both arms: they must never appear in the
    database the model searches, or there would be nothing to reject.
    """
    arms = {}
    arms["time"] = np.where(time_query, "query", "database")
    random_query = np.zeros(len(df), dtype=bool)
    for _, rows in df.groupby("identity", sort=True).groups.items():
        rows = np.asarray(rows)
        n = int(time_query.loc[rows].sum())
        if n:
            random_query[rng.choice(rows, size=n, replace=False)] = True
    arms["random"] = np.where(random_query, "query", "database")
    for arm in arms:
        arms[arm] = np.where(df.role.to_numpy() == "unknown", "query", arms[arm])
    return pd.DataFrame(arms, index=df.index).add_prefix("arm_")


def leakage(df, arm):
    """Fraction of query images with a same-individual same-day photo in the database."""
    col = f"arm_{arm}"
    database = set(zip(df.loc[df[col] == "database", "identity"], df.loc[df[col] == "database", "date"]))
    # Unknown individuals are absent from the database by construction, so counting
    # them here would only dilute the number the comparison is about.
    query = df[(df[col] == "query") & (df.role != "unknown")]
    if not len(query):
        return float("nan")
    return float(np.mean([(i, d) in database for i, d in zip(query.identity, query.date)]))


def build_catalogue():
    if os.path.exists(CATALOGUE):
        print(f"    already present: {CATALOGUE}")
        return pd.read_csv(CATALOGUE)

    rng = np.random.default_rng(SEED)
    df = load_catalogue()
    df["role"] = assign_roles(df, rng)
    time_query = query_counts(df)
    df = pd.concat([df, assign_arms(df, time_query, rng)], axis=1)
    df.to_csv(CATALOGUE, index=False)

    print(f"    wrote {CATALOGUE} {df.shape}")
    print(f"    {df.identity.nunique()} individuals, {len(df):,} head crops, "
          f"{df.date.min()} to {df.date.max()}")
    for role, rows in df.groupby("role"):
        print(f"      {role:8s} {rows.identity.nunique():3d} individuals, {len(rows):5,d} images")
    for arm in ("random", "time"):
        counts = df[f"arm_{arm}"].value_counts()
        print(f"      {arm:6s} database {counts.get('database', 0):5,d}  "
              f"query {counts.get('query', 0):5,d}  "
              f"same-day leak {leakage(df, arm):.1%}")
    return df


def build_images(df):
    if os.path.exists(IMAGES_CACHE):
        print(f"    already present: {IMAGES_CACHE}")
        return np.load(IMAGES_CACHE, mmap_mode="r")

    t0 = time.time()
    images = np.zeros((len(df), IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    for i, path in enumerate(df.path):
        with Image.open(os.path.join(DATASET_DIR, path)) as im:
            images[i] = np.asarray(im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    np.save(IMAGES_CACHE, images)
    print(f"    wrote {IMAGES_CACHE} {images.shape} in {time.time() - t0:.0f}s")
    return images


def effnet_embeddings(images):
    """Frozen EfficientNetV2S features, the same backbone as the companion notebook.

    EfficientNetV2 carries its own rescaling layer (`include_preprocessing=True`),
    so the uint8 0-255 images go in exactly as they are -- `preprocess_input` for
    this family is a pass-through.
    """
    out = os.path.join(DATA, "turtle_effnetv2s_embeddings.npy")
    if os.path.exists(out):
        print(f"    already present: {out}")
        return
    backbone = keras.applications.EfficientNetV2S(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False
    t0 = time.time()
    Z = backbone.predict(np.asarray(images), batch_size=64, verbose=0)
    np.save(out, Z.astype("float32"))
    print(f"    wrote {out} {Z.shape} in {time.time() - t0:.0f}s")


def clip_embeddings(images):
    """CLIP image embeddings, for the zero-shot opening.

    keras-hub is installed without its dependency tree, and its CLIP tokenizer then
    needs the patch that sessions/metric_learning.ipynb explains; only the image
    side is cached here, so the patch is imported from the companion script.
    """
    out = os.path.join(DATA, "turtle_clip_vitb16_images.npy")
    if os.path.exists(out):
        print(f"    already present: {out}")
        return
    try:
        from keras_hub.layers import CLIPImageConverter
        from keras_hub.models import CLIPBackbone
    except ImportError:
        print("    skipped: keras-hub is not installed (see README.md)")
        return

    backbone = CLIPBackbone.from_preset(CLIP_PRESET)
    converter = CLIPImageConverter.from_preset(CLIP_PRESET)
    t0 = time.time()
    chunks = []
    for start in range(0, len(images), 256):
        batch = np.asarray(images[start : start + 256]).astype("float32")
        chunks.append(np.asarray(backbone.get_vision_embeddings(converter(batch))))
    Z = np.concatenate(chunks)
    Z /= np.linalg.norm(Z, axis=-1, keepdims=True)
    np.save(out, Z.astype("float32"))
    print(f"    wrote {out} {Z.shape} in {time.time() - t0:.0f}s")


def main():
    print("catalogue and splits")
    df = build_catalogue()
    print("decoding head crops")
    images = build_images(df)
    print("EfficientNetV2S features")
    effnet_embeddings(images)
    print("CLIP ViT-B/16 image embeddings")
    clip_embeddings(images)


if __name__ == "__main__":
    main()
