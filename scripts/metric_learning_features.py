"""Precompute the cached arrays that sessions/metric_learning.ipynb loads.

Keras 3 on the JAX backend. Run once from the repository root:

    KERAS_BACKEND=jax python scripts/metric_learning_features.py

Two backbones are run over CUB-200-2011, and nothing is trained:

  data/cub_effnetv2s_embeddings.npy   (11788, 1280) float32 - frozen ImageNet
                                      features, shared with sessions/transfer.ipynb
  data/cub_clip_vitb16_images.npy     (11788,  512) float32 - CLIP image embeddings,
                                      L2-normalized

The text side is deliberately *not* cached: the notebook encodes its prompts live,
because comparing prompt templates is one of the things it teaches, and a hundred
sentences through the text encoder costs seconds even on a CPU.

On an RTX A4000 this takes about two minutes. On a CPU it takes closer to half an
hour, which is why the arrays are cached rather than computed in the notebook.

`keras-hub` must be installed without its dependencies, because it declares
`tensorflow-text` and this course does not install TensorFlow:

    python -m pip install --no-deps keras-hub
    python -m pip install regex tokenizers kagglehub
"""

import os
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd

import keras

DATA = "data"
DATASET_DIR = os.path.join(DATA, "CUB_200_2011")
IMAGES_CACHE = os.path.join(DATA, "cub_images_224.npy")
IMG_SIZE = 224
CLIP_PRESET = "clip_vit_base_patch16"

def load_metadata():
    images_df = pd.read_csv(
        os.path.join(DATASET_DIR, "images.txt"), sep=" ", names=["image_id", "filename"]
    )
    labels_df = pd.read_csv(
        os.path.join(DATASET_DIR, "image_class_labels.txt"),
        sep=" ",
        names=["image_id", "class_id"],
    )
    classes_df = pd.read_csv(
        os.path.join(DATASET_DIR, "classes.txt"), sep=" ", names=["class_id", "class_name"]
    )
    metadata = images_df.merge(labels_df, on="image_id")
    species = classes_df.class_name.str.split(".").str[1].str.replace("_", " ")
    return metadata, species.to_numpy()


def load_images():
    if not os.path.exists(IMAGES_CACHE):
        raise SystemExit(
            f"{IMAGES_CACHE} is missing. Run the decoding cell in "
            "sessions/transfer.ipynb, or sessions/metric_learning.ipynb, first."
        )
    return np.load(IMAGES_CACHE, mmap_mode="r")


def effnet_embeddings(images):
    """Frozen EfficientNetV2S features, exactly as sessions/transfer.ipynb uses them."""
    out = os.path.join(DATA, "cub_effnetv2s_embeddings.npy")
    if os.path.exists(out):
        print(f"    already present: {out}")
        return
    backbone = keras.applications.EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False
    t0 = time.time()
    # EfficientNetV2 rescales internally, so the uint8 0-255 images go in as they are.
    Z = backbone.predict(np.asarray(images), batch_size=64, verbose=0)
    np.save(out, Z.astype("float32"))
    print(f"    wrote {out} {Z.shape} in {time.time() - t0:.0f}s")


def patch_clip_tokenizer():
    """keras-hub 0.32 cannot build a CLIP tokenizer without TensorFlow.

    `BytePairTokenizer.set_vocabulary_and_merges` calls the TensorFlow
    implementation first and swallows the resulting `ImportError` -- but that is
    the call which assigns `self.vocabulary` and `self.merges`. The pure-Python
    implementation that runs next reads those attributes rather than its own
    arguments, and dies on `None`. Assigning them first is enough.
    """
    from keras_hub.models import CLIPTokenizer

    original = CLIPTokenizer._set_vocabulary_and_merges_tokenizers

    def patched(self, vocabulary, merges):
        self.vocabulary = vocabulary.copy()
        self.merges = list(merges)
        return original(self, vocabulary, merges)

    CLIPTokenizer._set_vocabulary_and_merges_tokenizers = patched
    return CLIPTokenizer


def clip_embeddings(images):
    image_out = os.path.join(DATA, "cub_clip_vitb16_images.npy")
    if os.path.exists(image_out):
        print(f"    already present: {image_out}")
        return

    from keras_hub.layers import CLIPImageConverter
    from keras_hub.models import CLIPBackbone

    patch_clip_tokenizer()
    backbone = CLIPBackbone.from_preset(CLIP_PRESET)
    converter = CLIPImageConverter.from_preset(CLIP_PRESET)

    t0 = time.time()
    chunks = []
    for start in range(0, len(images), 256):
        batch = np.asarray(images[start : start + 256]).astype("float32")
        chunks.append(np.asarray(backbone.get_vision_embeddings(converter(batch))))
    Z = normalize(np.concatenate(chunks))
    np.save(image_out, Z.astype("float32"))
    print(f"    wrote {image_out} {Z.shape} in {time.time() - t0:.0f}s")


def normalize(Z):
    return Z / np.linalg.norm(Z, axis=-1, keepdims=True)


def main():
    metadata, species = load_metadata()
    images = load_images()
    print(f"{len(metadata):,} images, {len(species)} species")

    print("EfficientNetV2S features")
    effnet_embeddings(images)
    print("CLIP ViT-B/16 image embeddings")
    clip_embeddings(images)


if __name__ == "__main__":
    main()
