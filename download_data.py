"""Download the datasets and pretrained weights used by the course notebooks.

Keras 3 on the JAX backend. Run from the repository root:

    python download_data.py                 # everything the Day 1-4 sessions and homework need
    python download_data.py --list          # what is available, how big, and what is already here
    python download_data.py cub esc50       # just these

Downloads are resumable-by-skipping: an archive that is already present is not
fetched again, and an archive that is already extracted is not extracted again.
Delete the target directory under data/ to force a refresh.
"""

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

os.environ.setdefault("KERAS_BACKEND", "jax")

DATA = "data"

# Datasets that arrive as an archive over HTTP.
#   url, archive name, the directory it extracts to, approximate size, why we need it
ARCHIVES = {
    "esc50": dict(
        url="https://github.com/karoldvl/ESC-50/archive/master.zip",
        archive="ESC-50-master.zip",
        target="ESC-50-master",
        size="~760 MB extracted",
        why="sessions/audio.ipynb (Day 3)",
    ),
    "cub": dict(
        # The tarball carries a stray attributes.txt at its top level, next to the
        # CUB_200_2011 directory, which would land directly in data/ and is not
        # gitignored. Nothing in the course uses CUB attribute data, so keep only
        # the dataset directory.
        url="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
        archive="CUB_200_2011.tgz",
        target="CUB_200_2011",
        keep=("CUB_200_2011",),
        size="1.1 GB download",
        why="sessions/transfer.ipynb and sessions/metric_learning.ipynb (Day 3)",
    ),
    "sign-language": dict(
        url="https://github.com/yoavram/Sign-Language/raw/master/Dataset.zip",
        archive="sign-lang.zip",
        target="Dataset",
        why="exercises/sign-lang.ipynb (Day 2, Day 3 homework)",
    ),
    "speech-emotion": dict(
        url="https://github.com/yoavram/SpeechEmotion/archive/master.zip",
        archive="SpeechEmotion.zip",
        target="SpeechEmotion-master",
        size="~4 MB",
        why="exercises/audio.ipynb (Day 3 homework)",
    ),
    "maf-benchmarks": dict(
        # Papamakarios's preprocessed datasets for the MAF paper (Zenodo 1161203,
        # CC-BY-4.0). One 857 MB tarball holding all five density-estimation
        # benchmarks plus mnist and cifar10, which we do not need. Its top level is
        # itself called "data", so we strip that component and keep only the two
        # datasets the course uses.
        url="https://zenodo.org/api/records/1161203/files/data.tar.gz/content",
        archive="maf_data.tar.gz",
        target="maf",
        into="maf",
        strip=1,
        keep=("power", "miniboone"),
        size="857 MB download, ~200 MB kept",
        why="sessions/flow.ipynb (Day 4); miniboone is for its exercise",
    ),
}

# Everything else: Keras datasets, pretrained weights, and a generated CSV.
# Each entry is a name -> (callable, description).


def _mnist():
    import keras

    keras.datasets.mnist.load_data()


def _fashion_mnist():
    import keras

    keras.datasets.fashion_mnist.load_data()


def _cifar10():
    import keras

    keras.datasets.cifar10.load_data()


def _resnet50():
    import keras

    keras.applications.ResNet50(weights="imagenet")


def _efficientnetv2s():
    import keras

    # transfer.ipynb uses the backbone without the classifier head, which is a
    # separate weights file from the full model.
    keras.applications.EfficientNetV2S(
        weights="imagenet", include_top=False, pooling="avg"
    )


def _clip():
    try:
        import keras_hub  # noqa: F401
    except ImportError:
        print("    skipped: keras-hub is not installed (see README.md); "
              "only sessions/metric_learning.ipynb needs it")
        return

    # sessions/metric_learning.ipynb. keras-hub is installed without its dependency
    # tree (see README.md), because it declares tensorflow-text; the CLIP backbone
    # itself runs fine on JAX without TensorFlow, but the tokenizer needs a patch,
    # which the notebook applies and explains.
    from keras_hub.models import CLIPBackbone, CLIPTokenizer

    original = CLIPTokenizer._set_vocabulary_and_merges_tokenizers

    def patched(self, vocabulary, merges):
        self.vocabulary = vocabulary.copy()
        self.merges = list(merges)
        return original(self, vocabulary, merges)

    CLIPTokenizer._set_vocabulary_and_merges_tokenizers = patched

    CLIPBackbone.from_preset("clip_vit_base_patch16")
    CLIPTokenizer.from_preset("clip_vit_base_patch16")


# The cached arrays for sessions/reid.ipynb. This holds only what cannot be rebuilt
# from the repository in reasonable time: the MiewID embeddings and ALIKED+LightGlue
# scores (PyTorch, which this course does not install), the fine-tuned backbone's
# embeddings (about 5 GPU-hours), the margin sweep's results (hours of CPU), and the
# two frozen-feature arrays. The 30 probe embeddings the notebook also loads are NOT
# in it -- scripts/reid_arcface.py rebuilds them bit-for-bit in about ten minutes on a
# CPU, which is a better trade than another 460 MB of download.
REID_ARRAYS_URL = (
    "https://github.com/yoavram/DataSciPy/releases/download/"
    "reid-arrays-v1/reid_arrays.tar.gz"
)


def _reid_arrays():
    """Cached arrays for sessions/reid.ipynb (~188 MB), then the probe rebuild."""
    import subprocess

    marker = os.path.join(DATA, "turtle_miewid_embeddings.npy")
    if not os.path.exists(marker):
        if REID_ARRAYS_URL is None:
            print("    no download URL is configured yet.")
            print("    Rebuild what you can locally instead:")
            print("        python download_data.py turtles")
            print("        python scripts/reid_data.py")
            print("    The MiewID embeddings and LightGlue scores need PyTorch; see")
            print("    scripts/reid_precompute_torch.py.")
            return
        archive = os.path.join(DATA, "reid_arrays.tar.gz")
        if not os.path.exists(archive):
            download_file(REID_ARRAYS_URL, archive)
        extract(archive, DATA)
    else:
        print(f"    already present: {marker}")

    # The probe embeddings: cheap to recompute, expensive to ship.
    if os.path.exists(os.path.join(DATA, "turtle_arcface_time_plain_selected_s44_embeddings.npy")):
        print("    probe embeddings already present")
        return
    if not os.path.exists(os.path.join(DATA, "turtle_effnetv2s_embeddings.npy")):
        print("    skipped the probe rebuild: run scripts/reid_data.py first")
        return
    print("    rebuilding the probe embeddings (about 10 minutes on a CPU)")
    script = os.path.join("scripts", "reid_arcface.py")
    for extra in ([], ["--head", "plain", "--sampler", "2"], ["--head", "plain", "--sampler", "4"]):
        subprocess.run([sys.executable, script, "--selected"] + extra, check=True)


def _turtles():
    """SeaTurtleIDHeads, via wildlife-datasets.

    The download goes through Kaggle, which needs an API token: create one at
    https://www.kaggle.com/settings ("Create New Token"), save the file as
    ~/.kaggle/kaggle.json, and accept the dataset's terms once at
    https://www.kaggle.com/datasets/wildlifedatasets/seaturtleidheads
    """
    target = os.path.join(DATA, "SeaTurtleIDHeads")
    if os.path.isdir(target):
        print(f"    already present: {target}")
        return
    try:
        from wildlife_datasets.datasets import SeaTurtleIDHeads
    except ImportError:
        print("    skipped: wildlife-datasets is not installed (see requirements.txt)")
        return
    try:
        SeaTurtleIDHeads.get_data(target)
    except Exception as error:
        print(f"    failed: {error}")
        print("    a Kaggle API token is needed; see the docstring of _turtles() "
              "in this file")


def _penguins():
    path = os.path.join(DATA, "penguins.csv")
    if os.path.exists(path):
        print(f"    already present: {path}")
        return
    from palmerpenguins import load_penguins

    load_penguins().to_csv(path, index=False)
    print(f"    wrote {path}")


KERAS_ITEMS = {
    "mnist": (_mnist, "MNIST digits (Keras cache) - Days 2, 3"),
    "fashion-mnist": (_fashion_mnist, "Fashion-MNIST (Keras cache) - Day 3 homework"),
    "cifar10": (_cifar10, "CIFAR-10 (Keras cache) - sessions/augmentation.ipynb (Day 3)"),
    "resnet50": (_resnet50, "ResNet50 ImageNet weights - sessions/pretrained.ipynb"),
    "efficientnetv2s": (
        _efficientnetv2s,
        "EfficientNetV2S ImageNet weights, no top - sessions/transfer.ipynb",
    ),
    "clip": (
        _clip,
        "CLIP ViT-B/16 weights, ~570 MB - sessions/metric_learning.ipynb "
        "(needs keras-hub, see README.md)",
    ),
    "turtles": (
        _turtles,
        "SeaTurtleIDHeads, ~425 MB, needs a Kaggle token - sessions/reid.ipynb",
    ),
    "reid-arrays": (
        _reid_arrays,
        "cached arrays, ~188 MB + a 10 min rebuild - sessions/reid.ipynb",
    ),
    "penguins": (_penguins, "data/penguins.csv - sessions/gamma_regression.ipynb"),
}


def _progress(done, total):
    if total <= 0:
        sys.stdout.write(f"\r    {done / 1e6:.0f} MB")
    else:
        pct = 100 * done / total
        sys.stdout.write(f"\r    {pct:5.1f}%  {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    sys.stdout.flush()


def download_file(url, path):
    """Stream url to path, showing progress. Writes to a .part file first."""
    part = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "DataSciPy/download_data"})
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            _progress(done, total)
    sys.stdout.write("\n")
    os.replace(part, path)


def extract(path, into, strip=0, keep=None):
    """Extract path into `into`.

    strip: drop this many leading path components (like tar --strip-components).
    keep:  if given, only extract members whose stripped path starts with one of
           these prefixes.
    """
    os.makedirs(into, exist_ok=True)
    if path.endswith(".zip"):
        if strip or keep:
            raise ValueError("strip/keep are only implemented for tar archives")
        with zipfile.ZipFile(path) as z:
            z.extractall(into)
    elif path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as t:
            if not strip and not keep:
                # filter='data' refuses absolute paths and paths escaping the target
                t.extractall(into, filter="data")
                return
            members = []
            for m in t.getmembers():
                parts = m.name.split("/")[strip:]
                if not parts or not parts[0]:
                    continue
                if keep and parts[0] not in keep:
                    continue
                m.name = "/".join(parts)
                members.append(m)
            if not members:
                raise tarfile.TarError(
                    f"no members matched keep={keep} after stripping {strip} component(s)"
                )
            t.extractall(into, members=members, filter="data")
    else:
        raise ValueError(f"do not know how to extract {path}")


def fetch_archive(name, spec, keep_archive=False):
    target = os.path.join(DATA, spec["target"])
    archive = os.path.join(DATA, spec["archive"])
    if os.path.isdir(target) and os.listdir(target):
        print(f"    already extracted: {target}")
        return
    if not os.path.exists(archive):
        print(f"    downloading {spec['url']}")
        try:
            download_file(spec["url"], archive)
        except Exception as e:  # noqa: BLE001 - report and continue with the rest
            print(f"    FAILED: {e}")
            return
    else:
        print(f"    using cached archive: {archive}")
    into = os.path.join(DATA, spec["into"]) if spec.get("into") else DATA
    print(f"    extracting to {target}")
    try:
        extract(archive, into, strip=spec.get("strip", 0), keep=spec.get("keep"))
    except (zipfile.BadZipFile, tarfile.TarError, ValueError) as e:
        print(f"    FAILED to extract ({e}); delete {archive} and try again")
        return
    if not keep_archive:
        os.remove(archive)


def do_list():
    print(f"{'name':16s} {'present':8s} size / notes")
    print("-" * 78)
    for name, spec in ARCHIVES.items():
        target = os.path.join(DATA, spec["target"])
        present = os.path.isdir(target) and bool(os.listdir(target))
        tag = "yes" if present else "no"
        print(f"{name:16s} {tag:8s} {spec.get('size', 'size not reported by host')}")
        print(f"{'':25s} {spec['why']}")
    for name, (_, desc) in KERAS_ITEMS.items():
        print(f"{name:16s} {'-':8s} {desc}")
    print("\nKeras caches datasets and weights under ~/.keras/, not under data/,")
    print("so 'present' is not tracked for those.")


def main():
    p = argparse.ArgumentParser(
        description="Download course datasets and pretrained weights.",
        epilog="With no arguments, fetches everything the Day 1-4 notebooks need.",
    )
    p.add_argument("names", nargs="*", help="specific items to fetch (see --list)")
    p.add_argument(
        "--list", action="store_true", help="show what is available and exit"
    )
    p.add_argument(
        "--keep-archives",
        action="store_true",
        help="do not delete archives after extracting",
    )
    args = p.parse_args()

    if args.list:
        do_list()
        return

    os.makedirs(DATA, exist_ok=True)

    known = set(ARCHIVES) | set(KERAS_ITEMS)
    if args.names:
        unknown = set(args.names) - known
        if unknown:
            p.error(f"unknown item(s): {', '.join(sorted(unknown))}. Try --list.")
        selected = list(args.names)
    else:
        selected = list(ARCHIVES) + list(KERAS_ITEMS)

    import keras

    print(f"keras {keras.__version__}, backend {keras.backend.backend()}\n")

    for name in selected:
        print(f"* {name}")
        if name in ARCHIVES:
            fetch_archive(name, ARCHIVES[name], keep_archive=args.keep_archives)
        else:
            fn, _ = KERAS_ITEMS[name]
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"    FAILED: {e}")

    free = shutil.disk_usage(DATA).free / 1e9
    print(f"\ndone. {free:.1f} GB free on this volume.")


if __name__ == "__main__":
    main()
