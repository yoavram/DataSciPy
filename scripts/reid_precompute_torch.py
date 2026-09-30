"""Precompute the MiewID and ALIKED+LightGlue arrays that sessions/reid.ipynb loads.

*** DO NOT RUN THIS IN THE COURSE ENVIRONMENT. ***

This is the only PyTorch in the repository, and it is here so that the notebook can
compare against MiewID and teach local-descriptor matching without the course ever
installing torch, timm or transformers. It is run once, by hand, in a separate
environment, and what reaches the students is three `.npy`/`.npz` files.

    cd ~/Work/Research/AnimalCLEF26
    .pixi/envs/default/bin/python ~/Work/Teaching/DataSciPy/scripts/reid_precompute_torch.py \
        --repo ~/Work/Teaching/DataSciPy

Outputs, under the course repository's data/ and gitignored:

  turtle_miewid_embeddings.npy   (7582, 2152) float32, L2-normalized
  turtle_lightglue_topk.npz      the local-matching scores for the re-ranking exercise
  turtle_match_examples.npz      a few keypoint correspondences, for the figures

Why a shortlist and not a score matrix
--------------------------------------
Local matching is O(N^2) and that is the entire point of the section: 3,744 queries
against a 3,838-image database is 14 million LightGlue calls, which is days. The
deployed answer -- and the thing the notebook's exercise implements -- is a hybrid:
retrieve a shortlist by embedding cosine similarity, then re-rank the shortlist by
local matching. So what is precomputed is exactly what a hybrid would compute, the
local score for each query against its top-k embedding neighbours, and the cost is
3,744 x 10 instead of 3,744 x 3,838.

Shipping a full matrix would also quietly hide the cost that motivates the hybrid.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

TOPK = 10
MIEWID_SIZE = 440
ALIKED_MAX_SIDE = 512


def course_paths(repo):
    data = os.path.join(repo, "data")
    return data, os.path.join(data, "SeaTurtleIDHeads")


def load_catalogue(data):
    path = os.path.join(data, "turtle_catalogue.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing. Run scripts/reid_data.py first.")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- MiewID


def miewid_embeddings(catalogue, dataset_dir, out, device):
    """MiewID-msv3 embeddings for every head crop.

    MiewID-msv3 is EfficientNetV2-RW-M with a sub-center ArcFace head, fine-tuned on
    64 wildlife species from Wildbook, and it emits a 2,152-d embedding. Because our
    own backbone is the same family, the comparison the notebook draws is roughly
    architecture-controlled -- it isolates training data and recipe rather than
    architecture. Two caveats the notebook repeats: "RW" is the timm variant, not
    `keras.applications.EfficientNetV2M`, so the control is partial; and MiewID's
    training corpus includes sea turtles, so this is not a zero-shot number.
    """
    if os.path.exists(out):
        print(f"    already present: {out}")
        return
    sys.path.insert(0, os.path.join(os.path.expanduser("~"), "Work/Research/AnimalCLEF26/src"))
    from reid.core.models import load_miewid

    model, preprocess = load_miewid(device=device, cache_only=False)
    model.eval()

    t0 = time.time()
    chunks, batch = [], []
    for i, path in enumerate(catalogue.path):
        with Image.open(os.path.join(dataset_dir, path)) as im:
            batch.append(preprocess(im.convert("RGB")))
        if len(batch) == 32 or i == len(catalogue) - 1:
            with torch.no_grad():
                out_batch = model(torch.stack(batch).to(device))
            chunks.append(out_batch.detach().cpu().numpy())
            batch = []
    Z = np.concatenate(chunks).astype("float32")
    Z /= np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-12
    np.save(out, Z)
    print(f"    wrote {out} {Z.shape} in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ ALIKED/LightGlue


def load_image(path, max_side=ALIKED_MAX_SIDE):
    """ALIKED wants a reasonable resolution; these crops are small, so only shrink."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
        return np.asarray(im)


def shortlists(catalogue, embeddings, arm):
    """Each query's top-k database neighbours, by embedding cosine similarity."""
    column = f"arm_{arm}"
    database = catalogue.index[catalogue[column] == "database"].to_numpy()
    query = catalogue.index[catalogue[column] == "query"].to_numpy()

    normalize = lambda Z: Z / (np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-12)
    similarity = normalize(embeddings[query]) @ normalize(embeddings[database]).T
    order = np.argsort(-similarity, axis=1)[:, :TOPK]
    return query, database[order], np.take_along_axis(similarity, order, axis=1)


def feature_cache_dir(data):
    return os.path.join(data, "turtle_aliked")


def extract_features(catalogue, dataset_dir, data, rows, device, shard=0, shards=1):
    """ALIKED keypoints and descriptors, cached one file per image.

    On a CPU this runs at about one image a second, so it is sharded: each process
    takes every `shards`-th image and skips whatever is already cached.
    """
    sys.path.insert(0, os.path.join(os.path.expanduser("~"), "Work/Research/AnimalCLEF26/src"))
    from reid.core.features import extract_aliked
    from reid.core.models import load_aliked

    cache = feature_cache_dir(data)
    os.makedirs(cache, exist_ok=True)
    todo = [r for i, r in enumerate(rows) if i % shards == shard
            and not os.path.exists(os.path.join(cache, f"{r}.npz"))]
    if not todo:
        print(f"    shard {shard}: nothing to extract")
        return

    extractor = load_aliked(device=device, cache_only=False)
    print(f"    shard {shard}: extracting {len(todo):,} images")
    t0 = time.time()
    for n, row in enumerate(todo, 1):
        image = load_image(os.path.join(dataset_dir, catalogue.path[row]))
        f = extract_aliked(image, extractor=extractor, device=device)[0]
        np.savez(os.path.join(cache, f"{row}.npz"),
                 keypoints=np.asarray(f.keypoints), descriptors=np.asarray(f.descriptors),
                 image_size=np.asarray(f.image_size))
        if n % 200 == 0:
            print(f"      shard {shard}: {n:,}/{len(todo):,}, {n / (time.time() - t0):.1f} img/s")


def load_features(data, row):
    from reid.core.contracts import LocalFeatures

    z = np.load(os.path.join(feature_cache_dir(data), f"{row}.npz"))
    return LocalFeatures(keypoints=z["keypoints"], descriptors=z["descriptors"],
                         image_size=z["image_size"])


def lightglue_scores(data, query_rows, shortlist, device, shard=0, shards=1):
    """LightGlue match counts for each (query, shortlisted database image) pair."""
    sys.path.insert(0, os.path.join(os.path.expanduser("~"), "Work/Research/AnimalCLEF26/src"))
    from reid.core.matching import lightglue_matching
    from reid.core.models import load_lightglue_matcher

    matcher = load_lightglue_matcher(features="aliked", device=device, cache_only=False)
    mine = np.arange(len(query_rows)) % shards == shard
    scores = np.full(shortlist.shape, np.nan, dtype="float32")

    print(f"    shard {shard}: matching {int(mine.sum()) * shortlist.shape[1]:,} pairs")
    t0 = time.time()
    for n, i in enumerate(np.flatnonzero(mine), 1):
        query = load_features(data, query_rows[i])
        for j, candidate in enumerate(shortlist[i]):
            scores[i, j] = lightglue_matching(
                [query], [load_features(data, candidate)], matcher=matcher, device=device)[0, 0]
        if n % 100 == 0:
            rate = n * shortlist.shape[1] / (time.time() - t0)
            print(f"      shard {shard}: {n:,}/{int(mine.sum()):,} queries, {rate:.0f} pairs/s")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--arm", default="time", choices=("time", "random"))
    parser.add_argument("--head", default="plain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-miewid", action="store_true")
    parser.add_argument("--skip-lightglue", action="store_true")
    parser.add_argument("--phase", choices=("extract", "match", "merge"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data, dataset_dir = course_paths(args.repo)
    catalogue = load_catalogue(data)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(catalogue):,} head crops, device {device}")

    if not args.skip_miewid:
        print("MiewID-msv3 embeddings")
        miewid_embeddings(catalogue, dataset_dir,
                          os.path.join(data, "turtle_miewid_embeddings.npy"), device)

    if args.skip_lightglue:
        return

    print("ALIKED + LightGlue shortlist scores")
    out = os.path.join(data, "turtle_lightglue_topk.npz")
    stem = f"turtle_arcface_{args.arm}_{args.head}_selected_s{args.seed}_embeddings.npy"
    embeddings = np.load(os.path.join(data, stem))
    query_rows, shortlist, similarity = shortlists(catalogue, embeddings, args.arm)

    if args.phase == "extract":
        needed = np.unique(np.concatenate([query_rows, shortlist.ravel()]))
        extract_features(catalogue, dataset_dir, data, needed, device, args.shard, args.shards)
        return

    if args.phase == "match":
        scores = lightglue_scores(data, query_rows, shortlist, device, args.shard, args.shards)
        part = os.path.join(data, f"turtle_lightglue_part{args.shard}.npy")
        np.save(part, scores)
        print(f"    wrote {part}")
        return

    if args.phase == "merge":
        parts = sorted(f for f in os.listdir(data) if f.startswith("turtle_lightglue_part"))
        scores = np.full(shortlist.shape, np.nan, dtype="float32")
        for name in parts:
            part = np.load(os.path.join(data, name))
            scores = np.where(np.isnan(scores), part, scores)
        missing = int(np.isnan(scores).sum())
        if missing:
            raise SystemExit(f"{missing} pairs are still missing from {len(parts)} shards")
        # Keypoint counts travel with the scores: the exercise's normalization needs
        # them, and shipping them here means the multi-gigabyte feature cache never
        # has to leave this machine.
        counts = {}
        for row in np.unique(np.concatenate([query_rows, shortlist.ravel()])):
            counts[int(row)] = int(np.load(
                os.path.join(feature_cache_dir(data), f"{row}.npz"))["keypoints"].shape[1])
        keypoints = np.vectorize(counts.get)
        np.savez(out, query_rows=query_rows, shortlist=shortlist,
                 embedding_similarity=similarity.astype("float32"), lightglue_score=scores,
                 keypoints_query=keypoints(query_rows).astype("int32"),
                 keypoints_shortlist=keypoints(shortlist).astype("int32"),
                 arm=args.arm, head=args.head, seed=args.seed)
        print(f"    wrote {out} from {len(parts)} shards")
        return


if __name__ == "__main__":
    main()
