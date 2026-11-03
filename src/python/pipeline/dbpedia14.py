"""Download DBpedia-14 and cache sentence embeddings for every example.

Output (inside <cache-dir>/dbpedia14/):
    embeddings.f32   N x 384 float32, little-endian, L2-normalized, no header
    labels.i32       N int32 class ids in [0, 13]
    meta.json        provenance: dataset id, model, shapes, class counts

The train and test splits are concatenated: the project resamples the data
anyway (see subsample.py), so the original split carries no meaning here.

The script is idempotent: if all three output files exist it does nothing
unless --force is given. Embeddings are written to a temporary directory and
moved into place only after the whole run succeeds.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("fetch_dbpedia14")

DATASET_ID = "fancyzhx/dbpedia_14"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
NUM_CLASSES = 14
OUTPUT_FILES = ("embeddings.f32", "labels.i32", "meta.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"),
                        help="Root cache directory (default: data/cache).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encoding batch size (default: 256).")
    parser.add_argument("--chunk-size", type=int, default=20_000,
                        help="Number of texts encoded per write to disk (default: 20000).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N examples (for quick tests).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the outputs already exist.")
    return parser.parse_args(argv)


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_texts_and_labels(limit: int | None) -> tuple[list[str], np.ndarray]:
    """Return (texts, labels) with train and test concatenated."""
    from datasets import concatenate_datasets, load_dataset

    LOGGER.info("Loading dataset %s", DATASET_ID)
    splits = load_dataset(DATASET_ID)
    full = concatenate_datasets([splits["train"], splits["test"]])
    if limit is not None:
        full = full.select(range(min(limit, len(full))))

    # The title is the article name and the content is its abstract; joining
    # them gives the encoder the entity name as well as its description.
    texts = [f"{title}. {content}" for title,
             content in zip(full["title"], full["content"])]
    labels = np.asarray(full["label"], dtype=np.int32)
    return texts, labels


def encode_to_file(texts: list[str], path: Path, batch_size: int, chunk_size: int) -> None:
    from sentence_transformers import SentenceTransformer

    device = pick_device()
    LOGGER.info("Loading model %s on device '%s'", MODEL_NAME, device)
    model = SentenceTransformer(MODEL_NAME, device=device)

    dim = model.get_sentence_embedding_dimension()
    if dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected embedding dimension: expected {EMBEDDING_DIM}, got {dim}")

    mmap = np.memmap(path, dtype="<f4", mode="w+",
                     shape=(len(texts), EMBEDDING_DIM))
    for start in range(0, len(texts), chunk_size):
        stop = min(start + chunk_size, len(texts))
        mmap[start:stop] = model.encode(
            texts[start:stop],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        LOGGER.info("Encoded %d / %d", stop, len(texts))
    mmap.flush()
    del mmap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    out_dir = args.cache_dir / "dbpedia14"
    if not args.force and all((out_dir / name).exists() for name in OUTPUT_FILES):
        LOGGER.info(
            "Outputs already exist in %s; use --force to recompute.", out_dir)
        return 0

    texts, labels = load_texts_and_labels(args.limit)
    LOGGER.info("Loaded %d examples", len(texts))

    tmp_dir = args.cache_dir / "dbpedia14.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    encode_to_file(texts, tmp_dir / "embeddings.f32",
                   args.batch_size, args.chunk_size)
    labels.astype("<i4").tofile(tmp_dir / "labels.i32")

    counts = np.bincount(labels, minlength=NUM_CLASSES)
    meta = {
        "dataset": DATASET_ID,
        "splits": ["train", "test"],
        "model": MODEL_NAME,
        "num_examples": int(len(texts)),
        "dim": EMBEDDING_DIM,
        "normalized": True,
        "num_classes": NUM_CLASSES,
        "class_counts": [int(c) for c in counts],
        "limit": args.limit,
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    LOGGER.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
