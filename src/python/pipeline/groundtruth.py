"""Exact k-nearest-neighbor ground truth by brute force.

The distance convention of the whole project is squared Euclidean distance (L2^2)
in float32. Distances stored here use that same convention.

Implementation notes:
  * Candidate search uses faiss IndexFlatL2 (exact, exhaustive). faiss computes
    distances through matrix products, which can introduce tiny rounding errors
    and swap neighbors that are almost tied.
  * To make the ground truth robust to that, each query asks for k + margin
    candidates. Their distances are then recomputed directly in float64 from the
    original vectors, re-sorted, and the best k are kept. The stored distances
    are therefore not faiss' approximate-arithmetic values.

This module is a library; export.py calls it. It reads and writes no files.
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger("groundtruth")

DEFAULT_K = 10
DEFAULT_BLOCK_SIZE = 500
DEFAULT_CANDIDATE_MARGIN = 10


def exact_knn(
    train: np.ndarray,
    queries: np.ndarray,
    k: int = DEFAULT_K,
    block_size: int = DEFAULT_BLOCK_SIZE,
    candidate_margin: int = DEFAULT_CANDIDATE_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, distances) of the exact k nearest train rows for every query.

    ids:       int32 array (Q, k), row indices into `train`, nearest first.
    distances: float32 array (Q, k), squared L2 distances, ascending per row.
    """
    import faiss

    _validate(train, queries, k, block_size, candidate_margin)

    num_train, dim = train.shape
    num_queries = queries.shape[0]
    num_candidates = min(k + candidate_margin, num_train)

    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(train))

    ids = np.empty((num_queries, k), dtype=np.int32)
    distances = np.empty((num_queries, k), dtype=np.float32)

    for start in range(0, num_queries, block_size):
        stop = min(start + block_size, num_queries)
        block = np.ascontiguousarray(queries[start:stop])

        _, candidates = index.search(block, num_candidates)

        # Recompute the candidate distances directly, in float64.
        diff = train[candidates].astype(
            np.float64) - block[:, None, :].astype(np.float64)
        exact = np.einsum("qcd,qcd->qc", diff, diff)

        order = np.argsort(exact, axis=1, kind="stable")[:, :k]
        ids[start:stop] = np.take_along_axis(candidates, order, axis=1)
        distances[start:stop] = np.take_along_axis(exact, order, axis=1)

        LOGGER.info("Ground truth: %d / %d queries", stop, num_queries)

    return ids, distances


def _validate(
    train: np.ndarray,
    queries: np.ndarray,
    k: int,
    block_size: int,
    candidate_margin: int,
) -> None:
    for name, array in (("train", train), ("queries", queries)):
        if array.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{name} must be float32, got {array.dtype}")
    if train.shape[1] != queries.shape[1]:
        raise ValueError(
            f"Dimension mismatch: train has {train.shape[1]}, queries have {queries.shape[1]}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > train.shape[0]:
        raise ValueError(
            f"k={k} is larger than the number of train vectors ({train.shape[0]})")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if candidate_margin < 0:
        raise ValueError(
            f"candidate_margin must be >= 0, got {candidate_margin}")
