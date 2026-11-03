"""Synthetic class-imbalance subsampling.

Given the labels of a balanced dataset, build a plan that says which examples
go into the index (in insertion order) and which are held out as queries.

Class sizes follow an exponential profile over class rank i = 0..C-1:

    n_i = n_max * rho ** (-i / (C - 1))

so the largest class has n_max examples and the smallest has n_max / rho.
Ranks are assigned to classes by a seeded random permutation, so class size is
not correlated with class semantics.

Properties of the plan:
  * Train and query sets are disjoint by construction.
  * For a fixed seed, the query set is the same for every rho, and the train
    set of a smaller rho is a subset of the train set of a larger one.
  * Queries are the same number per class.
  * The insertion order is a global shuffle (stationary stream) or, for the
    adversarial experiment, grouped by class.

This module only deals with indices. Writing the exported files is the job of
export.py.

Usage (prints the plan summary, writes nothing):
    python -m pipeline.subsample --dataset dbpedia14 --rho 100 --seed 0 --n-max 30000
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("subsample")

ORDER_SHUFFLED = "shuffled"
ORDER_BY_CLASS = "by_class"
MIN_QUERIES_PER_CLASS = 20
MIN_TOTAL_QUERIES = 10_000
MAX_CLASSES_LISTED = 50


@dataclass(frozen=True)
class SubsamplePlan:
    """Result of subsampling. All indices refer to rows of the cached dataset."""

    train_indices: np.ndarray  # int64, in insertion order
    query_indices: np.ndarray  # int64, grouped by class id
    class_sizes: np.ndarray  # int64, shape (C,), train size per class id
    queries_per_class: int
    rho: float
    n_max: int
    seed: int
    order: str


def class_size_profile(num_classes: int, n_max: int, rho: float) -> np.ndarray:
    """Return the class sizes by rank (largest first), each at least 1."""
    if rho < 1:
        raise ValueError(f"rho must be >= 1, got {rho}")
    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if num_classes == 1 or rho == 1:
        return np.full(num_classes, n_max, dtype=np.int64)
    ranks = np.arange(num_classes)
    sizes = n_max * rho ** (-ranks / (num_classes - 1))
    return np.maximum(np.rint(sizes).astype(np.int64), 1)


def default_queries_per_class(num_classes: int) -> int:
    """At least 20 per class and at least about 10k queries in total."""
    return max(MIN_QUERIES_PER_CLASS, math.ceil(MIN_TOTAL_QUERIES / num_classes))


def build_plan(
    labels: np.ndarray,
    rho: float,
    n_max: int,
    seed: int,
    queries_per_class: int | None = None,
    order: str = ORDER_SHUFFLED,
) -> SubsamplePlan:
    """Build the subsampling plan for one (rho, seed) combination."""
    if order not in (ORDER_SHUFFLED, ORDER_BY_CLASS):
        raise ValueError(f"Unknown insertion order '{order}'")

    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    available = np.bincount(labels, minlength=num_classes)
    if (available == 0).any():
        missing = np.flatnonzero(available == 0).tolist()
        raise ValueError(f"Classes with no examples: {missing}")

    if queries_per_class is None:
        queries_per_class = default_queries_per_class(num_classes)

    # Independent random streams, so that changing one choice (for example the
    # insertion order) never changes the others.
    rank_seq, class_seq, order_seq = np.random.SeedSequence(seed).spawn(3)
    rng_rank = np.random.default_rng(rank_seq)
    rng_class = np.random.default_rng(class_seq)
    rng_order = np.random.default_rng(order_seq)

    sizes_by_rank = class_size_profile(num_classes, n_max, rho)
    class_of_rank = rng_rank.permutation(num_classes)
    class_sizes = np.empty(num_classes, dtype=np.int64)
    class_sizes[class_of_rank] = sizes_by_rank

    train_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for class_id in range(num_classes):
        needed = int(class_sizes[class_id]) + queries_per_class
        if needed > available[class_id]:
            raise ValueError(
                f"Class {class_id} has {available[class_id]} examples but needs "
                f"{needed} ({class_sizes[class_id]} train + {queries_per_class} queries)"
            )
        # The stream is consumed in class order and its use does not depend on
        # rho, so each class gets the same permutation for every rho.
        permuted = rng_class.permutation(np.flatnonzero(labels == class_id))
        train_parts.append(permuted[: class_sizes[class_id]])
        query_parts.append(permuted[len(permuted) - queries_per_class:])

    train = np.concatenate(train_parts)
    if order == ORDER_SHUFFLED:
        train = rng_order.permutation(train)

    return SubsamplePlan(
        train_indices=train.astype(np.int64),
        query_indices=np.concatenate(query_parts).astype(np.int64),
        class_sizes=class_sizes,
        queries_per_class=queries_per_class,
        rho=float(rho),
        n_max=int(n_max),
        seed=int(seed),
        order=order,
    )


def expected_total(num_classes: int, n_max: int, rho: float) -> float:
    """Continuous approximation of the index size: n_max * C * (1 - 1/rho) / ln(rho)."""
    if rho == 1:
        return float(n_max * num_classes)
    return n_max * num_classes * (1 - 1 / rho) / math.log(rho)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the subsampling plan for a cached dataset.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (folder inside the cache).")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--rho", type=float, required=True,
                        help="Imbalance ratio, >= 1.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-max", type=int, required=True,
                        help="Size of the largest class.")
    parser.add_argument("--queries-per-class", type=int, default=None)
    parser.add_argument(
        "--order", choices=[ORDER_SHUFFLED, ORDER_BY_CLASS], default=ORDER_SHUFFLED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    labels_path = args.cache_dir / args.dataset / "labels.i32"
    if not labels_path.exists():
        LOGGER.error("Labels file not found: %s", labels_path)
        return 1
    labels = np.fromfile(labels_path, dtype="<i4")

    plan = build_plan(labels, args.rho, args.n_max, args.seed,
                      args.queries_per_class, args.order)

    num_classes = len(plan.class_sizes)
    LOGGER.info("Dataset: %d examples, %d classes", len(labels), num_classes)
    LOGGER.info("rho=%g seed=%d n_max=%d order=%s",
                plan.rho, plan.seed, plan.n_max, plan.order)
    LOGGER.info(
        "Train: %d examples (continuous estimate: %.0f)",
        len(plan.train_indices),
        expected_total(num_classes, plan.n_max, plan.rho),
    )
    LOGGER.info(
        "Queries: %d (%d per class)", len(
            plan.query_indices), plan.queries_per_class
    )
    LOGGER.info(
        "Class sizes: largest=%d smallest=%d", plan.class_sizes.max(), plan.class_sizes.min()
    )
    if num_classes <= MAX_CLASSES_LISTED:
        for class_id in np.argsort(-plan.class_sizes, kind="stable"):
            LOGGER.info("  class %3d: %d", class_id,
                        plan.class_sizes[class_id])
    return 0


if __name__ == "__main__":
    sys.exit(main())
