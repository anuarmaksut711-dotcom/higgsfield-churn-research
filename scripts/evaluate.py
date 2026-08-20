"""Run the strict three-fold walk-forward temporal evaluation."""

import argparse

import _bootstrap  # noqa: F401

from churn_pipeline import run_walk_forward_evaluation
from src.churn.config import DEFAULT_CONFIG


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_CONFIG.iterations)
    parser.add_argument("--thread-count", type=int, default=DEFAULT_CONFIG.thread_count)
    args = parser.parse_args()
    run_walk_forward_evaluation(args.iterations, args.thread_count)
