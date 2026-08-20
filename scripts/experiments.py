"""Entry point for baseline and focused research experiments."""

import argparse

import _bootstrap  # noqa: F401

from baseline_experiments import run as run_baselines
from research_experiments import (
    run_ablation,
    run_final_window_evaluation,
    run_hyperparameter_screen,
    run_observation_windows,
    run_seed_ensemble,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        choices=["baselines", "ablation", "windows", "ensemble", "tuning", "final-window"],
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--thread-count", type=int, default=-1)
    args = parser.parse_args()
    if args.study == "baselines":
        run_baselines(650, args.thread_count)
    elif args.study == "ablation":
        run_ablation(args.iterations, args.thread_count)
    elif args.study == "windows":
        run_observation_windows(args.iterations, args.thread_count)
    elif args.study == "ensemble":
        run_seed_ensemble(args.iterations, args.thread_count)
    elif args.study == "tuning":
        run_hyperparameter_screen(args.thread_count)
    else:
        run_final_window_evaluation(args.iterations, args.thread_count)
