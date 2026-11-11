#!/usr/bin/env python3
"""Recompute Push-T Table 2 aggregates from the archived evaluation metrics."""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = Path(__file__).resolve().parent / "evidence" / "table2_checkpoint_metrics.csv"
ROUNDS = tuple(range(2, 51, 2))
SEEDS = (16, 42, 43)
CLIENT_COUNTS = (2, 4, 8)
METHODS = ("fedavg", "gab")


def read_runs():
    runs = defaultdict(list)
    with METRICS.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            k = int(row["client_count"])
            method = row["method"]
            replica = int(row["replicate"])
            seed = int(row["training_seed"])
            round_index = int(row["checkpoint_round"])
            score = float(row["mean_score_x100"])
            success = float(row["success_rate"])
            episodes = int(row["episodes"])
            successful_episodes = int(row["successful_episodes"])
            if k not in CLIENT_COUNTS or method not in METHODS or replica not in (1, 2, 3):
                raise ValueError(f"Unexpected run key: {(k, method, replica)}")
            if seed != SEEDS[replica - 1] or episodes != 50:
                raise ValueError(f"Unexpected evaluation configuration: {(k, method, replica, round_index)}")
            if not 0 <= score <= 100 or not 0 <= successful_episodes <= episodes:
                raise ValueError(f"Out-of-range evaluation metrics: {(k, method, replica, round_index)}")
            if not math.isclose(success, successful_episodes / episodes, abs_tol=1e-12):
                raise ValueError(f"Success count does not match the stored rate: {(k, method, replica, round_index)}")
            runs[k, method, replica].append((round_index, score, success))
    return runs


def summarize(runs, k, method):
    scores = []
    successes = []
    for replica in (1, 2, 3):
        records = runs[k, method, replica]
        if tuple(sorted(record[0] for record in records)) != ROUNDS:
            raise ValueError(f"Incomplete or duplicate checkpoint rounds: {(k, method, replica)}")
        # Stable sorting preserves the archived order of equally scored rounds.
        top_five = sorted(records, key=lambda record: record[1], reverse=True)[:5]
        scores.append(statistics.mean(record[1] for record in top_five))
        successes.append(statistics.mean(record[2] for record in top_five))
    return statistics.mean(scores), statistics.stdev(scores), statistics.mean(successes), statistics.stdev(successes)


def main():
    runs = read_runs()
    print("| K | Method | Coverage score x100 | Success rate |")
    print("|---:|---|---:|---:|")
    for k in CLIENT_COUNTS:
        for method in METHODS:
            mean_score, sd_score, mean_success, sd_success = summarize(runs, k, method)
            print(f"| {k} | {method} | {mean_score:.2f} +/- {sd_score:.2f} | "
                  f"{mean_success:.3f} +/- {sd_success:.3f} |")


if __name__ == "__main__":
    main()
