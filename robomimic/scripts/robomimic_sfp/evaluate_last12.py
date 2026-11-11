"""Evaluate the last 12 RoboMimic checkpoints and report Avg5 and mean12."""

import argparse
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys


ROUNDS = tuple(range(228, 251, 2))
EVALUATOR = Path(__file__).with_name("rollout_table3_paper_action_anchor_v3_switchaware.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.dataset_path.is_file():
        parser.error(f"Dataset does not exist: {args.dataset_path}")
    checkpoints = [args.checkpoint_dir / f"global_round_{step:04d}.pt" for step in ROUNDS]
    missing = [path.name for path in checkpoints if not path.is_file()]
    if missing:
        parser.error(f"Missing required checkpoints: {', '.join(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    scores = []
    for step, checkpoint in zip(ROUNDS, checkpoints):
        output = args.output_dir / f"round_{step:04d}.json"
        command = [
            sys.executable, str(EVALUATOR),
            "--dataset-path", str(args.dataset_path), "--checkpoint", str(checkpoint),
            "--output", str(output), "--device", args.device,
            "--num-rollouts", "50", "--start-seed", "100000", "--max-steps", "500",
            "--action-horizon", "8", "--carry-previous-action",
        ]
        with output.with_suffix(".log").open("x", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["num_rollouts"] != 50 or result["integration_steps_per_action"] != 1:
            raise ValueError(f"Unexpected evaluation protocol in {output}")
        scores.append(result["success_rate"])

    summary = {
        "rounds": ROUNDS,
        "rollouts_per_checkpoint": 50,
        "success_rates": scores,
        "avg5": mean(sorted(scores, reverse=True)[:5]),
        "mean12": mean(scores),
    }
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
