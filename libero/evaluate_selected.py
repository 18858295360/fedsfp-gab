"""Run the recorded two-stage checkpoint selection protocol for one arm."""

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--lang-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for stage results and summary.json.")
    args = parser.parse_args()
    rounds = range(4, 21)
    checkpoints = {r: args.checkpoint_dir.resolve() / f"global_round_{r:04d}.pt"
                   for r in rounds}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        parser.error(f"missing checkpoints: {', '.join(missing)}")
    if not args.lang_npz.is_file():
        parser.error("--lang-npz must point to the same embeddings used for training")

    release_dir = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def score(round_number, episodes_per_task, stage):
        result = output / f"{stage}_r{round_number:02d}.json"
        command = [
            sys.executable, "-m", "scripts.robomimic_sfp.rollout_libero",
            "--checkpoint", str(checkpoints[round_number]),
            "--output", str(result),
            "--benchmark", "libero_10",
            "--tasks", "10",
            "--rollouts-per-task", str(episodes_per_task),
            "--max-steps", "400",
            "--action-horizon", "8",
            "--img-size", "128",
            "--start-seed", "100000",
            "--lang-npz", str(args.lang_npz.resolve()),
        ]
        subprocess.run(command, check=True, cwd=release_dir)
        record = json.loads(result.read_text(encoding="utf-8"))
        value = float(record["success_rate"])
        if (record["num_tasks"] != 10 or record["num_rollouts"] != 10 * episodes_per_task
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"incomplete or invalid evaluation: {result}")
        return value

    screening = {r: score(r, 2, "TS1") for r in rounds}
    selected = sorted(sorted(rounds, key=lambda r: (-screening[r], r))[:5])
    reevaluation = {r: score(r, 20, "TS2") for r in selected}
    summary = {
        "screening_success_rates": screening,
        "selected_rounds": selected,
        "reevaluation_success_rates": reevaluation,
        "arm_success_rate": statistics.mean(reevaluation.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Arm success rate: {summary['arm_success_rate']:.4f}")


if __name__ == "__main__":
    main()
