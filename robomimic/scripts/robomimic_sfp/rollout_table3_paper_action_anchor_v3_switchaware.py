#!/usr/bin/env python3
"""Strict rollout for switch-aware action-anchor checkpoints."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.robomimic_sfp import rollout_table3_papergrid_dp_v1 as base
from streaming_flow_policy.robomimic_sfp.model_v22_paper_aligned import (
    SFPConditionalUnet1D,
)


EXPECTED_ALIGNMENT = "previous_action_then_current_through_t_plus_15"
EXPECTED_TIME = "continuous_t_in_0_1_unscaled"
EXPECTED_SAMPLING = "uniform_0_1_with_switch_segment_oversampling_p0.5"


def cli_value(flag):
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Missing required argument {flag}") from exc


def validate_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    expected = {
        "action_alignment": EXPECTED_ALIGNMENT,
        "time_conditioning": EXPECTED_TIME,
        "time_sampling": EXPECTED_SAMPLING,
        "train_demo_policy": "all_300",
        "adam_betas": [0.9, 0.999],
        "biased_gripper_sampling": True,
        "biased_gripper_prob": 0.5,
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: "
                f"expected {value!r}, got {args.get(key)!r}"
            )
    if int(checkpoint.get("table3", {}).get("num_train_demos", -1)) != 300:
        raise ValueError("Checkpoint was not trained on all 300 demonstrations")


def annotate_output(path):
    output = Path(path)
    payload = json.loads(output.read_text())
    payload.update(
        {
            "max_steps": int(cli_value("--max-steps")),
            "action_alignment": EXPECTED_ALIGNMENT,
            "time_conditioning": EXPECTED_TIME,
            "time_sampling": EXPECTED_SAMPLING,
            "evaluation_protocol": {
                "start_seed": int(cli_value("--start-seed")),
                "num_rollouts": int(cli_value("--num-rollouts")),
                "max_steps": int(cli_value("--max-steps")),
                "action_horizon": int(cli_value("--action-horizon")),
                "carry_previous_action": "--carry-previous-action" in sys.argv,
            },
        }
    )
    output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    base.SFPConditionalUnet1D = SFPConditionalUnet1D
    if "--help" in sys.argv or "-h" in sys.argv:
        base.main()
        raise SystemExit(0)
    checkpoint_path = cli_value("--checkpoint")
    output_path = cli_value("--output")
    validate_checkpoint(checkpoint_path)
    base.main()
    annotate_output(output_path)
