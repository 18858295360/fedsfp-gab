#!/usr/bin/env python3
"""Deterministic v3 switch-aware trainer — reproducible per seed.

Same recipe as train_table3_paper_action_anchor_v3_switchaware.py, but with
full determinism (CUBLAS workspace, cuDNN deterministic, torch deterministic
algorithms) and num_workers forced to 0, so the SAME seed reproduces the SAME
training trajectory and the SAME checkpoints. This fixes the finding that
num_workers=4 made "seed56" non-reproducible (76.4 not re-obtainable).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.robomimic_sfp import train_table3_papergrid_dp_v1 as base
from streaming_flow_policy.robomimic_sfp.model_v22_paper_aligned import (
    SFPConditionalUnet1D,
)
from streaming_flow_policy.robomimic_sfp.table3_papergrid_switchaware_v1 import (
    Table3PaperGridSwitchAwareDataset,
)


class AllDemoSwitchAwareDataset(Table3PaperGridSwitchAwareDataset):
    def __init__(self, *a, **k):
        k["val_ratio"] = 0.0
        k["switch_probability"] = float(__import__("os").environ.get("SFP_SWITCH_PROB", "0.5"))
        super().__init__(*a, **k)


_parse_args = base.parse_args
_adamw = torch.optim.AdamW


def parse_args():
    args = _parse_args()
    args.action_alignment = "previous_action_then_current_through_t_plus_15"
    args.time_conditioning = "continuous_t_in_0_1_unscaled"
    args.time_sampling = "uniform_0_1_with_switch_segment_oversampling_p0.5"
    args.train_demo_policy = "all_300"
    args.adam_betas = [0.9, 0.999]
    args.biased_gripper_sampling = True
    args.biased_gripper_prob = 0.5
    args.deterministic = True
    args.num_workers = 0  # force single-process data loading for determinism
    args.compatible_rollout = "rollout_table3_paper_action_anchor_v3_switchaware.py"
    return args


def paper_adamw(params, *a, **k):
    k["betas"] = (0.9, 0.999)
    return _adamw(params, *a, **k)


base.parse_args = parse_args
base.Table3PaperGridDPDataset = AllDemoSwitchAwareDataset
base.SFPConditionalUnet1D = SFPConditionalUnet1D
base.torch.optim.AdamW = paper_adamw


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    base.main()
