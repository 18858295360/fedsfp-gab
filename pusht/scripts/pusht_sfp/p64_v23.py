import argparse
import csv
import json
import random
import math
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers.optimization import get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pusht_sfp.p39_noniid_severity_gate import make_severity_split
from scripts.pusht_sfp.moment_persistence import reset_adam_first_moment
from scripts.pusht_sfp.moment_persistence import (
    leave_one_out_cosines,
    moment_retention,
    project_adam_first_moment_against_gradients,
    scale_adam_first_moment,
)
from scripts.pusht_sfp.phase1_fedavg_state_ablation import (
    cpu_state_dict,
    make_optimizer,
    make_reset_scheduler,
    set_lr,
    state_delta,
)
from scripts.pusht_sfp.phase1_iid_action_fedavg import average_state_dicts
from scripts.pusht_sfp.train_sfp import build_dataset, build_policy


METHOD_CONFIGS = {
    "raw_reset_both": {
        "method": "Raw FedAvg",
        "keep_optimizer": False,
        "keep_scheduler": False,
    },
    "raw_global": {
        "method": "Raw FedAvg (global cosine)",
        "keep_optimizer": False,
        "keep_scheduler": True,
    },
    "raw_const": {
        "method": "Raw FedAvg (constant LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
    },
    "persistent_const": {
        "method": "Persistent Adam (constant LR)",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "const_lr": True,
    },
    "tail_anneal": {
        "method": "Tail-Annealed FedAvg (area-matched terminal cooling)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "tail_anneal": True,
    },
    "fedhyper_cl": {
        "method": "FedHyper-CL (client hypergradient LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "fedhyper_cl": True,
    },
    "fedhyper_sl": {
        "method": "FedHyper-SL (server-set uniform client LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "fedhyper_sl": True,
    },
    "twod_lrd": {
        "method": "2D-LRD (round x epoch decay, epoch-mapped)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "twod_lrd": True,
    },
    "twod_lrd_am": {
        "method": "2D-LRD area-matched",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "twod_lrd": True,
        "area_matched": True,
    },
    "linear_global": {
        "method": "Raw FedAvg (training-wide linear decay)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "shape_lr": "linear",
    },
    "persistent_linear": {
        "method": "Persistent Adam + linear decay (best-schedule x kept-state)",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "shape_lr": "linear",
    },
    "persistent_tail": {
        "method": "Persistent Adam + cosine envelope + tail cooling",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "tail_anneal": True,
    },
    "step_global": {
        "method": "Raw FedAvg (step decay x0.1 at 50 and 75 percent)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "shape_lr": "step",
    },
    "const_tail": {
        "method": "Raw FedAvg (constant envelope + area-matched tail cooling)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "shape_lr": "const_tail",
    },
    "fedexp": {
        "method": "FedExP (server extrapolation; const client LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "fedexp": True,
    },
    "fedhyper_g": {
        "method": "FedHyper-G (server hypergradient LR; const client LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "fedhyper_g": True,
    },
    "fedexp_cosine": {
        "method": "FedExP on cosine clients (adaptivity-over-correct-prior test)",
        "keep_optimizer": False,
        "keep_scheduler": True,
        "fedexp": True,
    },
    "fedavgm": {
        "method": "FedAvgM (server momentum 0.9; const client LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "server_momentum": 0.9,
    },
    "scaffold": {
        "method": "SCAFFOLD (control-variate drift correction; const client LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "scaffold": True,
    },
    "server_ema": {
        "method": "Server weight-EMA aggregation (beta 0.7; const client LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "server_ema": 0.7,
    },
    "head_selective_linear": {
        "method": "Head-Selective + linear LR (selective-m + local-v + best schedule)",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "shape_lr": "linear",
        "head_selective_reset": True,
    },
    "head_selective_half_linear": {
        "method": "Head-Selective half-retention + linear LR",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "shape_lr": "linear",
        "head_selective_reset": True,
        "head_conflict_retention": 0.5,
    },
    "fisher_agg": {
        "method": "Fisher-weighted aggregation (per-param v-weighted merge; reset m; const LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "fisher_agg": True,
    },
    "ties_merge": {
        "method": "TIES-merge aggregation (sign-elect + agree-mask; VP base: reset m, local v, linear)",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "shape_lr": "linear",
        "zero_component": "exp_avg",
        "ties_merge": True,
    },
    "sign_agree": {
        "method": "Sign-agreement masked aggregation (elect sign, drop conflicting; const LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "ties_merge": True,
    },
    "fedprox": {
        "method": "FedProx (proximal mu=0.01; reset both; const LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "prox_mu": 0.01,
    },
    "fedsfp_vt": {
        "method": "FedSFP-VT (reset m + shared-v transfer + cosine envelope + area-matched tail)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "fedadamw_style": True,
        "tail_anneal": True,
    },
    "fedsfp_vp": {
        "method": "FedSFP-VP (reset m + client-persistent v + global linear LR)",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "shape_lr": "linear",
        "zero_component": "exp_avg",
    },
    "pgs_L0": {"method": "FedSFP-PGS lambda=0.00 (=shared v)", "keep_optimizer": True,
               "keep_scheduler": False, "shape_lr": "linear", "zero_component": "exp_avg",
               "pgs_lambda": 0.0},
    "pgs_L25": {"method": "FedSFP-PGS lambda=0.25", "keep_optimizer": True,
                "keep_scheduler": False, "shape_lr": "linear", "zero_component": "exp_avg",
                "pgs_lambda": 0.25},
    "pgs_L50": {"method": "FedSFP-PGS lambda=0.50", "keep_optimizer": True,
                "keep_scheduler": False, "shape_lr": "linear", "zero_component": "exp_avg",
                "pgs_lambda": 0.50},
    "pgs_L75": {"method": "FedSFP-PGS lambda=0.75", "keep_optimizer": True,
                "keep_scheduler": False, "shape_lr": "linear", "zero_component": "exp_avg",
                "pgs_lambda": 0.75},
    "pgs_L100": {"method": "FedSFP-PGS lambda=1.00 (=local v = VP)", "keep_optimizer": True,
                 "keep_scheduler": False, "shape_lr": "linear", "zero_component": "exp_avg",
                 "pgs_lambda": 1.0},
    "svr_gate10": {"method": "FedSFP-SVR warm-only (persist v early, refresh after r10; reset m; linear)",
                   "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
                   "zero_component": "exp_avg", "svr_refresh_after": 10},
    "svr_gate20": {"method": "FedSFP-SVR warm-only (refresh v after r20)",
                   "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
                   "zero_component": "exp_avg", "svr_refresh_after": 20},
    "svr_gate30": {"method": "FedSFP-SVR warm-only (refresh v after r30)",
                   "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
                   "zero_component": "exp_avg", "svr_refresh_after": 30},
    "ftp2": {"method": "FedSFP-FTP 2-bin (flow-time-conditioned preconditioning: split batch by t, per-bin Adam m/v; reset m; persist per-bin v; linear)",
             "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
             "zero_component": "exp_avg", "ftp_bins": 2},
    "ftp3": {"method": "FedSFP-FTP 3-bin (flow-time-conditioned preconditioning; reset m; persist per-bin v; linear)",
             "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
             "zero_component": "exp_avg", "ftp_bins": 3},
    "ftp4": {"method": "FedSFP-FTP 4-bin (flow-time-conditioned preconditioning; reset m; persist per-bin v; linear)",
             "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
             "zero_component": "exp_avg", "ftp_bins": 4},
    "strict_eqw": {"method": "Strict mode-fair FedAvg (equal aggregation weights AND equal local steps per round; cleanest mode-fairness control)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "equal_weight": True, "equal_steps": True},
    "mca_head_strict": {"method": "FedSFP-MCA head under strict mode-fair budget (equal weights + equal local steps; final_conv mode-specific)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "equal_weight": True, "equal_steps": True,
             "mca_prefixes": ["velocity_net.final_conv"]},
    "fcc3_eqw": {"method": "FCC 3-bin under mode-equal weights (module-wise conflict diagnostics: D, cancellation C, norm ratio R per module per flow bin)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "fcc_bins": 3, "fcc_alpha": 1.0, "equal_weight": True},
    "linear_eqw": {"method": "Linear FedAvg, mode-equal weights (1/N per client instead of episode-proportional; rotation main setting)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "equal_weight": True},
    "pgs_L0_eqw": {"method": "PGS lambda=0 (shared v-bar) + mode-equal weights",
             "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
             "zero_component": "exp_avg", "pgs_lambda": 0.0, "equal_weight": True},
    "mca_head": {"method": "FedSFP-MCA head (shared trunk FedAvg mode-equal; velocity_net.final_conv kept mode-specific per client; evaluated per mode)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "equal_weight": True, "mca_prefixes": ["velocity_net.final_conv"]},
    "mca_up": {"method": "FedSFP-MCA adapter (shared trunk FedAvg mode-equal; velocity_net.up_modules.1 + final_conv mode-specific)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "equal_weight": True, "mca_prefixes": ["velocity_net.final_conv", "velocity_net.up_modules.1"]},
    "fedper_style": {"method": "FedPer-style (shared trunk averaged; ENTIRE velocity_net kept local per client; mode-equal)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "equal_weight": True, "mca_prefixes": ["velocity_net"]},
    "fca3": {"method": "FedSFP-FCA 3-bin (flow-conditioned aggregation: aggregate clients within matched flow stages, then combine stages; reset Adam each round; global linear LR)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "fca_bins": 3, "fca_weight": "binshare"},
    "fca3_ctrl": {"method": "FCA control (identical per-bin training, but client-size weights -> reduces exactly to FedAvg aggregation; isolates the aggregation-granularity effect)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "fca_bins": 3, "fca_weight": "clientsize"},
    "fcv3": {"method": "FedSFP-FCV 3-bin (flow-conditioned control variates: per-flow-bin SCAFFOLD correction c_b - c_ib; per-bin micro-steps; FedAvg; reset Adam; linear)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "fcv_bins": 3},
    "fcv3_d05": {"method": "FedSFP-FCV 3-bin damped 0.5 (undamped diverged r7: control-variate feedback; corr *= 0.5)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "fcv_bins": 3, "fcv_damp": 0.5},
    "fcv3_g": {"method": "FedSFP-FCV 3-bin gradient-unit CV (SCAFFOLD Option-I: c_ib = mean bin-b raw grad, c_b = cross-client mean; unit-consistent, no damping; drift-based variants diverged)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "fcv_bins": 3, "fcv_grad_cv": True},
    "fcc3": {"method": "FedSFP-FCC 3-bin (federated flow-conflict coordination: server reweights next-round flow-time sampling by cross-client per-bin update disagreement D_b; FedAvg; reset Adam; linear)",
             "keep_optimizer": False, "keep_scheduler": False, "shape_lr": "linear",
             "fcc_bins": 3, "fcc_alpha": 1.0},
    "warm_gated_vp5": {
        "method": "Warm-gated VP (persist v after round 5; reset m; linear)",
        "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
        "zero_component": "exp_avg", "warm_persist": 5,
    },
    "warm_gated_vp10": {
        "method": "Warm-gated VP (persist v after round 10; reset m; linear)",
        "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
        "zero_component": "exp_avg", "warm_persist": 10,
    },
    "warm_gated_vp20": {
        "method": "Warm-gated VP (persist v after round 20; reset m; linear)",
        "keep_optimizer": True, "keep_scheduler": False, "shape_lr": "linear",
        "zero_component": "exp_avg", "warm_persist": 20,
    },
    "persistent_vsync": {
        "method": "Persistent-VSync (keep m per-client, sync second-moment v globally each round)",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "vsync": True,
    },
    "persistent_vsync_linear": {
        "method": "Persistent-VSync + linear decay (best schedule x v-sync)",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "shape_lr": "linear",
        "vsync": True,
    },
    "pers_m_only": {
        "method": "Persistent m-only (v zeroed each round; const LR)",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "const_lr": True,
        "zero_component": "exp_avg_sq",
    },
    "pers_v_only": {
        "method": "Persistent v-only (m zeroed each round; const LR)",
        "keep_optimizer": True,
        "keep_scheduler": False,
        "const_lr": True,
        "zero_component": "exp_avg",
    },
    "fedadamw_style": {
        "method": "FedAdamW-style (fresh m; server-averaged v inherited; const LR)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "fedadamw_style": True,
    },
    "plateau": {
        "method": "Server ReduceLROnPlateau (x0.5 after 3 stale rounds; const base)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "plateau_lr": True,
    },
    "drift_lr": {
        "method": "Drift-normalized client LR (lr_i ~ median||d||/||d_i||; const base)",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "drift_lr": True,
    },
    "fedhyper_cl_norm": {
        "method": "FedHyper-CL sign-normalized (ours: beta += 0.05*sign(X+Y/K))",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "fedhyper_cl": True,
        "norm_update": True,
    },
    "fedhyper_sl_norm": {
        "method": "FedHyper-SL sign-normalized (ours: beta += 0.05*sign(s))",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "fedhyper_sl": True,
        "norm_update": True,
    },
    "fedhyper_g_norm": {
        "method": "FedHyper-G sign-normalized (ours: alpha += 0.05*sign(s))",
        "keep_optimizer": False,
        "keep_scheduler": False,
        "const_lr": True,
        "fedhyper_g": True,
        "norm_update": True,
    },
    "persistent_both": {
        "method": "Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
    },

    "persistent_eqw": {
        "method": "equal-weight Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
             "equal_weight": True},
    "endpoint_projected": {
        "method": "Endpoint-Projected Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "endpoint_projection": True,
    },
    "head_selective": {
        "method": "Head-Selective Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
    },
    "head_selective_half": {
        "method": "Head-Selective Half-Retention Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_conflict_retention": 0.5,
    },
    "head_selective_075": {
        "method": "Head-Selective 0.75-Retention Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_conflict_retention": 0.75,
    },
    "head_selective_late20": {
        "method": "Head-Selective Late-20 Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_gate_start_round": 21,
    },
    "head_selective_late20_half": {
        "method": "Head-Selective Late-20 Half-Retention Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_gate_start_round": 21,
        "head_conflict_retention": 0.5,
    },
    "head_selective_late20_075": {
        "method": "Head-Selective Late-20 0.75-Retention Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_gate_start_round": 21,
        "head_conflict_retention": 0.75,
    },
    "head_selective_late30_half": {
        "method": "Head-Selective Late-30 Half-Retention Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_gate_start_round": 31,
        "head_conflict_retention": 0.5,
    },
    "head_selective_half_svm09": {
        "method": "Head-Selective Half-Retention + Server-Momentum(0.9) Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_conflict_retention": 0.5,
        "server_momentum": 0.9,
    },
    "head_selective_svm09": {
        "method": "Head-Selective + Server-Momentum(0.9) Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "server_momentum": 0.9,
    },
    "head_selective_half_svm05": {
        "method": "Head-Selective Half-Retention + Server-Momentum(0.5) Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_conflict_retention": 0.5,
        "server_momentum": 0.5,
    },
    "head_selective_half_svm09d": {
        "method": "Head-Selective Half-Retention + Dampened Server-Momentum(0.9) Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_conflict_retention": 0.5,
        "server_momentum": 0.9,
        "server_momentum_dampened": True,
    },
    "head_selective_half_svmdec": {
        "method": "Head-Selective Half-Retention + Decayed Server-Momentum(0.9->0@40) Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_conflict_retention": 0.5,
        "server_momentum": 0.9,
        "server_momentum_decay_end": 40,
    },
    "head_selective_half_svm09t": {
        "method": "Head-Selective Half-Retention + Trunk-Only Server-Momentum(0.9) Persistent FedAvg",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "head_selective_reset": True,
        "head_conflict_retention": 0.5,
        "server_momentum": 0.9,
        "server_momentum_scope": "trunk",
    },
    "v_only": {
        "method": "v-only moment persistence",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "reset_first_moment": True,
    },
    "fixed_decay_half": {
        "method": "Fixed 0.5 first-moment decay",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "moment_mode": "fixed_half",
    },
    "loo_drift_aware": {
        "method": "Leave-One-Out Drift-Aware Moment Persistence",
        "keep_optimizer": True,
        "keep_scheduler": True,
        "moment_mode": "loo",
    },
}

DEFAULT_CHECKPOINT_ROUNDS = list(range(2, 51, 2))   # the 25 leaderboard checkpoints


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_csv(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        "P47 image-action Non-IID S1 Persistent target-generalization run"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-split-json", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260709)
    parser.add_argument("--severity-p", type=float, default=1.0)
    parser.add_argument("--centralized", action="store_true",
                        help="collapse the split to ONE client holding all episodes "
                             "(true centralized training; pair with persistent_both)")
    # Defaults below reproduce the byinit leaderboard setting, so omitting a
    # flag can no longer silently produce a run that is not comparable to it.
    parser.add_argument("--seeds", default="16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-rounds", default=",".join(str(x) for x in DEFAULT_CHECKPOINT_ROUNDS))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-num-tests", type=int, default=50)
    parser.add_argument("--env-start-seed", type=int, default=1300)
    parser.add_argument("--max-rollout-steps", type=int, default=300)
    parser.add_argument("--methods", default="persistent_both")
    parser.add_argument("--endpoint-tau-min", type=float, default=0.7)
    parser.add_argument("--endpoint-projection-epsilon", type=float, default=1e-12)
    parser.add_argument("--initial-state-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--weight-by-samples", action="store_true",
                        help="aggregate clients by training-sample count (canonical FTP3) "
                             "instead of episode count")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay; exactly one update per batch. Canonical FTP3 enables "
                             "EMA and FedAvgs the EMA weights alongside the raw weights.")
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="fixed parameter updates per local epoch (canonical FTP3 uses 100 "
                             "-> 7*100=700 per client per round, 35,000-step global LR clock). "
                             "0 = iterate the whole loader.")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="global linear warmup length (canonical FTP3 uses 10)")
    parser.add_argument("--obs-mode", choices=["image", "state"], default="image",
                        help="observation modality used for policy conditioning")
    parser.add_argument("--imitation-mode", choices=["action", "state"], default="action",
                        help="action target: raw env action, or next pusher state")
    parser.add_argument("--anchor-broaden", type=float, default=0.0,
                        help="widen the flow anchor by this multiple of "
                             "sigma0 during local training, so a client "
                             "learns to restore from starting points its "
                             "own mode never reaches")
    parser.add_argument("--residual-gate", type=float, default=0.0,
                        help="weight each local sample by the frozen "
                             "round-start model's error on it, so a "
                             "client learns only what the global does "
                             "not know; 0 disables")
    parser.add_argument("--flare-experts", type=int, default=0,
                        help="wrap the velocity net as a windowed "
                             "mixture with this many experts; 0 leaves "
                             "the policy untouched")
    parser.add_argument("--flare-t-split", type=float, default=0.5,
                        help="flow time below which experts apply")
    parser.add_argument("--flare-gate-weight", type=float, default=0.1)
    parser.add_argument("--env-shift", type=float, default=0.0,
                        help="total hue spread in degrees across clients; each client gets a fixed rotation, evenly spaced over the spread")
    parser.add_argument("--env-shift-control", action="store_true",
                        help="homogeneous control: every client redraws the hue per sample from the same range, so the marginal appearance matches "
                             "the shifted arm while the per-client structure is removed")
    parser.add_argument("--client-fraction", type=float, default=1.0,
                        help="fraction of clients that train and aggregate "
                             "each round, resampled per round; 1.0 is full "
                             "participation")
    parser.add_argument("--global-anchor-frac", type=float, default=0.0,
                        help="probability that a training anchor is drawn "
                             "from the round-start global's generated "
                             "actions instead of the expert trajectory")
    parser.add_argument("--global-anchor-refresh", type=int, default=5,
                        help="regenerate the anchor tables every this many "
                             "rounds")
    parser.add_argument("--anchor-time-jitter", type=float, default=0.0,
                        help="draw the anchor around the trajectory at a flow "
                             "time up to this far from t, so the displacement "
                             "follows the manifold instead of a noise ball; "
                             "0 disables")
    parser.add_argument("--offsupport-consistency", type=float, default=0.0,
                        help="strength of the pull toward the round-start "
                             "global where the anchor lies outside the "
                             "client's own support; 0 disables")
    parser.add_argument("--osc-d0", type=float, default=1.5,
                        help="anchor distance, in units of sigma0, at which "
                             "the pull reaches half strength")
    parser.add_argument("--osc-tau", type=float, default=0.5,
                        help="width of the transition around --osc-d0")
    parser.add_argument("--flare-confidence-gate", action="store_true",
                        help="scale the expert modulation by the router's confidence, so an unsure router leaves the conditioning "
                             "unmodified instead of picking a row at random")
    parser.add_argument("--flare-router", choices=["gate", "prototype"],
                        default="gate",
                        help="prototype replaces the learned gate with a "
                             "per-client running mean of its own conditioning, "
                             "so each row has exactly one writer and federated "
                             "averaging cannot collapse it")
    parser.add_argument("--flare-cond-dim", type=int, default=0,
                        help="conditioning width; the network does not "
                             "keep it as an attribute, so pass the same "
                             "value the builder used")
    parser.add_argument("--custom-split", action="store_true",
                        help="use the split JSON's client_* partition verbatim "
                             "(rotation CW/CCW mode shift) instead of byinit severity split")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def phase_args(args, seed, run_dir, rounds):
    smoke = bool(args.smoke)
    return SimpleNamespace(
        out_dir=run_dir,
        model_type="sfps",
        imitation_mode=getattr(args, "imitation_mode", "action"),
        obs_mode=getattr(args, "obs_mode", "image"),
        epochs=rounds,
        batch_size=min(args.batch_size, 128) if smoke else args.batch_size,
        num_workers=0 if smoke else args.num_workers,
        lr=args.lr,
        weight_decay=1e-6,
        warmup_steps=getattr(args, "warmup_steps", 500),
        # canonical FTP3 knobs must be forwarded explicitly: run_variant reads
        # them off p_args, so anything missing here silently falls back
        steps_per_epoch=getattr(args, "steps_per_epoch", 0),
        # the residual-gate / anchor-broadening / FLARE knobs are read the
        # same way and were missing here, which made every arm that sets
        # one of them identical to the baseline
        anchor_broaden=getattr(args, "anchor_broaden", 0.0),
        anchor_time_jitter=getattr(args, "anchor_time_jitter", 0.0),
        global_anchor_frac=getattr(args, "global_anchor_frac", 0.0),
        global_anchor_refresh=getattr(args, "global_anchor_refresh", 5),
        env_shift=getattr(args, "env_shift", 0.0),
        env_shift_control=getattr(args, "env_shift_control", False),
        client_fraction=getattr(args, "client_fraction", 1.0),
        offsupport_consistency=getattr(args, "offsupport_consistency", 0.0),
        osc_d0=getattr(args, "osc_d0", 1.5),
        osc_tau=getattr(args, "osc_tau", 0.5),
        residual_gate=getattr(args, "residual_gate", 0.0),
        flare_experts=getattr(args, "flare_experts", 0),
        flare_t_split=getattr(args, "flare_t_split", 0.5),
        flare_gate_weight=getattr(args, "flare_gate_weight", 0.1),
        flare_cond_dim=getattr(args, "flare_cond_dim", 0),
        flare_router=getattr(args, "flare_router", "gate"),
        flare_confidence_gate=getattr(args, "flare_confidence_gate", False),
        weight_by_samples=getattr(args, "weight_by_samples", False),
        seed=seed,
        device=args.device,
        save_dir=run_dir,
        save_every=50,
        sigma=0.1,
        sigma0=0.1,
        sigma1=0.1,
        use_stabilized_objective=True,
        stabilization_k=2.0,
        use_time_decayed_noise=True,
        debug_stabilized_objective=False,
        debug_dataset_info=False,
        max_train_steps=None,
        init_ckpt_path=None,
        init_load_velocity_only=False,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        image_cond_dim=256,
        image_encoder="final_v1",
        # agent-pos conditioning is an image-line feature; build_policy rejects
        # it under --obs-mode state
        image_agent_pos_cond=(getattr(args, "obs_mode", "image") == "image"),
        image_key="img",
        image_random_crop_size=84,
        episode_indices_json=None,
        episode_indices_key=None,
        stats_episode_indices_key=None,
        ema_decay=float(getattr(args, "ema_decay", 0.999) or 0.0),
        rounds=rounds,
        fedavg_rounds=rounds,
        fedavg_local_epochs=args.local_epochs,
        local_steps_E=args.local_epochs,
        fedavg_warmup_steps=0,
        fedavg_max_local_steps=None,
        eval_num_tests=2 if smoke else args.eval_num_tests,
        env_start_seed=args.env_start_seed,
        max_rollout_steps=20 if smoke else args.max_rollout_steps,
        success_score_threshold=0.95,
        integration_steps_per_action=1,
        solver="euler",
        deterministic_z0_eval=True,
        endpoint_tau_min=getattr(args, "endpoint_tau_min", 0.7),
        endpoint_projection_epsilon=getattr(args, "endpoint_projection_epsilon", 1e-12),
        initial_state_path=getattr(args, "initial_state_path", None),
        force=args.force,
        smoke=smoke,
    )


def make_initial_state(args):
    if args.initial_state_path is not None:
        return torch.load(args.initial_state_path, map_location="cpu"), args.initial_state_path
    initial_path = args.out_dir / "initial_global.pth"
    if initial_path.exists() and not args.force:
        return torch.load(initial_path, map_location="cpu"), initial_path
    policy = build_policy(args)
    state = cpu_state_dict(policy)
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, initial_path)
    return state, initial_path


def _apply_env_shift(dataset, args, client_idx, n_clients):
    """Give this client its own appearance, or the shared randomised control.

    Angles are spread evenly over [-spread/2, +spread/2] so the clients differ
    from each other by the full spread while the mean appearance stays put --
    otherwise the shifted arm and its control would differ in average colour as
    well as in per-client structure.
    """
    import numpy as _np
    deg = float(getattr(args, "env_shift", 0.0) or 0.0)
    if deg <= 0.0:
        return dataset
    spread = _np.deg2rad(deg)
    if bool(getattr(args, "env_shift_control", False)):
        dataset.env_shift_theta = 0.0
        dataset.env_shift_range = spread
    else:
        if n_clients > 1:
            frac = client_idx / (n_clients - 1) - 0.5
        else:
            frac = 0.0
        dataset.env_shift_theta = float(spread * frac)
        dataset.env_shift_range = 0.0
    return dataset


def _refresh_global_anchor_tables(args, clients, frozen):
    """Integrate the frozen global's flow over every client window.

    Stochastic z0, deliberately: the global carries both modes, and z0 is what
    selects between them at generation time -- a deterministic z0 would feed
    every sample the same winding and starve the method of the coverage it is
    for. Tables are float32 (n_windows, H, action_dim), clipped to a sane range
    so an early ill-trained global cannot inject infinities.
    """
    frozen.eval()
    H = int(frozen.pred_horizon.item())
    for client in clients:
        ds = client["loader"].dataset
        _old = ds.transform_datum_fn
        ds.transform_datum_fn = None
        try:
            dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                             num_workers=0)
            rows = []
            with torch.no_grad():
                for b in dl:
                    obs = b["obs"].to(args.device).float()
                    so = b.get("state_obs")
                    so = so.to(args.device).float() if so is not None else None
                    cond = frozen.ObservationCondition(obs, state_obs=so)
                    B = obs.shape[0]
                    a = b["action"][:, :1].to(args.device).float()
                    a = a + float(frozen.σ0.item()) * torch.randn_like(a)
                    z = torch.randn_like(a)
                    traj = [a.squeeze(1)]
                    for j in range(H - 1):
                        t = torch.full((B,), j / (H - 1), device=args.device)
                        x = torch.cat((a, z), dim=1)
                        v = frozen.velocity_net(sample=x, timestep=t,
                                                global_cond=cond)
                        dt = 1.0 / (H - 1)
                        a = a + v[:, :1] * dt
                        z = z + v[:, 1:] * dt
                        traj.append(a.squeeze(1))
                    rows.append(torch.stack(traj, 1).clamp(-3, 3).cpu())
            tbl = torch.cat(rows).numpy().astype("float32")
        finally:
            ds.transform_datum_fn = _old
        client["policy"].global_anchor_table = tbl
        client["policy"].global_anchor_frac = float(args.global_anchor_frac)
        print("[gab] client %d: table %s refreshed" % (client["idx"], tbl.shape),
              flush=True)


def _apply_anchor_broaden(policy, args):
    """Carried on the policy so the datum transform can read it."""
    policy.anchor_broaden = float(getattr(args, "anchor_broaden", 0.0) or 0.0)
    policy.anchor_time_jitter = float(
        getattr(args, "anchor_time_jitter", 0.0) or 0.0)
    return policy


def _maybe_flare(policy, args):
    """Wrap the velocity net when FLARE is on.

    Applied at every build site: the clients, the global model and the EMA
    shadows all have to share a class, or the state dicts stop loading into
    each other.
    """
    if int(getattr(args, "flare_experts", 0) or 0) <= 0:
        return _apply_anchor_broaden(policy, args)
    if policy.velocity_net.__class__.__name__ == "FLAREVelocityNet":
        return policy
    from scripts.pusht_sfp.flare_pusht import wrap_policy
    cd = int(getattr(args, "flare_cond_dim", 0) or 0)
    policy = _apply_anchor_broaden(policy, args)
    return wrap_policy(policy, n_experts=int(args.flare_experts),
                       t_split=float(args.flare_t_split),
                       confidence_gate=bool(getattr(args, "flare_confidence_gate", False)),
                       router=str(getattr(args, "flare_router", "gate")),
                       cond_dim=(cd or None),
                       device=getattr(policy, "device", None))


def load_initial_state(policy, initial_state):
    policy.load_state_dict(deepcopy(initial_state))
    return policy


def client_keys(split):
    keys = []
    for key, value in split.items():
        if key.startswith("client_") and isinstance(value, list):
            suffix = key.split("_", 1)[1]
            if suffix.isdigit():
                keys.append(key)
    return sorted(keys, key=lambda k: int(k.split("_", 1)[1]))


def build_subset_dataset(args, policy, episode_indices, stats_episode_indices):
    # Episode filtering happens while the PushT dataset builds sample indices, so
    # pass the split through train_sfp's existing JSON path before construction.
    tmp_split = args.out_dir / f"_tmp_episode_indices_seed{args.seed}.json"
    tmp_split.parent.mkdir(parents=True, exist_ok=True)
    tmp_split.write_text(json.dumps({
        "train": [int(x) for x in episode_indices],
        "stats": [int(x) for x in stats_episode_indices],
    }))
    old_json = args.episode_indices_json
    old_key = args.episode_indices_key
    old_stats_key = args.stats_episode_indices_key
    try:
        args.episode_indices_json = tmp_split
        args.episode_indices_key = "train"
        args.stats_episode_indices_key = "stats"
        return build_dataset(args, policy)
    finally:
        args.episode_indices_json = old_json
        args.episode_indices_key = old_key
        args.stats_episode_indices_key = old_stats_key


def make_loader(args, dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        # GAB refresh replaces the parent's NumPy table. Recreate workers on
        # each iterator so they see the latest table after a round refresh.
        persistent_workers=(args.num_workers > 0 and
                            float(getattr(args, "global_anchor_frac", 0.0) or 0.0) <= 0.0),
    )


def make_endpoint_loader(args, dataset, client_idx):
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) * 1000 + int(client_idx))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        generator=generator,
    )


@contextmanager
def isolated_probe_rng(seed):
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def endpoint_subbatch(batch, tau_min):
    times = batch.get("t")
    if times is None or not torch.is_tensor(times):
        raise ValueError("endpoint projection requires tensor batch['t']")
    mask = times.reshape(-1) > float(tau_min)
    batch_size = int(mask.numel())
    count = int(mask.sum().item())
    if count == 0:
        return None, {"endpoint_count": 0, "endpoint_batch_size": batch_size}
    filtered = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            filtered[key] = value[mask]
        else:
            filtered[key] = value
    selected_times = times.reshape(-1)[mask].float()
    return filtered, {
        "endpoint_count": count,
        "endpoint_batch_size": batch_size,
        "endpoint_t_min": float(selected_times.min().item()),
        "endpoint_t_mean": float(selected_times.mean().item()),
        "endpoint_t_max": float(selected_times.max().item()),
    }


def next_endpoint_subbatch(client, tau_min, max_attempts=4):
    for _ in range(max_attempts):
        try:
            batch = next(client["endpoint_iterator"])
        except StopIteration:
            client["endpoint_iterator"] = iter(client["endpoint_loader"])
            batch = next(client["endpoint_iterator"])
        filtered, info = endpoint_subbatch(batch, tau_min)
        if filtered is not None:
            return filtered, info
    raise RuntimeError(
        f"no endpoint samples with t>{tau_min} after {max_attempts} batches"
    )


def project_client_momentum_from_endpoint(client, args, round_idx):
    probe_seed = int(args.seed) * 1_000_000 + int(round_idx) * 100 + int(client["idx"])
    optimizer = client["optimizer"]
    policy = client["policy"]
    with isolated_probe_rng(probe_seed):
        batch, batch_info = next_endpoint_subbatch(client, args.endpoint_tau_min)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        endpoint_loss = policy.Loss(batch)
        endpoint_loss.backward()
        projection_info = project_adam_first_moment_against_gradients(
            optimizer,
            policy.parameters(),
            epsilon=args.endpoint_projection_epsilon,
        )
        optimizer.zero_grad(set_to_none=True)
    return {
        **batch_info,
        **projection_info,
        "endpoint_loss": float(endpoint_loss.item()),
        "endpoint_tau_min": float(args.endpoint_tau_min),
        "probe_seed": probe_seed,
    }


def _optimizer_buffer_norm(optimizer, named_parameters, buffer_name):
    total_sq = 0.0
    count = 0
    for _name, parameter in named_parameters:
        value = optimizer.state.get(parameter, {}).get(buffer_name)
        if value is None:
            continue
        total_sq += float(torch.sum(value.detach().float() ** 2).item())
        count += 1
    return math.sqrt(total_sq), count


def retain_head_first_moment(optimizer, head_parameters, retention):
    if not 0.0 <= retention <= 1.0:
        raise ValueError("head first-moment retention must be in [0, 1]")
    changed_tensors = 0
    for _name, parameter in head_parameters:
        momentum = optimizer.state.get(parameter, {}).get("exp_avg")
        if momentum is not None:
            momentum.mul_(retention)
            changed_tensors += 1
    return changed_tensors


def head_selective_reset_from_endpoint(client, args, round_idx, retention=0.0):
    """Attenuate conflicting flow-head exp_avg; preserve vision exp_avg and all exp_avg_sq."""
    probe_seed = int(args.seed) * 1_000_000 + int(round_idx) * 100 + int(client["idx"])
    optimizer = client["optimizer"]
    policy = client["policy"]
    named_parameters = list(policy.named_parameters())
    head_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("velocity_net.")
    ]
    vision_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("obs_encoder.")
    ]
    if not head_parameters or not vision_parameters:
        raise RuntimeError(
            f"parameter partition failed: head={len(head_parameters)} vision={len(vision_parameters)}"
        )

    with isolated_probe_rng(probe_seed):
        batch, batch_info = next_endpoint_subbatch(client, args.endpoint_tau_min)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        endpoint_loss = policy.Loss(batch)
        endpoint_loss.backward()

        dot = 0.0
        momentum_sq = 0.0
        gradient_sq = 0.0
        comparable_tensors = 0
        for _name, parameter in head_parameters:
            momentum = optimizer.state.get(parameter, {}).get("exp_avg")
            gradient = parameter.grad
            if momentum is None or gradient is None:
                continue
            momentum_f = momentum.detach().float()
            gradient_f = gradient.detach().float()
            dot += float(torch.sum(momentum_f * gradient_f).item())
            momentum_sq += float(torch.sum(momentum_f ** 2).item())
            gradient_sq += float(torch.sum(gradient_f ** 2).item())
            comparable_tensors += 1

        head_m_before, head_m_tensors = _optimizer_buffer_norm(
            optimizer, head_parameters, "exp_avg"
        )
        vision_m_before, vision_m_tensors = _optimizer_buffer_norm(
            optimizer, vision_parameters, "exp_avg"
        )
        all_v_before, all_v_tensors = _optimizer_buffer_norm(
            optimizer, named_parameters, "exp_avg_sq"
        )

        cosine = None
        conflict = False
        if comparable_tensors > 0 and momentum_sq > 0.0 and gradient_sq > 0.0:
            cosine = dot / math.sqrt(momentum_sq * gradient_sq)
            conflict = cosine < 0.0
        reset_tensors = 0
        if conflict:
            reset_tensors = retain_head_first_moment(
                optimizer, head_parameters, retention
            )

        head_m_after, _ = _optimizer_buffer_norm(optimizer, head_parameters, "exp_avg")
        vision_m_after, _ = _optimizer_buffer_norm(optimizer, vision_parameters, "exp_avg")
        all_v_after, _ = _optimizer_buffer_norm(optimizer, named_parameters, "exp_avg_sq")
        optimizer.zero_grad(set_to_none=True)

    tolerance = 1e-12
    if abs(vision_m_after - vision_m_before) > tolerance:
        raise RuntimeError("head-selective audit changed vision exp_avg")
    if abs(all_v_after - all_v_before) > tolerance:
        raise RuntimeError("head-selective audit changed exp_avg_sq")
    return {
        **batch_info,
        "endpoint_loss": float(endpoint_loss.item()),
        "endpoint_tau_min": float(args.endpoint_tau_min),
        "probe_seed": probe_seed,
        "head_old_momentum_gradient_cosine": cosine,
        "head_conflict": conflict,
        "head_gate_active": True,
        "head_gate_start_round": 1,
        "head_conflict_retention": float(retention),
        "head_reset_tensors": reset_tensors,
        "head_comparable_tensors": comparable_tensors,
        "head_exp_avg_norm_before": head_m_before,
        "head_exp_avg_norm_after": head_m_after,
        "head_exp_avg_tensors": head_m_tensors,
        "vision_exp_avg_norm_before": vision_m_before,
        "vision_exp_avg_norm_after": vision_m_after,
        "vision_exp_avg_tensors": vision_m_tensors,
        "all_exp_avg_sq_norm_before": all_v_before,
        "all_exp_avg_sq_norm_after": all_v_after,
        "all_exp_avg_sq_tensors": all_v_tensors,
    }


def continuous_lr(args, global_step, total_steps):
    step = max(global_step, 1)
    if step <= args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def ema_update(ema_policy, policy, decay):
    """Exactly one EMA update per batch (never one per FTP bin)."""
    ep = dict(ema_policy.named_parameters())
    for n, p in policy.named_parameters():
        if n in ep:
            ep[n].mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    eb = dict(ema_policy.named_buffers())
    for n, b in policy.named_buffers():
        if n in eb:
            eb[n].copy_(b)


def ftp_step(policy, nbatch, optimizer, ftp):
    """FedSFP-FTP: flow-time-conditioned preconditioning.

    Split the batch by flow-time t into K bins; each bin runs its own
    forward/backward (total compute unchanged: sum n_k = B) and owns an
    independent Adam state (m_k, v_k). One parameter update per batch:
        theta <- theta * (1 - lr*wd) - lr * sum_k (n_k/B) * m_hat_k / (sqrt(v_hat_k)+eps)
    so different flow-time regimes (transport / trajectory / endpoint) never
    fight over one shared second-moment scale.
    """
    K = int(ftp["bins"])
    states = ftp["states"]
    lr = optimizer.param_groups[0]["lr"]
    wd = float(ftp["weight_decay"])
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = nbatch["t"].view(-1)
    B = int(t.shape[0])
    bin_idx = torch.clamp((t.float() * K).long(), 0, K - 1)
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    upd = {}
    total_loss = 0.0
    for k in range(K):
        idx = (bin_idx == k).nonzero(as_tuple=True)[0]
        nk = int(idx.numel())
        if nk == 0:
            continue
        sub = {key: (val[idx] if torch.is_tensor(val) and val.shape[0] == B else val)
               for key, val in nbatch.items()}
        policy.zero_grad(set_to_none=True)
        loss_k = policy.Loss(sub)
        loss_k.backward()
        w = nk / B
        total_loss += w * float(loss_k.item())
        st_k = states[k]
        for n, p in named:
            if p.grad is None:
                continue
            st = st_k.get(n)
            if st is None:
                st = {"step": 0,
                      "exp_avg": torch.zeros_like(p),
                      "exp_avg_sq": torch.zeros_like(p)}
                st_k[n] = st
            st["step"] += 1
            g = p.grad
            st["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
            st["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bc1 = 1 - beta1 ** st["step"]
            bc2 = 1 - beta2 ** st["step"]
            denom = (st["exp_avg_sq"] / bc2).sqrt_().add_(eps)
            direction = (st["exp_avg"] / bc1).div_(denom)
            if n in upd:
                upd[n].add_(direction, alpha=w)
            else:
                upd[n] = direction.mul_(w)
    with torch.no_grad():
        for n, p in named:
            if n in upd:
                if wd:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(upd[n], alpha=-lr)
    policy.zero_grad(set_to_none=True)
    return total_loss


MODULE_GROUPS = [
    ("all", None),
    ("obs_encoder", ("obs_encoder.",)),
    ("trunk_down_mid", ("velocity_net.down_modules.", "velocity_net.mid_modules.")),
    ("up_modules", ("velocity_net.up_modules.",)),
    ("up_modules_1", ("velocity_net.up_modules.1",)),
    ("final_conv", ("velocity_net.final_conv",)),
]


def pair_conflict_stats(acc_i, acc_j, prefixes, w_i=0.5, w_j=0.5):
    """Conflict statistics between two clients' per-bin update vectors.

    Returns (D, C, R, norm_i, norm_j) where
      D = 1 - cos(d_i, d_j)                 > 1 means anti-correlated
      C = 1 - |w_i d_i + w_j d_j| / (w_i|d_i| + w_j|d_j|)   energy lost to averaging
      R = |d_j| / |d_i|                     update-magnitude asymmetry
    Norms are accumulated per tensor so nothing large is materialised.
    """
    names = [n for n in acc_i if n in acc_j
             and (prefixes is None or n.startswith(tuple(prefixes)))]
    if not names:
        return None
    dot = ni2 = nj2 = merged2 = 0.0
    for n in names:
        a = acc_i[n].float()
        b = acc_j[n].float()
        dot += float((a * b).sum())
        ni2 += float((a * a).sum())
        nj2 += float((b * b).sum())
        m = w_i * a + w_j * b
        merged2 += float((m * m).sum())
    ni, nj, merged = ni2 ** 0.5, nj2 ** 0.5, merged2 ** 0.5
    cos = dot / (ni * nj + 1e-12)
    C = 1.0 - merged / (w_i * ni + w_j * nj + 1e-12)
    return (1.0 - cos, C, nj / (ni + 1e-12), ni, nj)


def fca_step(policy, nbatch, optimizer, fca):
    """FedSFP-FCA local step: produce per-flow-bin parameter deltas.

    Splits the batch by flow-time t into K bins and takes a per-bin Adam
    micro-step (round-local Adam state, reset each round). Each bin's applied
    update is weighted by its sample share n_k/B so the total per-batch update
    magnitude matches a standard step, and is accumulated into a per-bin delta
    buffer acc[k]. By construction sum_k acc[k] == Delta_i exactly, so the
    server can recombine bins across clients (flow-conditioned aggregation)
    while the client-size-weighted recombination reduces to plain FedAvg.
    """
    K = int(fca["bins"])
    states = fca["states"]
    acc = fca["acc"]
    counts = fca["counts"]
    corr = fca.get("corr")  # FCV: per-bin control-variate correction (c_b - c_ib)
    lr = optimizer.param_groups[0]["lr"]
    wd = float(fca["weight_decay"])
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = nbatch["t"].view(-1)
    B = int(t.shape[0])
    bin_idx = torch.clamp((t.float() * K).long(), 0, K - 1)
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    upd = {}
    total_loss = 0.0
    for k in range(K):
        idx = (bin_idx == k).nonzero(as_tuple=True)[0]
        nk = int(idx.numel())
        if nk == 0:
            continue
        counts[k] += nk
        sub = {key: (val[idx] if torch.is_tensor(val) and val.shape[0] == B else val)
               for key, val in nbatch.items()}
        policy.zero_grad(set_to_none=True)
        loss_k = policy.Loss(sub)
        loss_k.backward()
        w = nk / B
        total_loss += w * float(loss_k.item())
        stk = states[k]
        ak = acc[k]
        ck = corr[k] if corr is not None else None
        gsum = fca.get("gsum")
        if gsum is not None:
            fca["gw"][k] += w
        for n, p in named:
            if p.grad is None:
                continue
            if gsum is not None:
                # gradient-unit control variates: weighted mean of RAW bin grads
                # (accumulated BEFORE correction is applied)
                if n in gsum[k]:
                    gsum[k][n].add_(p.grad.detach(), alpha=w)
                else:
                    gsum[k][n] = p.grad.detach().clone().mul_(w)
            if ck is not None and n in ck:
                # SCAFFOLD-style drift correction, flow-stage matched
                p.grad.add_(ck[n])
            st = stk.get(n)
            if st is None:
                st = {"m_step": 0, "v_step": 0,
                      "exp_avg": torch.zeros_like(p),
                      "exp_avg_sq": torch.zeros_like(p)}
                stk[n] = st
            st["m_step"] += 1
            st["v_step"] += 1
            g = p.grad
            st["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
            st["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bc1 = 1 - beta1 ** st["m_step"]
            bc2 = 1 - beta2 ** st["v_step"]
            denom = (st["exp_avg_sq"] / bc2).sqrt_().add_(eps)
            direction = (st["exp_avg"] / bc1).div_(denom).mul_(w)
            delta_b = direction.mul(-lr)                       # gradient-driven part
            if wd:
                delta_b = delta_b.add_(p.detach(), alpha=-lr * wd * w)  # decoupled wd share
            if n in upd:
                upd[n].add_(delta_b)
            else:
                upd[n] = delta_b.clone()
            if n in ak:
                ak[n].add_(delta_b)
            else:
                ak[n] = delta_b.clone()
    with torch.no_grad():
        for n, p in named:
            if n in upd:
                p.data.add_(upd[n])
    policy.zero_grad(set_to_none=True)
    return total_loss


def train_one_round(policy, loader, optimizer, scheduler, args, *, lr_mode, global_step, total_steps,
                    fh=None, max_steps=None, ema_policy=None, ema_decay=0.0,
                    rg_frozen=None):
    losses = []
    policy.train()
    steps_round = max(args.local_steps_E * len(loader), 1)
    if max_steps:
        # strict mode-fair budget: every client takes the same number of
        # optimizer steps per round regardless of local dataset size
        steps_round = min(steps_round, int(max_steps))
    tail = max(1, int(0.1 * steps_round))
    k_in_round = 0
    lr_seen = []
    fh_prev_grads = None
    twod_norm = 1.0
    if lr_mode == "twodlrd" and fh is not None and fh.get("area_matched"):
        a = fh.get("alpha", 1.0)
        twod_norm = float(np.mean([a ** e for e in range(args.local_steps_E)]))
    for _local_epoch in range(args.local_steps_E):
        if lr_mode == "twodlrd" and fh is not None:
            # 2D-LRD (IJCNN'21) epoch-mapped: lr = eta0 * alpha^epoch_idx;
            # area-matched variant divides by mean(alpha^e) over the round
            set_lr(optimizer, args.lr * (fh.get("alpha", 1.0) ** _local_epoch) / max(twod_norm, 1e-8))
        elif lr_mode == "fedhyper_sl" and fh is not None:
            set_lr(optimizer, args.lr * fh.get("sl_beta", 1.0))
        for nbatch in loader:
            if max_steps and k_in_round >= int(max_steps):
                break
            if lr_mode == "continuous_manual":
                set_lr(optimizer, continuous_lr(args, global_step + 1, total_steps))
            elif lr_mode == "const_manual":
                step = global_step + 1
                lr = (args.lr * step / max(args.warmup_steps, 1)
                      if step <= args.warmup_steps else args.lr)
                set_lr(optimizer, lr)
            elif lr_mode in ("plateau_manual", "drift_manual"):
                step = global_step + 1
                scale = (fh or {}).get("lr_scale", 1.0)
                lr = (args.lr * step / max(args.warmup_steps, 1)
                      if step <= args.warmup_steps else args.lr * scale)
                set_lr(optimizer, lr)
            elif lr_mode in ("shape_linear", "shape_step", "shape_const_tail"):
                step = global_step + 1
                if step <= args.warmup_steps:
                    lr = args.lr * step / max(args.warmup_steps, 1)
                else:
                    p = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
                    if lr_mode == "shape_linear":
                        lr = args.lr * max(1.0 - p, 0.0)
                    elif lr_mode == "shape_step":
                        lr = args.lr * (1.0 if p < 0.5 else (0.1 if p < 0.75 else 0.01))
                    else:  # const_tail: flat envelope x area-matched tail
                        h = (1.0 if k_in_round < steps_round - tail
                             else (steps_round - k_in_round) / tail)
                        lr = args.lr * h / 0.95
                set_lr(optimizer, lr)
            elif lr_mode == "tailanneal_manual":
                # area-matched terminal cooling: global cosine envelope x
                # per-round tail shape (h=1 until 90%, linear->0), /mean(h)=0.95
                genv = continuous_lr(args, global_step + 1, total_steps)
                h = (1.0 if k_in_round < steps_round - tail
                     else (steps_round - k_in_round) / tail)
                set_lr(optimizer, genv * h / 0.95)
            if fh is not None and fh.get("ftp") is not None:
                loss_val = ftp_step(policy, nbatch, optimizer, fh["ftp"])
                if ema_policy is not None:
                    ema_update(ema_policy, policy, ema_decay)
                lr_seen.append(optimizer.param_groups[0]["lr"])
                global_step += 1
                k_in_round += 1
                losses.append(loss_val)
                continue
            if fh is not None and fh.get("fca") is not None:
                loss_val = fca_step(policy, nbatch, optimizer, fh["fca"])
                lr_seen.append(optimizer.param_groups[0]["lr"])
                global_step += 1
                k_in_round += 1
                losses.append(loss_val)
                continue
            _osc = float(getattr(args, "offsupport_consistency", 0.0) or 0.0)
            _rg = float(getattr(args, "residual_gate", 0.0) or 0.0)
            _kw = {}
            if rg_frozen is not None and _osc > 0:
                _kw = dict(teacher=rg_frozen, osc=_osc,
                           osc_d0=float(getattr(args, "osc_d0", 1.5)),
                           osc_tau=float(getattr(args, "osc_tau", 0.5)))
            if rg_frozen is not None and _rg > 0:
                with torch.no_grad():
                    _e = rg_frozen.Loss(nbatch, reduction="none")
                _w = 1.0 + _rg * (_e / (_e.mean() + 1e-8) - 1.0)
                loss = policy.Loss(nbatch, sample_weight=_w.clamp(min=0.0), **_kw)
            else:
                loss = policy.Loss(nbatch, **_kw)
            if int(getattr(args, "flare_experts", 0) or 0) > 0:
                loss = loss + float(args.flare_gate_weight) * \
                    policy.velocity_net.gate_loss()
            loss.backward()
            if lr_mode == "fedhyper_cl":
                # FedHyper-CL (arXiv:2310.03156 Eq.20): beta += <g_k, g_{k-1}>
                # + (1/K)<g_k, Delta_prev>; clip to [1/gamma, gamma], gamma=10.
                params = [p for p in policy.parameters() if p.grad is not None]
                x_term = 0.0
                if fh_prev_grads is not None:
                    x_term = float(sum((p.grad * gp).sum() for p, gp in zip(params, fh_prev_grads)))
                y_term = 0.0
                if fh.get("delta") is not None:
                    dmap = fh["delta"]
                    y_term = float(sum((p.grad * dmap[n]).sum()
                                       for n, p in policy.named_parameters()
                                       if p.grad is not None and n in dmap))
                if fh.get("norm"):
                    # per-batch sign step DYNAMICALLY calibrated: one round drifts
                    # at most 0.05 (matches per-round SL/G-norm tempo). steps_round
                    # ~1260 for Push-T -> ~4e-5/batch (0.002 still saturated).
                    step = 0.05 / max(steps_round, 1)
                    upd = step * (1.0 if (x_term + y_term / steps_round) > 0 else -1.0)
                else:
                    upd = x_term + y_term / steps_round
                fh["beta"] = min(max(fh["beta"] + upd, 0.1), 10.0)
                fh_prev_grads = [p.grad.detach().clone() for p in params]
                set_lr(optimizer, args.lr * fh["beta"])
            if fh is not None and fh.get("scaffold_corr") is not None:
                # SCAFFOLD drift correction: grad += (c_global - c_i)
                corr = fh["scaffold_corr"]
                for n, p in policy.named_parameters():
                    if p.grad is not None and n in corr:
                        p.grad.add_(corr[n])
            if fh is not None and fh.get("prox_mu"):
                # FedProx: grad += mu * (w - w_global)
                mu = fh["prox_mu"]; ref = fh["prox_ref"]
                for n, p in policy.named_parameters():
                    if p.grad is not None and n in ref:
                        p.grad.add_(mu * (p.detach() - ref[n]))
            lr_seen.append(optimizer.param_groups[0]["lr"])
            loss = loss  # noqa
            optimizer.step()
            optimizer.zero_grad()
            if ema_policy is not None:
                ema_update(ema_policy, policy, ema_decay)
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            k_in_round += 1
            losses.append(float(loss.item()))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "steps": len(losses),
        "global_step": global_step,
        "lr": optimizer.param_groups[0]["lr"],
        "lr_min": float(min(lr_seen)) if lr_seen else float("nan"),
        "lr_max": float(max(lr_seen)) if lr_seen else float("nan"),
    }


def save_metadata(path, args, split, client_info, initial_path):
    payload = {
        "model_type": args.model_type,
        "imitation_mode": args.imitation_mode,
        "obs_mode": args.obs_mode,
        "seed": args.seed,
        "rounds": args.rounds,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "sigma0": args.sigma0,
        "sigma1": args.sigma1,
        "use_stabilized_objective": args.use_stabilized_objective,
        "stabilization_k": args.stabilization_k,
        "use_time_decayed_noise": args.use_time_decayed_noise,
        "pred_horizon": args.pred_horizon,
        "obs_horizon": args.obs_horizon,
        "action_horizon": args.action_horizon,
        "image_cond_dim": args.image_cond_dim,
        "image_encoder": args.image_encoder,
        "image_agent_pos_cond": args.image_agent_pos_cond,
        "image_key": args.image_key,
        "image_random_crop_size": args.image_random_crop_size,
        "endpoint_tau_min": args.endpoint_tau_min,
        "endpoint_projection_epsilon": args.endpoint_projection_epsilon,
        "split": split,
        "client_info": client_info,
        "initial_global": str(initial_path),
        "bridge_controller_frozen_check": {
            "controller": "PushT analytic target-state-to-action bridge in evaluate.Rollout",
            "kp": 500,
            "kv": 20,
            "shared_across_arms": True,
            "same_code_path_for_raw_and_persistent": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def moment_retentions(cfg, global_policy, clients):
    mode = cfg.get("moment_mode")
    if mode == "loo":
        global_parameters = dict(global_policy.named_parameters())
        local_parameters = [dict(client["policy"].named_parameters()) for client in clients]
        alignments = leave_one_out_cosines(global_parameters, local_parameters, [client["weight"] for client in clients])
        return alignments, [moment_retention(alignment) for alignment in alignments]
    if mode == "fixed_half":
        return [None] * len(clients), [0.5] * len(clients)
    if cfg.get("reset_first_moment", False):
        return [None] * len(clients), [0.0] * len(clients)
    return [None] * len(clients), [None] * len(clients)

def run_variant(args, split, initial_state, initial_path, out_dir, variant):
    cfg = METHOD_CONFIGS[variant]
    keep_optimizer = cfg["keep_optimizer"]
    keep_scheduler = cfg["keep_scheduler"]
    reset_first_moment = cfg.get("reset_first_moment", False)
    endpoint_projection = cfg.get("endpoint_projection", False)
    head_selective_reset = cfg.get("head_selective_reset", False)
    head_gate_start_round = int(cfg.get("head_gate_start_round", 1))
    server_momentum = float(cfg.get("server_momentum", 0.0))
    server_momentum_dampened = bool(cfg.get("server_momentum_dampened", False))
    server_momentum_decay_end = cfg.get("server_momentum_decay_end")
    server_momentum_scope = cfg.get("server_momentum_scope", "all")
    server_momentum_buffer = None
    head_conflict_retention = float(cfg.get("head_conflict_retention", 0.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = out_dir / f"round_{args.rounds:04d}.pth"
    latest = out_dir / "latest.pth"
    if final_ckpt.exists() and latest.exists() and not args.force and not getattr(args, "force_train", False):
        return final_ckpt

    # print what actually arrived, not what was typed on the command line

    print("[knobs] anchor_broaden=%s residual_gate=%s flare_experts=%s "

          "flare_t_split=%s confidence_gate=%s offsupport=%s d0=%s tau=%s "
          "jitter=%s participation=%s envshift=%s ctrl=%s gabfrac=%s"

          % (getattr(args, "anchor_broaden", "ABSENT"),

             getattr(args, "residual_gate", "ABSENT"),

             getattr(args, "flare_experts", "ABSENT"),

             getattr(args, "flare_t_split", "ABSENT"),

             getattr(args, "flare_confidence_gate", "ABSENT"),
         getattr(args, "offsupport_consistency", "ABSENT"),
         getattr(args, "osc_d0", "ABSENT"),
         getattr(args, "osc_tau", "ABSENT"),
         getattr(args, "anchor_time_jitter", "ABSENT"),
         getattr(args, "client_fraction", "ABSENT"),
         getattr(args, "env_shift", "ABSENT"),
         getattr(args, "env_shift_control", "ABSENT"),
         getattr(args, "global_anchor_frac", "ABSENT")), flush=True)

    set_seed(args.seed)
    global_policy = _maybe_flare(load_initial_state(build_policy(args), initial_state), args)
    clients = []
    for client_idx, key in enumerate(client_keys(split)):
        policy = _maybe_flare(load_initial_state(build_policy(args), initial_state), args)
        dataset = build_subset_dataset(args, policy, split[key], split["union_200"])
        _apply_env_shift(dataset, args, client_idx, len(client_keys(split)))
        loader = make_loader(args, dataset)
        # canonical FTP3: a fixed number of updates per local epoch, so the
        # global LR clock is exactly rounds*epochs*steps (35,000) regardless of
        # how many batches a client's dataset happens to yield
        spe = int(getattr(args, "steps_per_epoch", 0) or 0)
        local_steps_per_round = (args.local_steps_E * spe if spe > 0
                                 else args.local_steps_E * len(loader))
        total_steps = args.rounds * local_steps_per_round
        client = {
            "idx": client_idx,
            "key": key,
            "policy": policy,
            "loader": loader,
            "weight": len(split[key]),
            "dataset_len": len(dataset),
            "num_batches": len(loader),
            "global_step": 0,
            "local_steps_per_round": local_steps_per_round,
            "total_steps": total_steps,
        }
        if endpoint_projection or head_selective_reset:
            client["endpoint_loader"] = make_endpoint_loader(args, dataset, client_idx)
            client["endpoint_iterator"] = iter(client["endpoint_loader"])
        client["optimizer"] = make_optimizer(args, policy) if keep_optimizer else None
        if float(getattr(args, "ema_decay", 0.0) or 0.0) > 0:
            client["ema_policy"] = _maybe_flare(load_initial_state(build_policy(args), initial_state), args)
        if cfg.get("ftp_bins"):
            # FTP: per-flow-time-bin Adam states, persistent across rounds
            # (client-local, like VP's v); populated lazily on first use.
            client["ftp_states"] = [dict() for _ in range(int(cfg["ftp_bins"]))]
        if keep_scheduler and keep_optimizer:
            client["scheduler"] = get_scheduler(
                name="cosine",
                optimizer=client["optimizer"],
                num_warmup_steps=args.warmup_steps,
                num_training_steps=max(total_steps, 1),
            )
            client["lr_mode"] = "scheduler_object"
        elif keep_scheduler:
            client["scheduler"] = None
            client["lr_mode"] = "continuous_manual"
        elif cfg.get("const_lr", False):
            client["scheduler"] = None
            client["lr_mode"] = "const_manual"
        elif cfg.get("tail_anneal", False):
            client["scheduler"] = None
            client["lr_mode"] = "tailanneal_manual"
        elif cfg.get("fedhyper_cl", False):
            client["scheduler"] = None
            client["lr_mode"] = "fedhyper_cl"
            client["fh_beta"] = 1.0
        elif cfg.get("fedhyper_sl", False):
            client["scheduler"] = None
            client["lr_mode"] = "fedhyper_sl"
        elif cfg.get("twod_lrd", False):
            client["scheduler"] = None
            client["lr_mode"] = "twodlrd"
        elif cfg.get("shape_lr"):
            client["scheduler"] = None
            client["lr_mode"] = "shape_" + cfg["shape_lr"]
        elif cfg.get("plateau_lr", False):
            client["scheduler"] = None
            client["lr_mode"] = "plateau_manual"
        elif cfg.get("drift_lr", False):
            client["scheduler"] = None
            client["lr_mode"] = "drift_manual"
        else:
            client["scheduler"] = None
            client["lr_mode"] = "reset_scheduler"
        clients.append(client)

    save_rounds = set(args.checkpoint_rounds)
    save_rounds.add(args.rounds)
    rows = []
    moment_rows = []
    conflict_rows = []
    # strict mode-fair budget: cap every client at the smallest client's
    # per-round step count so aggregation weight AND local compute are equal
    equal_step_budget = (min(c["local_steps_per_round"] for c in clients)
                         if cfg.get("equal_steps") else None)
    if int(getattr(args, "steps_per_epoch", 0) or 0) > 0:
        # fixed per-round update budget dominates any equal-steps capping
        equal_step_budget = clients[0]["local_steps_per_round"]
    # EMA shadow of the global weights, FedAvg'd alongside the raw weights
    global_ema_state = (deepcopy(clients[0]["ema_policy"].state_dict())
                        if clients and clients[0].get("ema_policy") is not None else None)
    started = time.time()
    save_metadata(
        out_dir / "metadata.json",
        args,
        split,
        [
            {
                "client": c["idx"],
                "key": c["key"],
                "weight": c["weight"],
                "dataset_len": c["dataset_len"],
                "num_batches": c["num_batches"],
                "lr_mode": c["lr_mode"],
            }
            for c in clients
        ],
        initial_path,
    )
    fh_delta_prev_dev = None
    sched_ctrl = {"sl_beta": 1.0, "d": 0, "last_adj": -99, "prev_delta": None}
    if cfg.get("fcc_bins"):
        # FCC broadcast channel: shared-memory q tensor attached to every client
        # policy BEFORE loader workers fork, so in-place server updates are
        # visible inside persistent dataloader workers (flow-time sampling).
        _Bn = int(cfg["fcc_bins"])
        _q = torch.full((_Bn,), 1.0 / _Bn, dtype=torch.float32)
        _q.share_memory_()
        sched_ctrl["fcc_q_tensor"] = _q
        for _cl in clients:
            _cl["policy"].flow_time_bin_weights = _q
    for round_idx in range(1, args.rounds + 1):
        print(f"[p47:image_action:{variant}:seed{args.seed}] round {round_idx}/{args.rounds}", flush=True)
        global_state = deepcopy(global_policy.state_dict())
        if global_ema_state is not None:
            for _c in clients:
                _c["ema_policy"].load_state_dict(global_ema_state)
        weighted_states = []
        _rg_frozen = None
        if (float(getattr(args, "residual_gate", 0.0) or 0.0) > 0
                or float(getattr(args, "offsupport_consistency", 0.0) or 0.0) > 0
                or float(getattr(args, "global_anchor_frac", 0.0) or 0.0) > 0):
            # the round's starting point, held fixed for the whole round
            # rather than chasing each client as it moves
            _rg_frozen = _maybe_flare(build_policy(args), args)
            _rg_frozen.load_state_dict(global_state)
            _rg_frozen.eval()
            for _q in _rg_frozen.parameters():
                _q.requires_grad_(False)
        _gaf = float(getattr(args, "global_anchor_frac", 0.0) or 0.0)
        if _gaf > 0 and _rg_frozen is not None and \
                (round_idx - 1) % max(int(getattr(args, "global_anchor_refresh", 5) or 5), 1) == 0:
            _refresh_global_anchor_tables(args, clients, _rg_frozen)
        round_client_losses = []
        projection_by_client = {}
        # Partial participation. Seeded from the round index so the same
        # configuration draws the same clients on a rerun, and so two arms
        # differing only in a method flag see identical participation.
        _frac = float(getattr(args, "client_fraction", 1.0) or 1.0)
        if _frac < 1.0:
            _k = max(1, int(round(_frac * len(clients))))
            _rng = random.Random(args.split_seed * 100003 + round_idx)
            active_clients = _rng.sample(list(clients), _k)
            print("[participation] round %d: %d/%d clients %s"
                  % (round_idx, len(active_clients), len(clients),
                     sorted(c["idx"] for c in active_clients)), flush=True)
        else:
            active_clients = clients
        for client in active_clients:
            client["policy"].load_state_dict(global_state)
            if int(getattr(args, "flare_experts", 0) or 0) > 0:
                # the split already knows which mode a client holds;
                # a hard assignment keeps its gradient on its own row
                client["policy"].velocity_net.set_expert(client["idx"])
            if cfg.get("mca_prefixes") and client.get("mca_own"):
                # mode-specific components never cross the mode boundary
                _sd = client["policy"].state_dict()
                for _k, _v in client["mca_own"].items():
                    _sd[_k] = _v.to(_sd[_k].device)
                client["policy"].load_state_dict(_sd)
            optimizer = client["optimizer"] if keep_optimizer else make_optimizer(args, client["policy"])
            if endpoint_projection:
                projection_by_client[client["idx"]] = project_client_momentum_from_endpoint(
                    client, args, round_idx
                )
            elif head_selective_reset:
                if round_idx >= head_gate_start_round:
                    projection_by_client[client["idx"]] = head_selective_reset_from_endpoint(
                        client, args, round_idx, retention=head_conflict_retention
                    )
                    projection_by_client[client["idx"]]["head_gate_start_round"] = head_gate_start_round
                else:
                    projection_by_client[client["idx"]] = {
                        "head_gate_active": False,
                        "head_gate_start_round": head_gate_start_round,
                        "head_conflict_retention": head_conflict_retention,
                    }
            scheduler = client["scheduler"]
            if client["lr_mode"] == "reset_scheduler":
                scheduler = make_reset_scheduler(optimizer, client["local_steps_per_round"])
                lr_mode = "scheduler_object"
            else:
                lr_mode = client["lr_mode"]
            fh_state = None
            if client.get("lr_mode") == "fedhyper_cl":
                fh_state = {"beta": client.get("fh_beta", 1.0), "delta": fh_delta_prev_dev,
                            "norm": cfg.get("norm_update", False)}
            elif client.get("lr_mode") == "fedhyper_sl":
                fh_state = {"sl_beta": sched_ctrl["sl_beta"]}
            elif client.get("lr_mode") == "twodlrd":
                fh_state = {"alpha": max(1.0 - 0.1 * sched_ctrl["d"], 0.05),
                            "area_matched": cfg.get("area_matched", False)}
            elif client.get("lr_mode") == "plateau_manual":
                fh_state = {"lr_scale": sched_ctrl.get("pl_scale", 1.0)}
            elif client.get("lr_mode") == "drift_manual":
                fh_state = {"lr_scale": sched_ctrl.get("drift_scale", {}).get(client["idx"], 1.0)}
            if cfg.get("ftp_bins"):
                fh_state = {"ftp": {"bins": int(cfg["ftp_bins"]),
                                    "states": client["ftp_states"],
                                    "weight_decay": args.weight_decay}}
            fca_ctx = None
            _bins_cfg = cfg.get("fca_bins") or cfg.get("fcv_bins") or cfg.get("fcc_bins")
            if _bins_cfg:
                _Bn = int(_bins_cfg)
                fca_ctx = {"bins": _Bn,
                           "states": [dict() for _ in range(_Bn)],   # round-local Adam (reset each round)
                           "acc": [dict() for _ in range(_Bn)],       # per-bin applied delta
                           "counts": [0] * _Bn,
                           "weight_decay": args.weight_decay}
                if cfg.get("fcv_bins"):
                    if cfg.get("fcv_grad_cv"):
                        fca_ctx["gsum"] = [dict() for _ in range(_Bn)]
                        fca_ctx["gw"] = [0.0] * _Bn
                    # FCV correction c_b - c_ib per bin (zeros round 1 -> corr None)
                    _c = sched_ctrl.get("fcv_c")
                    _ci = client.get("fcv_ci")
                    if _c is not None and _ci is not None and any(_c[b] for b in range(_Bn)):
                        _damp = float(cfg.get("fcv_damp", 1.0))
                        fca_ctx["corr"] = [
                            {n: (_damp * (_c[b][n] - _ci[b].get(n, torch.zeros_like(_c[b][n])))).to(args.device)
                             for n in _c[b]}
                            for b in range(_Bn)]
                fh_state = {"fca": fca_ctx}
            if cfg.get("prox_mu"):
                fh_state = {"prox_mu": cfg["prox_mu"],
                            "prox_ref": {n: global_state[n].to(args.device)
                                         for n in global_state
                                         if torch.is_floating_point(global_state[n])}}
            if cfg.get("scaffold"):
                if sched_ctrl.get("sc_c") is None:
                    sched_ctrl["sc_c"] = {n: torch.zeros_like(p, device="cpu")
                                          for n, p in client["policy"].named_parameters()}
                if client.get("sc_ci") is None:
                    client["sc_ci"] = {n: torch.zeros_like(p, device="cpu")
                                       for n, p in client["policy"].named_parameters()}
                corr = {n: (sched_ctrl["sc_c"][n] - client["sc_ci"][n]).to(args.device)
                        for n in sched_ctrl["sc_c"]}
                fh_state = {"scaffold_corr": corr}
            if cfg.get("zero_component") and optimizer is not None:
                comp = cfg["zero_component"]
                for _st in optimizer.state.values():
                    if comp in _st:
                        _st[comp].zero_()
                # warm-gated VP: also zero v while specialization memory is still
                # forming (round <= warm_persist), only persist v afterwards
                if cfg.get("warm_persist") and round_idx <= cfg["warm_persist"]:
                    for _st in optimizer.state.values():
                        if "exp_avg_sq" in _st:
                            _st["exp_avg_sq"].zero_()
                # SVR warm-only: persist v EARLY (build scale), REFRESH v late
                # (round >= gate) to avoid stale client geometry dominating
                if cfg.get("svr_refresh_after") and round_idx >= cfg["svr_refresh_after"]:
                    for _st in optimizer.state.values():
                        if "exp_avg_sq" in _st:
                            _st["exp_avg_sq"].zero_()
                # FTP: reset each bin's first moment at round start (VP-style);
                # per-bin second moments persist across rounds
                if cfg.get("ftp_bins") and client.get("ftp_states") is not None:
                    # canonical FTP3 round rule: m zeroed AND its bias-correction
                    # clock restarted; v and v_step carry over
                    for _stk in client["ftp_states"]:
                        for _st in _stk.values():
                            _st["exp_avg"].zero_()
                            _st["m_step"] = 0
            if cfg.get("fedadamw_style") and sched_ctrl.get("avg_v") is not None:
                # FedAdamW-style: fresh m (new optimizer) + inherit server-averaged v
                import torch as _torch
                for _n, _p in client["policy"].named_parameters():
                    if _n in sched_ctrl["avg_v"]:
                        optimizer.state[_p] = {
                            "step": _torch.tensor(float(client["global_step"])),
                            "exp_avg": _torch.zeros_like(_p),
                            "exp_avg_sq": sched_ctrl["avg_v"][_n].to(_p.device).clone()}
            info = train_one_round(
                client["policy"],
                client["loader"],
                optimizer,
                scheduler,
                args,
                lr_mode=lr_mode,
                global_step=client["global_step"],
                total_steps=client["total_steps"],
                fh=fh_state,
                max_steps=equal_step_budget,
                ema_policy=client.get("ema_policy"),
                ema_decay=float(getattr(args, "ema_decay", 0.0) or 0.0),
                rg_frozen=_rg_frozen,
            )
            if fh_state is not None and "beta" in fh_state:
                client["fh_beta"] = fh_state["beta"]
            client["global_step"] = info["global_step"]
            round_client_losses.append(info["loss"])
            if fca_ctx is not None:
                # move per-bin deltas to CPU (cross-client aggregation / D_b / c updates)
                _Bn = fca_ctx["bins"]
                client["fca_acc"] = [
                    {n: t.detach().cpu() for n, t in fca_ctx["acc"][b].items()}
                    for b in range(_Bn)]
                client["fca_counts"] = list(fca_ctx["counts"])
                if cfg.get("fcv_bins") and cfg.get("fcv_grad_cv"):
                    # gradient-unit CV: c_ib = weighted mean of raw bin-b grads
                    client["fcv_ci"] = [
                        {n: (g / max(fca_ctx["gw"][b], 1e-12)).detach().cpu()
                         for n, g in fca_ctx["gsum"][b].items()}
                        for b in range(_Bn)]
                elif cfg.get("fcv_bins"):
                    # per-bin SCAFFOLD option II: c_ib' = c_ib - c_b - acc_b/(K_b*lr_bar)
                    # (acc_b = applied per-bin delta = -(x - y_i)|_b;
                    #  K_b = effective weighted step count of bin b)
                    lr_bar = 0.5 * (info.get("lr_min", args.lr) + info.get("lr_max", args.lr))
                    if client.get("fcv_ci") is None:
                        client["fcv_ci"] = [dict() for _ in range(_Bn)]
                    if sched_ctrl.get("fcv_c") is None:
                        sched_ctrl["fcv_c"] = [dict() for _ in range(_Bn)]
                    dci_bins = []
                    for b in range(_Bn):
                        K_b = max(fca_ctx["counts"][b] / max(args.batch_size, 1), 1e-8)
                        dci = {}
                        for n, dt in client["fca_acc"][b].items():
                            ci_old = client["fcv_ci"][b].get(n)
                            c_g = sched_ctrl["fcv_c"][b].get(n)
                            if ci_old is None:
                                ci_old = torch.zeros_like(dt)
                            if c_g is None:
                                c_g = torch.zeros_like(dt)
                            nc = ci_old - c_g - dt / (K_b * max(lr_bar, 1e-12))
                            dci[n] = nc - ci_old
                            client["fcv_ci"][b][n] = nc
                        dci_bins.append(dci)
                    client["fcv_dci"] = dci_bins
            if cfg.get("ftp_bins") and client.get("ftp_states") is not None:
                # per-bin sum of exp_avg_sq: shows whether flow-time regimes
                # really carry different curvature scales (the FTP premise)
                _vb = []
                for _stk in client["ftp_states"]:
                    _tot = 0.0
                    for _st in _stk.values():
                        _tot += float(_st["exp_avg_sq"].sum().item())
                    _vb.append(_tot)
                client["ftp_v_bins"] = "|".join(f"{_x:.4e}" for _x in _vb)
            if cfg.get("scaffold"):
                # option II: c_i_new = c_i - c + (x - y_i)/(K*lr); persist Delta c_i
                K = max(info["steps"], 1)
                new_ci, dci = {}, {}
                for n, p in client["policy"].named_parameters():
                    x = global_state[n].detach().cpu()
                    yi = p.detach().cpu()
                    nc = client["sc_ci"][n] - sched_ctrl["sc_c"][n] + (x - yi) / (K * args.lr)
                    dci[n] = nc - client["sc_ci"][n]
                    new_ci[n] = nc
                client["sc_ci"] = new_ci
                client["sc_dci"] = dci
            if cfg.get("fedadamw_style"):
                vv = {}
                for _n, _p in client["policy"].named_parameters():
                    _st = optimizer.state.get(_p)
                    if _st is not None and "exp_avg_sq" in _st:
                        vv[_n] = _st["exp_avg_sq"].detach().cpu()
                sched_ctrl.setdefault("v_harvest", []).append(vv)
            if cfg.get("vsync"):
                vv = {}
                for _n, _p in client["policy"].named_parameters():
                    _st = optimizer.state.get(_p)
                    if _st is not None and "exp_avg_sq" in _st:
                        vv[_n] = _st["exp_avg_sq"].detach().cpu()
                sched_ctrl.setdefault("vsync_harvest", []).append(vv)
            if cfg.get("pgs_lambda") is not None:
                vv = {}
                for _n, _p in client["policy"].named_parameters():
                    _st = optimizer.state.get(_p)
                    if _st is not None and "exp_avg_sq" in _st:
                        vv[_n] = _st["exp_avg_sq"].detach().cpu()
                sched_ctrl.setdefault("pgs_harvest", []).append((client, vv))
            if cfg.get("fisher_agg"):
                vv = {}
                for _n, _p in client["policy"].named_parameters():
                    _st = optimizer.state.get(_p)
                    if _st is not None and "exp_avg_sq" in _st:
                        vv[_n] = _st["exp_avg_sq"].detach().cpu()
                sched_ctrl.setdefault("fisher_harvest", []).append(
                    (cpu_state_dict(client["policy"]), vv))
            # canonical FTP3 aggregates by each client's SAMPLE count, not its
            # episode count (67/67/66 episodes but 8033/7713/7743 windows here)
            agg_w = (1.0 if cfg.get("equal_weight")
                     else (client["dataset_len"] if getattr(args, "weight_by_samples", False)
                           else client["weight"]))
            weighted_states.append((cpu_state_dict(client["policy"]), agg_w))
            if cfg.get("mca_prefixes"):
                # MCA: retain this client's mode-specific components locally;
                # they are excluded from the cross-mode average below.
                pre = tuple(cfg["mca_prefixes"])
                client["mca_own"] = {k: v.clone()
                                     for k, v in weighted_states[-1][0].items()
                                     if k.startswith(pre)}
            rows.append({
                "round": round_idx,
                "variant": variant,
                "method": cfg["method"],
                "client": client["idx"],
                "client_size": client["weight"],
                "dataset_len": client["dataset_len"],
                "num_batches": client["num_batches"],
                "loss": info["loss"],
                "steps": info["steps"],
                "lr": info["lr"],
                "lr_min": info.get("lr_min"),
                "lr_max": info.get("lr_max"),
                "fh_beta": client.get("fh_beta"),
                "sl_beta": sched_ctrl["sl_beta"],
                "sched_d": sched_ctrl["d"],
                "eta_g": sched_ctrl.get("eta_g"),
                "client_global_step": client["global_step"],
                "keep_optimizer": keep_optimizer,
                "keep_scheduler": keep_scheduler,
                "reset_first_moment": reset_first_moment,
                "endpoint_projection": endpoint_projection,
                "head_selective_reset": head_selective_reset,
                "head_gate_start_round": head_gate_start_round,
                "head_conflict_retention": head_conflict_retention,
                "scheduler_mode": client["lr_mode"],
                "ftp_v_bins": client.get("ftp_v_bins"),
                "fcc_D_bins": sched_ctrl.get("fcc_D_str"),
                "fcc_q_bins": sched_ctrl.get("fcc_q_str"),
            })
        # over the clients that actually trained this round: leave-one-out
        # cosines are undefined for a client that produced no update, and the
        # loop below zips this against active_clients
        alignments, retentions = moment_retentions(cfg, global_policy, active_clients)
        for client, alignment, retention in zip(active_clients, alignments, retentions):
            if retention is not None:
                if retention == 0.0:
                    reset_adam_first_moment(client["optimizer"])
                elif retention < 1.0:
                    scale_adam_first_moment(client["optimizer"], retention)
            projection_info = projection_by_client.get(client["idx"], {})
            moment_rows.append({
                "round": round_idx,
                "variant": variant,
                "client": client["idx"],
                "alignment": alignment,
                "retention": retention,
                **projection_info,
            })
        if global_ema_state is not None:
            global_ema_state = average_state_dicts(
                [(cpu_state_dict(c["ema_policy"]), w)
                 for (c, (_s, w)) in zip(active_clients, weighted_states)])
        aggregated = average_state_dicts(weighted_states)
        if int(getattr(args, "flare_experts", 0) or 0) > 0:
            # only client i writes row i, so the mean returns
            # g + (w_i/W)*delta_i and shrinks each update by its share
            _W = float(sum(w for _s, w in weighted_states))
            for _k in aggregated:
                if ("expert_scale" not in _k and "expert_shift" not in _k
                        and "prototypes" not in _k):
                    continue
                _g = global_state[_k].detach().float().cpu()
                for _i, (_c, (_s, _w)) in enumerate(zip(active_clients, weighted_states)):
                    if _i >= aggregated[_k].shape[0]:
                        break
                    _sc = _W / max(float(_w), 1e-12)
                    aggregated[_k][_i] = _g[_i] + _sc * (aggregated[_k][_i] - _g[_i])
        if cfg.get("fca_bins"):
            # Flow-Conditioned Aggregation: aggregate client deltas WITHIN each
            # matched flow stage, then combine stages (alpha_b = 1).
            #   binshare  : w_{i,b} = n_{i,b} / sum_j n_{j,b}   (real FCA)
            #   clientsize: w_{i,b} = |D_i| / sum_j |D_j|       (== FedAvg; control)
            _Bn = int(cfg["fca_bins"])
            _wmode = cfg.get("fca_weight", "binshare")
            gcpu = {n: global_state[n].detach().cpu() for n in global_state}
            Nb = [sum(cl.get("fca_counts", [0] * _Bn)[b] for cl in clients)
                  for b in range(_Bn)]
            Wtot = sum(cl["weight"] for cl in clients)
            new_agg = {n: gcpu[n].clone() for n in gcpu}
            for b in range(_Bn):
                for cl in clients:
                    if _wmode == "binshare":
                        w = cl["fca_counts"][b] / max(Nb[b], 1)
                    else:
                        w = cl["weight"] / max(Wtot, 1)
                    if w == 0.0:
                        continue
                    for n, dt in cl["fca_acc"][b].items():
                        if n in new_agg and torch.is_floating_point(new_agg[n]):
                            new_agg[n] = new_agg[n] + w * dt
            aggregated = new_agg
        if cfg.get("fcv_bins") and cfg.get("fcv_grad_cv"):
            # gradient-unit CV server side: c_b = mean_i c_ib (exact SCAFFOLD c
            # under full participation)
            _Bn = int(cfg["fcv_bins"])
            if sched_ctrl.get("fcv_c") is None:
                sched_ctrl["fcv_c"] = [dict() for _ in range(_Bn)]
            for b in range(_Bn):
                names = set()
                for cl in clients:
                    if cl.get("fcv_ci"):
                        names.update(cl["fcv_ci"][b].keys())
                for n in names:
                    vals = [cl["fcv_ci"][b][n] for cl in clients
                            if cl.get("fcv_ci") and n in cl["fcv_ci"][b]]
                    sched_ctrl["fcv_c"][b][n] = sum(vals) / len(vals)
        elif cfg.get("fcv_bins"):
            # server control variates: c_b += mean_i dc_ib
            _Bn = int(cfg["fcv_bins"])
            for b in range(_Bn):
                for cl in clients:
                    dci = cl.get("fcv_dci")
                    if dci is None:
                        continue
                    for n, dv in dci[b].items():
                        cur = sched_ctrl["fcv_c"][b].get(n)
                        if cur is None:
                            cur = torch.zeros_like(dv)
                        sched_ctrl["fcv_c"][b][n] = cur + dv / max(len(clients), 1)
        if cfg.get("fcc_bins"):
            # FCC: cross-client per-flow-bin disagreement D_b from real per-bin
            # deltas (stride-subsampled cos), then bounded sampling weights q_b
            # broadcast for next round's flow-time sampling.
            _Bn = int(cfg["fcc_bins"])
            _alpha = float(cfg.get("fcc_alpha", 1.0))
            _stride = 16
            D = []
            for b in range(_Bn):
                vecs = []
                for cl in clients:
                    accb = cl.get("fca_acc", [None] * _Bn)[b]
                    if not accb:
                        continue
                    names = sorted(accb.keys())
                    vecs.append(torch.cat([accb[n].flatten()[::_stride] for n in names]))
                cosd = []
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        cosd.append(float(torch.nn.functional.cosine_similarity(
                            vecs[i], vecs[j], dim=0)))
                D.append(1.0 - (sum(cosd) / len(cosd)) if cosd else 0.0)
            # module-wise conflict diagnostics: full-model cosine is dominated by
            # the 99.55% shared trunk and cannot see head-level mode conflict,
            # so resolve D / cancellation / norm-ratio per module group per bin.
            for b in range(_Bn):
                for gname, gpre in MODULE_GROUPS:
                    for i in range(len(clients)):
                        for j in range(i + 1, len(clients)):
                            ai = clients[i].get("fca_acc", [None] * _Bn)[b]
                            aj = clients[j].get("fca_acc", [None] * _Bn)[b]
                            if not ai or not aj:
                                continue
                            st = pair_conflict_stats(ai, aj, gpre)
                            if st is None:
                                continue
                            conflict_rows.append({
                                "round": round_idx, "variant": variant,
                                "bin": b, "module": gname,
                                "client_i": clients[i]["idx"], "client_j": clients[j]["idx"],
                                "D_1_minus_cos": st[0], "cancellation_C": st[1],
                                "norm_ratio_j_over_i": st[2],
                                "norm_i": st[3], "norm_j": st[4],
                            })
            raw = [max(d, 1e-6) ** _alpha for d in D]
            q = [r / sum(raw) for r in raw]
            q = [min(max(x, 0.5 / _Bn), 2.0 / _Bn) for x in q]
            q = [x / sum(q) for x in q]
            if sched_ctrl.get("fcc_q_tensor") is not None:
                sched_ctrl["fcc_q_tensor"].copy_(torch.tensor(q, dtype=torch.float32))
            sched_ctrl["fcc_D_str"] = "|".join(f"{d:.4f}" for d in D)
            sched_ctrl["fcc_q_str"] = "|".join(f"{x:.4f}" for x in q)
        if cfg.get("ties_merge"):
            # TIES: per-param elect majority sign of client deltas, then average
            # only the deltas agreeing with elected sign (drop conflicting).
            gcpu = {n: global_state[n].detach().cpu() for n in global_state}
            states = [st for st, _ in weighted_states]
            for n in aggregated:
                if torch.is_floating_point(aggregated[n]) and n in gcpu:
                    d = torch.stack([st[n] - gcpu[n] for st in states])
                    elected = torch.sign(d.sum(0))
                    agree = (torch.sign(d) == elected.unsqueeze(0)).float()
                    merged = (d * agree).sum(0) / (agree.sum(0) + 1e-12)
                    aggregated[n] = gcpu[n] + merged
        if cfg.get("fisher_agg") and sched_ctrl.get("fisher_harvest"):
            # per-parameter Fisher(v)-weighted merge: theta = sum_i v_i theta_i / sum_i v_i
            # (byinit: lets each client dominate the params it specialized on)
            fh_list = sched_ctrl.pop("fisher_harvest")
            for n in aggregated:
                if torch.is_floating_point(aggregated[n]) and all(n in v for _, v in fh_list):
                    num = sum(v[n] * st[n] for st, v in fh_list)
                    den = sum(v[n] for _, v in fh_list) + 1e-12
                    aggregated[n] = num / den
        if any(c.get("lr_mode") in ("fedhyper_cl", "fedhyper_sl", "twodlrd") for c in clients):
            delta_cpu = {k: (aggregated[k] - global_state[k].cpu())
                         for k in aggregated if torch.is_floating_point(aggregated[k])}
            if any(c.get("lr_mode") == "fedhyper_cl" for c in clients):
                fh_delta_prev_dev = {k: v.to(args.device) for k, v in delta_cpu.items()}
            if sched_ctrl["prev_delta"] is not None:
                s = float(sum((delta_cpu[k] * sched_ctrl["prev_delta"][k]).sum()
                              for k in delta_cpu))
                # FedHyper-SL: beta += <Delta_t, Delta_{t-1}>, clip [0.1, 10]
                # (sign-normalized variant: beta += 0.05*sign(s))
                s_upd = (0.05 * (1.0 if s > 0 else -1.0)) if cfg.get("norm_update") else s
                sched_ctrl["sl_beta"] = min(max(sched_ctrl["sl_beta"] + s_upd, 0.1), 10.0)
                # 2D-LRD stationary-phase detection (observation window 3 rounds)
                if s < 0 and round_idx - sched_ctrl["last_adj"] >= 3:
                    sched_ctrl["d"] += 1
                    sched_ctrl["last_adj"] = round_idx
            sched_ctrl["prev_delta"] = delta_cpu
        if cfg.get("fedexp") or cfg.get("fedhyper_g"):
            keys_f = [k for k in aggregated if torch.is_floating_point(aggregated[k])]
            gprev = {k: global_state[k].cpu() for k in keys_f}
            dbar = {k: aggregated[k] - gprev[k] for k in keys_f}
            dbar_sq = float(sum((dbar[k] ** 2).sum() for k in keys_f))
            if cfg.get("fedexp"):
                # FedExP (ICLR'23): eta_g = max(1, mean||d_i||^2 / (2(||dbar||^2+eps)))
                di_sq = [float(sum(((st[k] - gprev[k]) ** 2).sum() for k in keys_f))
                         for st, _w in weighted_states]
                eta_g = max(1.0, (sum(di_sq) / max(len(di_sq), 1)) / (2.0 * (dbar_sq + 1e-10)))
            else:
                # FedHyper-G: server alpha += <dbar_t, dbar_{t-1}>, clip [0.1, 10]
                s = 0.0
                if sched_ctrl.get("prev_dbar") is not None:
                    s = float(sum((dbar[k] * sched_ctrl["prev_dbar"][k]).sum() for k in keys_f))
                if cfg.get("norm_update"):
                    s = 0.05 * (1.0 if s > 0 else -1.0) if s != 0.0 else 0.0
                sched_ctrl["g_alpha"] = min(max(sched_ctrl.get("g_alpha", 1.0) + s, 0.1), 10.0)
                eta_g = sched_ctrl["g_alpha"]
                sched_ctrl["prev_dbar"] = dbar
            for k in keys_f:
                aggregated[k] = gprev[k] + eta_g * dbar[k]
            sched_ctrl["eta_g"] = eta_g
        if cfg.get("drift_lr"):
            keys_f2 = [k for k in aggregated if torch.is_floating_point(aggregated[k])]
            gprev2 = {k: global_state[k].cpu() for k in keys_f2}
            norms = [float(sum(((st[k] - gprev2[k]) ** 2).sum() for k in keys_f2)) ** 0.5
                     for st, _w in weighted_states]
            med = sorted(norms)[len(norms) // 2]
            sched_ctrl["drift_scale"] = {clients[i]["idx"]: min(max(med / max(norms[i], 1e-9), 0.5), 2.0)
                                         for i in range(len(clients))}
        if cfg.get("plateau_lr") and round_client_losses:
            ml = float(np.mean(round_client_losses))
            best = sched_ctrl.get("pl_best")
            if best is None or ml < best - 1e-4:
                sched_ctrl["pl_best"] = ml if best is None else min(ml, best)
                sched_ctrl["pl_wait"] = 0
            else:
                sched_ctrl["pl_wait"] = sched_ctrl.get("pl_wait", 0) + 1
                if sched_ctrl["pl_wait"] >= 3:
                    sched_ctrl["pl_scale"] = max(sched_ctrl.get("pl_scale", 1.0) * 0.5, 0.05)
                    sched_ctrl["pl_wait"] = 0
        if cfg.get("fedadamw_style") and sched_ctrl.get("v_harvest"):
            vh = sched_ctrl.pop("v_harvest")
            sched_ctrl["avg_v"] = {n: sum(v[n] for v in vh) / len(vh) for n in vh[0]}
        if cfg.get("vsync") and sched_ctrl.get("vsync_harvest"):
            vh = sched_ctrl.pop("vsync_harvest")
            avg_v = {n: sum(v[n] for v in vh) / len(vh) for n in vh[0]}
            # broadcast averaged second-moment back to every client optimizer
            for cl in clients:
                opt = cl.get("optimizer")
                if opt is None:
                    continue
                for n, p in cl["policy"].named_parameters():
                    st = opt.state.get(p)
                    if st is not None and "exp_avg_sq" in st and n in avg_v:
                        st["exp_avg_sq"].copy_(avg_v[n].to(st["exp_avg_sq"].device))
        if cfg.get("pgs_lambda") is not None and sched_ctrl.get("pgs_harvest"):
            # PGS: v_bar = mean_i v_i; each client next-round v = v_bar + lam(v_i - v_bar)
            ph = sched_ctrl.pop("pgs_harvest")
            lam = float(cfg["pgs_lambda"])
            names = list(ph[0][1].keys())
            vbar = {n: sum(vv[n] for _, vv in ph) / len(ph) for n in names}
            for cl, vv_i in ph:
                opt = cl.get("optimizer")
                if opt is None:
                    continue
                for n, p in cl["policy"].named_parameters():
                    st = opt.state.get(p)
                    if st is not None and "exp_avg_sq" in st and n in vbar:
                        shrunk = vbar[n] + lam * (vv_i[n] - vbar[n])
                        st["exp_avg_sq"].copy_(shrunk.to(st["exp_avg_sq"].device))
        if cfg.get("scaffold") and all(c.get("sc_dci") for c in clients):
            N = len(clients)
            for n in sched_ctrl["sc_c"]:
                sched_ctrl["sc_c"][n] = (sched_ctrl["sc_c"][n]
                                         + sum(c["sc_dci"][n] for c in clients) / N)
        if cfg.get("server_ema"):
            beta = float(cfg["server_ema"])
            for k in list(aggregated.keys()):
                if torch.is_floating_point(aggregated[k]):
                    aggregated[k] = (beta * global_state[k].detach().cpu()
                                     + (1 - beta) * aggregated[k])
        effective_server_momentum = server_momentum
        if server_momentum > 0.0 and server_momentum_decay_end is not None:
            decay_end = float(server_momentum_decay_end)
            effective_server_momentum = server_momentum * max(0.0, 1.0 - (round_idx - 1) / decay_end)
        if effective_server_momentum > 0.0 or (server_momentum > 0.0 and server_momentum_buffer is not None):
            prev_global = {k: v.detach().cpu() for k, v in global_state.items()}
            if server_momentum_buffer is None:
                server_momentum_buffer = {
                    key: torch.zeros_like(value)
                    for key, value in aggregated.items()
                    if torch.is_floating_point(value)
                    and (server_momentum_scope != "trunk" or key.startswith("obs_encoder."))
                }
            for key, value in aggregated.items():
                if key in server_momentum_buffer:
                    delta = value - prev_global[key]
                    if server_momentum_dampened:
                        server_momentum_buffer[key].mul_(effective_server_momentum).add_(delta, alpha=1.0 - effective_server_momentum)
                    else:
                        server_momentum_buffer[key].mul_(effective_server_momentum).add_(delta)
                    aggregated[key] = prev_global[key] + server_momentum_buffer[key]
        global_policy.load_state_dict(aggregated)
        if round_idx in save_rounds and cfg.get("mca_prefixes"):
            # MCA has no single deployable model: save one checkpoint per mode
            # (shared trunk + that mode's own components). round_XXXX.pth is
            # mode 0 so the existing eval path stays valid; _modeN.pth holds
            # the rest and is evaluated separately.
            _base = cpu_state_dict(global_policy)
            for _ci, _cl in enumerate(clients):
                _sd = {k: v.clone() for k, v in _base.items()}
                for _k, _v in (_cl.get("mca_own") or {}).items():
                    _sd[_k] = _v
                _name = (f"round_{round_idx:04d}.pth" if _ci == 0
                         else f"round_{round_idx:04d}_mode{_ci}.pth")
                torch.save(_sd, out_dir / _name)
        if round_idx in save_rounds:
            ckpt = out_dir / f"round_{round_idx:04d}.pth"
            if not cfg.get("mca_prefixes"):
                torch.save(cpu_state_dict(global_policy), ckpt)
            torch.save(cpu_state_dict(global_policy), latest)
            write_csv(out_dir / "round_log.csv", rows)
            write_csv(out_dir / "moment_diagnostics.csv", moment_rows)
            if conflict_rows:
                write_csv(out_dir / "flow_conflict_diagnostics.csv", conflict_rows)
    torch.save(cpu_state_dict(global_policy), final_ckpt)
    torch.save(cpu_state_dict(global_policy), latest)
    write_csv(out_dir / "round_log.csv", rows)
    write_csv(out_dir / "moment_diagnostics.csv", moment_rows)
    if conflict_rows:
        write_csv(out_dir / "flow_conflict_diagnostics.csv", conflict_rows)
    with (out_dir / "train_done.json").open("w") as f:
        json.dump({"elapsed_seconds": time.time() - started, "final_ckpt": str(final_ckpt)}, f, indent=2)
    return final_ckpt


def run_evaluate(args, ckpt, eval_dir, name):
    json_path = eval_dir / f"{name}.json"
    if json_path.exists() and not args.force:
        return json.loads(json_path.read_text())
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "streaming_flow_policy/pusht/evaluate.py"),
        "--model-type", "sfps",
        "--imitation-mode", args.imitation_mode,
        "--obs-mode", args.obs_mode,
        "--ckpt-path", str(ckpt),
        "--save-dir", str(eval_dir),
        "--name", name,
        "--integration-steps-per-action", str(args.integration_steps_per_action),
        "--solver", args.solver,
        "--num-tests", str(args.eval_num_tests),
        "--seed", str(args.seed),
        "--device", args.device,
        "--env-start-seed", str(args.env_start_seed),
        "--max-rollout-steps", str(args.max_rollout_steps),
        "--success-score-threshold", str(args.success_score_threshold),
        "--deterministic-z0-eval",
        "--obs-horizon", str(args.obs_horizon),
        "--action-horizon", str(args.action_horizon),
        # image-conditioning flags only apply to the image observation line;
        # rollout start follows the imitation mode (action -> current pusher)
        "--rollout-start-mode",
        "current_pusher" if args.imitation_mode == "action" else "state",
    ]
    if args.obs_mode == "image":
        cmd.extend([
            "--image-cond-dim", str(args.image_cond_dim),
            "--image-encoder", args.image_encoder,
            "--image-agent-pos-cond",
            "--image-key", args.image_key,
            "--image-crop-size", str(args.image_random_crop_size),
        ])
    if int(getattr(args, "flare_experts", 0) or 0) > 0:
        # the evaluator reads expert count and width from the checkpoint;
        # these two are not in the weights and must be told
        cmd.extend(["--flare-t-split", str(args.flare_t_split)])
        if getattr(args, "flare_confidence_gate", False):
            cmd.append("--flare-confidence-gate")
    if getattr(args, "eval_episode_indices_json", None) is not None:
        cmd.extend([
            "--episode-indices-json", str(args.eval_episode_indices_json),
            "--stats-episode-indices-key", "union_200",
        ])
    import subprocess
    print("[p47 eval]", " ".join(cmd), flush=True)
    with (eval_dir / f"{name}.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(json_path.read_text())


def eval_checkpoint(args, seed_dir, method, round_idx, ckpt, initial_state, cfg_key=None):
    name = f"p47_image_action_{method}_seed{args.seed}_round{round_idx:04d}_max{args.max_rollout_steps}_detz0"
    result = run_evaluate(args, ckpt, seed_dir / "eval", name)
    _ck = torch.load(ckpt, map_location="cpu")
    if int(getattr(args, "flare_experts", 0) or 0) > 0:
        # the checkpoint is wrapped and the initial state is not; strip the
        # prefix so this still measures backbone drift, and drop the rows the
        # wrapper added, which have no initial value to drift from
        _ck = {k.replace("velocity_net.base.", "velocity_net.", 1): v
               for k, v in _ck.items()
               if not k.startswith(("velocity_net.expert_", "velocity_net.gate."))}
    delta = state_delta(initial_state, _ck)
    # MCA per-mode rows carry a synthetic variant label ("<method>__modeN");
    # cfg_key points back at the real registry entry.
    cfg = METHOD_CONFIGS[cfg_key or method]
    return {
        "seed": args.seed,
        "severity_p": args.severity_p,
        "split_seed": args.split_seed,
        "variant": method,
        "method": cfg["method"],
        "checkpoint_round": round_idx,
        "score_x100": result["mean_score_x100"],
        "std_score_x100": result["std_score"] * 100.0,
        "success": result["success_rate"],
        "num_success": result["num_success"],
        "num_tests": result["num_tests"],
        "max_score_x100": result["max_score_x100"],
        "latency_ms_per_action": result["amortized_action_latency_ms"],
        "latency_ms_per_policy_call": result["mean_inference_latency_ms"],
        "image_encoder_ms_per_call": result.get("mean_image_encoder_latency_ms", 0.0),
        "sfp_predictor_ms_per_call": result.get("mean_sfp_predictor_latency_ms", 0.0),
        "mean_num_policy_calls": result.get("mean_num_policy_calls", 0.0),
        "mean_signed_rotation": result.get("mean_signed_rotation", ""),
        "mean_rotation_purity": result.get("mean_rotation_purity", ""),
        "frac_ccw_episodes": result.get("frac_ccw_episodes", ""),
        "mean_trajectory_length": result.get("mean_trajectory_length", 0.0),
        "max_rollout_steps": args.max_rollout_steps,
        "deterministic_z0_eval": args.deterministic_z0_eval,
        "obs_horizon": args.obs_horizon,
        "pred_horizon": args.pred_horizon,
        "executed_H": args.action_horizon,
        "start_mode": "current_pusher",
        "all_zero_scores": bool(all(float(e.get("score", 0.0)) == 0.0 for e in result.get("episodes", []))),
        "nonzero_score_count": int(sum(float(e.get("score", 0.0)) != 0.0 for e in result.get("episodes", []))),
        "rel_l2_from_init": delta.get("rel_l2"),
        "ckpt_path": str(ckpt),
    }


def summarize_avgmax(rows):
    summaries = []
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["variant"]), []).append(row)
    for (seed, variant), group_rows in sorted(groups.items()):
        ranked = sorted(group_rows, key=lambda r: r["score_x100"], reverse=True)
        top5 = ranked[: min(5, len(ranked))]
        best = ranked[0]
        summaries.append({
            "seed": seed,
            "variant": variant,
            "method": best["method"],
            "avg5_score_x100": float(np.mean([r["score_x100"] for r in top5])),
            "avg5_success": float(np.mean([r["success"] for r in top5])),
            "avg5_latency_ms_per_action": float(np.mean([r["latency_ms_per_action"] for r in top5])),
            "avg5_rounds": ";".join(str(r["checkpoint_round"]) for r in top5),
            "max_score_x100": best["score_x100"],
            "max_success": best["success"],
            "max_latency_ms_per_action": best["latency_ms_per_action"],
            "max_checkpoint_round": best["checkpoint_round"],
            "num_checkpoints": len(group_rows),
        })
    return summaries


def summarize_seed_delta(summary_rows):
    by_seed = {}
    for row in summary_rows:
        by_seed.setdefault(row["seed"], {})[row["variant"]] = row
    rows = []
    for seed, data in sorted(by_seed.items()):
        raw = data.get("raw_reset_both")
        pers = data.get("persistent_both")
        if not raw or not pers:
            continue
        rows.append({
            "seed": seed,
            "persistent_minus_raw_avg5_score_x100": pers["avg5_score_x100"] - raw["avg5_score_x100"],
            "persistent_minus_raw_avg5_success": pers["avg5_success"] - raw["avg5_success"],
            "raw_avg5_score_x100": raw["avg5_score_x100"],
            "persistent_avg5_score_x100": pers["avg5_score_x100"],
            "raw_avg5_success": raw["avg5_success"],
            "persistent_avg5_success": pers["avg5_success"],
        })
    if rows:
        rows.append({
            "seed": "mean",
            "persistent_minus_raw_avg5_score_x100": float(np.mean([r["persistent_minus_raw_avg5_score_x100"] for r in rows])),
            "persistent_minus_raw_avg5_success": float(np.mean([r["persistent_minus_raw_avg5_success"] for r in rows])),
            "raw_avg5_score_x100": float(np.mean([r["raw_avg5_score_x100"] for r in rows])),
            "persistent_avg5_score_x100": float(np.mean([r["persistent_avg5_score_x100"] for r in rows])),
            "raw_avg5_success": float(np.mean([r["raw_avg5_success"] for r in rows])),
            "persistent_avg5_success": float(np.mean([r["persistent_avg5_success"] for r in rows])),
        })
    return rows


def summarize_endpoint_delta(summary_rows):
    by_seed = {}
    for row in summary_rows:
        by_seed.setdefault(row["seed"], {})[row["variant"]] = row
    rows = []
    for seed, data in sorted(by_seed.items()):
        persistent = data.get("persistent_both")
        projected = data.get("endpoint_projected")
        if not persistent or not projected:
            continue
        rows.append({
            "seed": seed,
            "projected_minus_persistent_avg5_score_x100": projected["avg5_score_x100"] - persistent["avg5_score_x100"],
            "projected_minus_persistent_avg5_success": projected["avg5_success"] - persistent["avg5_success"],
            "persistent_avg5_score_x100": persistent["avg5_score_x100"],
            "projected_avg5_score_x100": projected["avg5_score_x100"],
            "persistent_avg5_success": persistent["avg5_success"],
            "projected_avg5_success": projected["avg5_success"],
        })
    return rows


def write_prereg(path, args, checkpoint_rounds, methods):
    lines = [
        "# P47 Image-Action Non-IID S1 Persistent Preregistration",
        "",
        "Design: fill the image_action Persistent half on the same first200 union split source, severity p=1.0 split seed, eval seeds, deterministic z0, fixed rollout/eval protocol, gate logic and diagnostics.",
        "",
        "## Scope",
        "",
        "- This run only evaluates image_action Persistent for seeds 16/17/18. If paired image_action Raw results are unavailable, Raw delta remains pending.",
        "- The setting follows the current image_action main line: k=2, obs_horizon=2, current_pusher rollout start, no action marker, deterministic z0=0, max_rollout_steps=300.",
        "- Cross comparison should use paired image_action Raw if available; otherwise this is a Persistent-only completion run.",
        "",
        "## Run Protocol",
        "",
        f"- seeds: {args.seeds}",
        f"- split_seed: {args.split_seed}",
        f"- severity_p: {args.severity_p}",
        f"- checkpoint rounds: {','.join(str(x) for x in checkpoint_rounds)}",
        f"- methods: {','.join(methods)}",
        "- model: SFPS, obs=image, imitation=action, image_encoder=final_v1, image_agent_pos_cond=true, crop=84",
        "- training: k=2, sigma0=sigma1=0.1, time-decayed stabilized objective, pred_horizon=16, obs_horizon=2, action_horizon=8",
        "- eval: 50 episodes, env seeds 1300-1349, max_rollout_steps=300, Euler, integration_steps_per_action=1, deterministic z0=0, rollout_start_mode=current_pusher, stats=union_200",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_report(path, checkpoint_rows, summary_rows, delta_rows, split_rows, args):
    lines = [
        "# P47 Image-Action Non-IID S1 Persistent",
        "",
        "Setting: Push-T image input, action imitation, SFPS stabilized k=2, obs_horizon=2, pred_horizon=16, fixed H=8, Non-IID p=1.0 initial-x split, current-pusher rollout start.",
        "Scope: Persistent-only image_action completion run unless paired Raw results are separately available.",
        "",
        "## Avg5 / Max Summary",
        "",
        "| Seed | Method | Avg5 Score x100 | Avg5 Success | Avg5 Latency | Avg5 Rounds | Max Score x100 | Max Success | Max Latency | Max Round |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['avg5_score_x100']:.3f} | {row['avg5_success']:.3f} | "
            f"{row['avg5_latency_ms_per_action']:.3f} | {row['avg5_rounds']} | {row['max_score_x100']:.3f} | "
            f"{row['max_success']:.3f} | {row['max_latency_ms_per_action']:.3f} | {row['max_checkpoint_round']} |"
        )
    lines += [
        "",
        "## Persistent - Raw Delta",
        "",
        "| Seed | Delta Avg5 Score x100 | Delta Avg5 Success | Raw Avg5 | Persistent Avg5 | Raw Success | Persistent Success |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in delta_rows:
        lines.append(
            f"| {row['seed']} | {row['persistent_minus_raw_avg5_score_x100']:.3f} | {row['persistent_minus_raw_avg5_success']:.3f} | "
            f"{row['raw_avg5_score_x100']:.3f} | {row['persistent_avg5_score_x100']:.3f} | "
            f"{row['raw_avg5_success']:.3f} | {row['persistent_avg5_success']:.3f} |"
        )
    lines += [
        "",
        "## Per-Checkpoint Results",
        "",
        "| Seed | Method | Round | Score x100 | Success | Latency ms/action |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in checkpoint_rows:
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['checkpoint_round']} | "
            f"{row['score_x100']:.3f} +/- {row['std_score_x100']:.3f} | "
            f"{row['success']:.3f} ({row['num_success']}/{row['num_tests']}) | "
            f"{row['latency_ms_per_action']:.3f} |"
        )
    lines += [
        "",
        "## Split Diagnostics",
        "",
        "| Seed | Client | Size | Own-bin ratio | x min | x max | x mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            f"| {row.get('seed', '')} | {row['client']} | {row['size']} | {row.get('own_bin_ratio', float('nan')):.3f} | "
            f"{row['x_min']:.3f} | {row['x_max']:.3f} | {row['x_mean']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    seeds = parse_int_csv(args.seeds)
    checkpoint_rounds = sorted(set(parse_int_list(args.checkpoint_rounds)))
    if args.smoke:
        checkpoint_rounds = [1]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_CONFIGS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_prereg(args.out_dir / "P47_PREREGISTRATION.md", args, checkpoint_rounds, methods)
    base_split = json.loads(args.base_split_json.read_text())
    all_checkpoint_rows = []
    all_split_rows = []
    for seed in seeds:
        seed_dir = args.out_dir / f"seed{seed}_split{args.split_seed}_p1p0"
        if args.smoke:
            seed_dir = seed_dir / "smoke"
        seed_dir.mkdir(parents=True, exist_ok=True)
        p_args = phase_args(args, seed, seed_dir, max(checkpoint_rounds))
        p_args.checkpoint_rounds = checkpoint_rounds
        p_args.severity_p = args.severity_p
        p_args.split_seed = args.split_seed
        p_args.force_train = args.force_train
        if getattr(args, "custom_split", False):
            # Use the split JSON's client_* partition verbatim (rotation CW/CCW
            # mode-shift etc.) instead of re-partitioning by initial pusher x.
            split = dict(base_split)
            ckeys = sorted(
                [k for k in base_split
                 if k.startswith("client_") and k.split("_", 1)[1].isdigit()],
                key=lambda k: int(k.split("_", 1)[1]))
            split_rows = [{"client": int(k.split("_", 1)[1]),
                           "size": len(base_split[k]),
                           "note": base_split.get("split", "custom")} for k in ckeys]
        else:
            split, split_rows = make_severity_split(base_split, args.severity_p, args.split_seed)
        if getattr(args, "centralized", False):
            ckeys = client_keys(split)
            union = sorted({int(e) for k in ckeys for e in split[k]})
            for k in ckeys:
                split.pop(k, None)
            split["client_0"] = union
            split_rows = []
        for row in split_rows:
            row["seed"] = seed
        (seed_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")
        p_args.eval_episode_indices_json = seed_dir / "split.json"
        write_csv(seed_dir / "split_diagnostics.csv", split_rows)
        all_split_rows.extend(split_rows)
        set_seed(seed)
        initial_state, initial_path = make_initial_state(p_args)
        for method in methods:
            method_dir = seed_dir / method
            print(f"[p47] seed={seed} method={method} rounds={p_args.rounds}", flush=True)
            run_variant(p_args, split, initial_state, initial_path, method_dir, method)
            for round_idx in checkpoint_rounds:
                ckpt = method_dir / f"round_{round_idx:04d}.pth"
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)
                row = eval_checkpoint(p_args, seed_dir, method, round_idx, ckpt, initial_state)
                row["mca_mode"] = 0 if METHOD_CONFIGS[method].get("mca_prefixes") else ""
                all_checkpoint_rows.append(row)
                # MCA: no single deployable model -> score every mode separately
                # (S_CW, S_CCW reported side by side; never collapsed to one number)
                if METHOD_CONFIGS[method].get("mca_prefixes"):
                    for extra in sorted(method_dir.glob(f"round_{round_idx:04d}_mode*.pth")):
                        mode_id = int(extra.stem.split("_mode")[1])
                        mrow = eval_checkpoint(p_args, seed_dir, f"{method}__mode{mode_id}",
                                               round_idx, extra, initial_state,
                                               cfg_key=method)
                        mrow["mca_mode"] = mode_id
                        all_checkpoint_rows.append(mrow)
                write_csv(args.out_dir / "p47_checkpoint_results_partial.csv", all_checkpoint_rows)
    summary_rows = summarize_avgmax(all_checkpoint_rows)
    delta_rows = summarize_seed_delta(summary_rows)
    endpoint_delta_rows = summarize_endpoint_delta(summary_rows)
    write_csv(args.out_dir / "p47_checkpoint_results.csv", all_checkpoint_rows)
    write_csv(args.out_dir / "p47_avgmax_summary.csv", summary_rows)
    write_csv(args.out_dir / "p47_persistent_minus_raw_delta.csv", delta_rows)
    write_csv(args.out_dir / "p47_endpoint_projected_minus_persistent_delta.csv", endpoint_delta_rows)
    write_csv(args.out_dir / "split_diagnostics_all.csv", all_split_rows)
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "script_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seeds": seeds,
        "checkpoint_rounds": checkpoint_rounds,
        "methods": methods,
        "anchor": {"line": "image_action h=2 k=2 current main line", "gate_pass": True},
        "paired_raw_status": "pending unless provided by a separate image_action Raw run",
        "eval_check": {
            "render_action_marker": False,
            "rollout_start_mode": "current_pusher",
            "eval_code_path": "streaming_flow_policy/pusht/evaluate.py -> Rollout",
        },
    }, indent=2))
    write_report(args.out_dir / "P47_IMAGE_ACTION_NONIID_S1_PERSISTENT_REPORT.md", all_checkpoint_rows, summary_rows, delta_rows, all_split_rows, args)
    print(args.out_dir / "P47_IMAGE_ACTION_NONIID_S1_PERSISTENT_REPORT.md")


if __name__ == "__main__":
    main()
