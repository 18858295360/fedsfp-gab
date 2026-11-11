#!/usr/bin/env python3
import argparse

import os as _sfp_os, json as _sfp_json
_SFP_DOWN = _sfp_json.loads(_sfp_os.environ.get('SFP_DOWN_DIMS', '[256, 512, 1024]'))
_SFP_EMBED = int(_sfp_os.environ.get('SFP_EMBED', '256'))
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

from streaming_flow_policy.robomimic_sfp.model_v21 import SFPConditionalUnet1D
from streaming_flow_policy.robomimic_sfp.table3_papergrid_dp_v1 import (
    Table3PaperGridDPDataset,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--gripper-source",
        choices=["next_qpos", "binary_command"],
        default="next_qpos",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sigma0", type=float, default=0.4)
    parser.add_argument("--k", type=float, default=15.0)
    parser.add_argument("--integration-steps-per-action", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--max-train-demos", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--overfit-one-batch-steps", type=int, default=0)
    parser.add_argument("--scheduler-total-steps", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def save_checkpoint(path, model, ema, optimizer, scheduler, dataset, args, epoch, step, loss):
    torch.save(
        {
            "model_state": model.state_dict(),
            "ema_model_state": ema.averaged_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "normalizer": dataset.normalizer.state_dict(),
            "obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "obs_horizon": dataset.obs_horizon,
            "pred_horizon": dataset.pred_horizon,
            "model_config": {
                "diffusion_step_embed_dim": _SFP_EMBED,
                "down_dims": _SFP_DOWN,
                "kernel_size": 5,
                "n_groups": 8,
                "fc_timesteps": 1,
            },
            "table3": {
                "dataset_variant": "square_mh_dp_absolute_standard_foh_gripper_v1",
                "num_total_demos": len(dataset.episodes),
                "num_train_demos": len(dataset.train_indices),
                "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                "action_space": "world EEF pos + rotation6d + gripper open fraction",
                "trajectory_grid_points": dataset.pred_horizon + 1,
                "trajectory_grid_intervals": dataset.pred_horizon,
                "trajectory_alignment": "point0=action[t-1], point1=action[t]",
                "gripper_velocity": "piecewise_linear_first_order_hold",
                "gripper_source": dataset.gripper_source,
                "observation_source": "raw_hdf_observations",
                "controller_anchor_cache": False,
                "biased_gripper_sampling": bool(
                    getattr(args, "biased_gripper_sampling", False)
                ),
                "biased_gripper_prob": getattr(args, "biased_gripper_prob", None),
                "sigma0": args.sigma0,
                "k": args.k,
                "integration_steps_per_action": args.integration_steps_per_action,
                "action_horizon_square": 8,
            },
            "args": vars(args),
            "epoch": epoch,
            "global_step": step,
            "loss": loss,
        },
        path,
    )


def main():
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Table3PaperGridDPDataset(
        args.dataset_path,
        sigma0=args.sigma0,
        stabilization_k=args.k,
        integration_steps_per_action=args.integration_steps_per_action,
        gripper_source=args.gripper_source,
        max_train_demos=args.max_train_demos,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    model = SFPConditionalUnet1D(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        obs_horizon=dataset.obs_horizon,
        down_dims=_SFP_DOWN,
        diffusion_step_embed_dim=_SFP_EMBED,
        fc_timesteps=1,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = min(len(loader), args.steps_per_epoch or len(loader))
    run_total_steps = args.overfit_one_batch_steps or (steps_per_epoch * args.epochs)
    total_steps = args.scheduler_total_steps or run_total_steps
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    ema = EMAModel(model, power=0.75)
    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(resume["model_state"])
        ema.averaged_model.load_state_dict(resume["ema_model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        start_epoch = int(resume["epoch"])
        global_step = int(resume["global_step"])
    (output / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "dataset_samples": len(dataset),
                "total_demos": len(dataset.episodes),
                "train_demos": len(dataset.train_indices),
                "obs_dim": dataset.obs_dim,
                "action_dim": dataset.action_dim,
                "total_optimizer_steps": total_steps,
                "run_optimizer_steps": run_total_steps,
            },
            indent=2,
        )
    )

    fixed_batch = next(iter(loader)) if args.overfit_one_batch_steps else None
    if fixed_batch is not None and args.resume_checkpoint is not None:
        raise ValueError("one-batch overfit cannot resume")
    initial_loss = None
    epochs = 1 if fixed_batch is not None else args.epochs
    for epoch in range(start_epoch + 1, epochs + 1):
        batches = [fixed_batch] * args.overfit_one_batch_steps if fixed_batch is not None else loader
        losses = []
        for batch_idx, batch in enumerate(batches):
            if fixed_batch is None and batch_idx >= steps_per_epoch:
                break
            obs = batch["obs"].to(device, non_blocking=True)
            action = batch["a"].to(device, non_blocking=True)
            t = batch["t"].to(device, non_blocking=True)
            target = batch["v"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(obs, action, t), target)
            loss.backward()
            # Match the upstream deterministic SFP trainer: no gradient clipping.
            optimizer.step()
            scheduler.step()
            ema.step(model)
            value = float(loss.detach().cpu())
            if initial_loss is None:
                initial_loss = value
            losses.append(value)
            global_step += 1
        mean_loss = float(np.mean(losses))
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": mean_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record), flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if fixed_batch is None and (epoch % args.save_every == 0 or epoch == epochs):
            save_checkpoint(
                output / f"checkpoint_epoch_{epoch:04d}.pt",
                model, ema, optimizer, scheduler, dataset, args, epoch, global_step, mean_loss,
            )
            save_checkpoint(
                output / "latest.pt",
                model, ema, optimizer, scheduler, dataset, args, epoch, global_step, mean_loss,
            )

    if fixed_batch is not None:
        gate = {
            "initial_loss": initial_loss,
            "final_loss": losses[-1],
            "ratio": losses[-1] / initial_loss,
            "passed": bool(losses[-1] < 0.1 * initial_loss),
        }
        (output / "overfit_gate.json").write_text(json.dumps(gate, indent=2))
        print(json.dumps(gate), flush=True)
        if not gate["passed"]:
            raise RuntimeError(f"One-batch overfit gate failed: {gate}")


if __name__ == "__main__":
    main()
