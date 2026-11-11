"""Helpers from the study module phase1_fedavg_state_ablation.py used by the Push-T trainer."""

import numpy as np
import torch
from diffusers.optimization import get_scheduler


def make_optimizer(args, policy):
    return torch.optim.AdamW(
        params=policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def make_reset_scheduler(optimizer, steps_per_round):
    return get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max(steps_per_round, 1),
    )


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def cpu_state_dict(policy):
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


def state_delta(base, other):
    sq = 0.0
    base_sq = 0.0
    abs_sum = 0.0
    count = 0
    max_abs = 0.0
    changed = 0
    total = 0
    for key, base_value in base.items():
        if not torch.is_floating_point(base_value):
            continue
        total += 1
        diff = other[key].float() - base_value.float()
        diff_sq = float(torch.sum(diff * diff).item())
        sq += diff_sq
        base_sq += float(torch.sum(base_value.float() * base_value.float()).item())
        abs_sum += float(torch.sum(torch.abs(diff)).item())
        count += diff.numel()
        max_abs = max(max_abs, float(torch.max(torch.abs(diff)).item()))
        if diff_sq > 0.0:
            changed += 1
    l2 = float(np.sqrt(sq))
    base_l2 = float(np.sqrt(base_sq))
    return {
        "l2": l2,
        "base_l2": base_l2,
        "rel_l2": float(l2 / max(base_l2, 1e-12)),
        "mean_abs": float(abs_sum / max(count, 1)),
        "max_abs": max_abs,
        "changed_float_tensors": changed,
        "float_tensors": total,
    }
