from __future__ import annotations

from typing import Tuple

import numpy as np


def first_order_hold_value_and_derivative(
    times: np.ndarray,
    positions: np.ndarray,
    time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a piecewise-linear trajectory and its segment derivative.

    Drake's PiecewisePolynomial.FirstOrderHold is convenient but optional for
    Push-T SFP. This helper keeps the training target identical for the linear
    trajectories used here without requiring pydrake in the runtime env.
    """
    times = np.asarray(times, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    if times.ndim != 1:
        raise ValueError(f"times must be 1D, got shape {times.shape}")
    if positions.ndim != 2 or positions.shape[0] != times.shape[0]:
        raise ValueError(
            "positions must have shape (len(times), dim), "
            f"got times {times.shape}, positions {positions.shape}"
        )
    if len(times) < 2:
        raise ValueError("at least two trajectory knots are required")

    time = float(np.clip(time, times[0], times[-1]))
    seg_idx = int(np.searchsorted(times, time, side="right") - 1)
    seg_idx = min(max(seg_idx, 0), len(times) - 2)

    t0 = times[seg_idx]
    t1 = times[seg_idx + 1]
    p0 = positions[seg_idx]
    p1 = positions[seg_idx + 1]
    alpha = (time - t0) / max(float(t1 - t0), np.finfo(np.float32).eps)
    value = (1.0 - alpha) * p0 + alpha * p1
    derivative = (p1 - p0) / max(float(t1 - t0), np.finfo(np.float32).eps)
    return value[None, :].astype(np.float32), derivative[None, :].astype(np.float32)
