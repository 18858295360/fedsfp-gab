#!/usr/bin/env python3
"""Convert RoboMimic delta actions using live sequential controller state.

This is for Square PH where static per-HDF-state conversion is task-invalid:
the delta controller must be replayed sequentially, and each absolute target
is exported from the live controller state immediately before the raw step.
"""

import argparse
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory

from scripts.robomimic_sfp.convert_table3_absolute_actions import (
    arm_controller,
    reset_arm_goal,
    set_absolute_pose_controller,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-demos", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def initialize_obs_utils():
    ObsUtils.initialize_obs_utils_with_config(config_factory("bc"))


def create_envs(dataset_path):
    initialize_obs_utils()
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    delta_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    abs_env = EnvUtils.create_env_from_metadata(
        env_meta=set_absolute_pose_controller(env_meta),
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    return delta_env, abs_env


def initial_state(demo):
    state = {"states": demo["states"][0]}
    if "model_file" in demo.attrs:
        state["model"] = demo.attrs["model_file"]
    if "ep_meta" in demo.attrs:
        state["ep_meta"] = demo.attrs["ep_meta"]
    return state


def convert_demo_live(delta_env, demo):
    actions = demo["actions"][:].astype(np.float32)
    delta_env.reset_to(initial_state(demo))
    reset_arm_goal(delta_env)
    converted = np.empty_like(actions, dtype=np.float32)
    raw_success = False
    for idx, action in enumerate(actions):
        controller = arm_controller(delta_env)
        controller.update()
        abs_pose = controller.delta_to_abs_action(
            controller.scale_action(action[:6]),
            goal_update_mode=getattr(controller, "_goal_update_mode", "achieved"),
        ).astype(np.float32)
        open_fraction = np.float32((1.0 - np.clip(action[-1], -1.0, 1.0)) / 2.0)
        converted[idx] = np.concatenate((abs_pose, [open_fraction]))
        _, _, done, info = delta_env.step(action)
        raw_success = raw_success or bool(info.get("is_success", {}).get("task", False))
        if done:
            converted[idx + 1 :] = converted[idx]
            break
    return converted, raw_success


def validate_demo(abs_env, demo, converted):
    abs_env.reset_to(initial_state(demo))
    reset_arm_goal(abs_env)
    success = False
    rewards = []
    for step, action in enumerate(converted):
        env_action = action.copy()
        env_action[-1] = 1.0 - 2.0 * np.clip(env_action[-1], 0.0, 1.0)
        _, reward, done, info = abs_env.step(env_action)
        rewards.append(float(reward))
        success = success or bool(info.get("is_success", {}).get("task", False))
        if success or done:
            break
    return {
        "success": bool(success),
        "steps": int(step + 1),
        "max_reward": float(max(rewards) if rewards else 0.0),
    }


def worker(payload):
    dataset_path, demo_indices, validate_indices = payload
    delta_env, abs_env = create_envs(dataset_path)
    converted = {}
    raw_success = {}
    validation = {}
    with h5py.File(dataset_path, "r") as src:
        for demo_idx in demo_indices:
            demo = src[f"data/demo_{demo_idx}"]
            actions, success = convert_demo_live(delta_env, demo)
            converted[demo_idx] = actions
            raw_success[demo_idx] = success
            if demo_idx in validate_indices:
                validation[demo_idx] = validate_demo(abs_env, demo, actions)
    return converted, raw_success, validation


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        num_demos = len(src["data"])
    indices = np.arange(num_demos)
    chunks = [chunk.tolist() for chunk in np.array_split(indices, args.workers) if len(chunk)]
    validate_indices = set(args.validate_demos)
    payloads = [(str(input_path), chunk, validate_indices) for chunk in chunks]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(payloads)) as pool:
        results = pool.map(worker, payloads)

    shutil.copy2(input_path, output_path)
    validation = {}
    raw_success = {}
    with h5py.File(output_path, "r+") as dst:
        for converted, worker_raw_success, worker_validation in results:
            for demo_idx, actions in converted.items():
                dst[f"data/demo_{demo_idx}/actions"][:] = actions
            raw_success.update(worker_raw_success)
            validation.update(worker_validation)
        dst["data"].attrs["table3_action_space"] = (
            "live_sequential_world_eef_pos_axis_angle_plus_continuous_gripper_open_fraction"
        )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "num_demos": num_demos,
        "converter": "live sequential delta-controller state, goal_update_mode from controller",
        "raw_success_rate_during_conversion": float(np.mean(list(raw_success.values()))),
        "validation": {str(key): value for key, value in sorted(validation.items())},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
