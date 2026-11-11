#!/usr/bin/env python3
"""Convert Can actions from recorded states with an audited controller round trip."""

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.robomimic_sfp.convert_table3_absolute_actions import arm_controller, create_envs


POSITION_TOLERANCE = 1e-5
ORIENTATION_TOLERANCE = 1e-5
GRIPPER_TOLERANCE = 1e-6


def initial_state(demo):
    state = {"states": demo["states"][0]}
    if "model_file" in demo.attrs:
        state["model"] = demo.attrs["model_file"]
    if "ep_meta" in demo.attrs:
        state["ep_meta"] = demo.attrs["ep_meta"]
    return state


def state_at(demo, states, step):
    return initial_state(demo) if step == 0 else {"states": states[step]}


def convert_demo(delta_env, demo):
    states = demo["states"][:]
    raw = demo["actions"][:].astype(np.float32)
    absolute = np.empty_like(raw)
    for step in range(len(states)):
        delta_env.reset_to(state_at(demo, states, step))
        robot = delta_env.env.robots[0]
        robot.control(raw[step], policy_step=True)
        controller = arm_controller(delta_env)
        absolute[step, :3] = controller.goal_pos
        absolute[step, 3:6] = Rotation.from_matrix(
            controller.goal_ori
        ).as_rotvec()
        absolute[step, 6] = raw[step, 6]
    return absolute


def validate_round_trip(delta_env, absolute_env, demo, absolute):
    states = demo["states"][:]
    raw = demo["actions"][:].astype(np.float32)
    sampled = sorted(set(range(0, len(states), 10)) | {len(states) - 1})
    max_position_error = 0.0
    max_orientation_error = 0.0
    max_gripper_error = 0.0
    for step in sampled:
        state = state_at(demo, states, step)
        delta_env.reset_to(state)
        absolute_env.reset_to(state)
        delta_robot = delta_env.env.robots[0]
        absolute_robot = absolute_env.env.robots[0]
        delta_robot.control(raw[step], policy_step=True)
        absolute_robot.control(absolute[step], policy_step=True)
        delta_controller = arm_controller(delta_env)
        absolute_controller = arm_controller(absolute_env)
        max_position_error = max(
            max_position_error,
            float(
                np.linalg.norm(
                    delta_controller.goal_pos - absolute_controller.goal_pos
                )
            ),
        )
        max_orientation_error = max(
            max_orientation_error,
            float(
                np.linalg.norm(
                    delta_controller.goal_ori - absolute_controller.goal_ori
                )
            ),
        )
        max_gripper_error = max(
            max_gripper_error,
            float(abs(raw[step, 6] - absolute[step, 6])),
        )
    passed = (
        max_position_error <= POSITION_TOLERANCE
        and max_orientation_error <= ORIENTATION_TOLERANCE
        and max_gripper_error <= GRIPPER_TOLERANCE
    )
    return {
        "sampled_steps": len(sampled),
        "max_position_error": max_position_error,
        "max_orientation_matrix_error": max_orientation_error,
        "max_gripper_error": max_gripper_error,
        "success": passed,
    }


def to_training_actions(absolute):
    result = absolute.copy()
    result[:, 6] = (1.0 - np.clip(result[:, 6], -1.0, 1.0)) / 2.0
    return result.astype(np.float32)


def worker(payload):
    dataset_path, demo_indices, validate_indices = payload
    delta_env, absolute_env = create_envs(dataset_path)
    converted = {}
    validation = {}
    with h5py.File(dataset_path, "r") as source:
        for demo_index in demo_indices:
            demo = source[f"data/demo_{demo_index}"]
            absolute = convert_demo(delta_env, demo)
            if demo_index in validate_indices:
                validation[demo_index] = validate_round_trip(
                    delta_env, absolute_env, demo, absolute
                )
            converted[demo_index] = to_training_actions(absolute)
    return converted, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-demos", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as source:
        num_demos = len(source["data"])
    chunks = [
        chunk.tolist()
        for chunk in np.array_split(np.arange(num_demos), args.workers)
        if len(chunk)
    ]
    payloads = [
        (str(input_path), chunk, set(args.validate_demos)) for chunk in chunks
    ]
    with mp.get_context("spawn").Pool(len(payloads)) as pool:
        results = pool.map(worker, payloads)
    shutil.copy2(input_path, output_path)
    validation = {}
    with h5py.File(output_path, "r+") as target:
        for converted, worker_validation in results:
            for demo_index, actions in converted.items():
                target[f"data/demo_{demo_index}/actions"][:] = actions
            validation.update(worker_validation)
        target["data"].attrs["table3_action_space"] = (
            "state_conditioned_world_eef_pos_axis_angle_plus_open_fraction"
        )
    quality_pass = all(item["success"] for item in validation.values())
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "num_demos": num_demos,
        "converter": (
            "recorded-state-conditioned delta goal generation with explicit "
            "absolute controller input"
        ),
        "quality_thresholds": {
            "position_error_max": POSITION_TOLERANCE,
            "orientation_matrix_error_max": ORIENTATION_TOLERANCE,
            "gripper_error_max": GRIPPER_TOLERANCE,
        },
        "validation": {
            str(key): value for key, value in sorted(validation.items())
        },
        "quality_pass": quality_pass,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not quality_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
