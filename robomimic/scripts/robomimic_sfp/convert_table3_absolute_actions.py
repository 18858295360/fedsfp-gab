#!/usr/bin/env python3
"""Convert RoboMimic delta actions to the SFP Table 3 action space.

The pose action is an absolute world-frame EEF setpoint. The gripper action is
represented as a continuous open fraction in [0, 1], where 1 is open.
"""

import argparse
import copy
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-demos", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def initialize_obs_utils():
    ObsUtils.initialize_obs_utils_with_config(config_factory("bc"))


def set_absolute_pose_controller(env_meta):
    result = copy.deepcopy(env_meta)
    controller = result["env_kwargs"]["controller_configs"]
    if "body_parts" in controller:
        controller = controller["body_parts"]["right"]
    controller["control_delta"] = False
    controller["input_type"] = "absolute"
    return result


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


def arm_controller(env):
    robot = env.env.robots[0]
    if hasattr(robot, "part_controllers"):
        return robot.part_controllers["right"]
    return robot.controller


def reset_arm_goal(env):
    controller = arm_controller(env)
    controller.update()
    controller.reset_goal(goal_update_mode="achieved")


def convert_demo(delta_env, states, actions):
    converted = np.empty_like(actions, dtype=np.float32)
    for idx, (state, action) in enumerate(zip(states, actions)):
        delta_env.reset_to({"states": state})
        reset_arm_goal(delta_env)
        controller = arm_controller(delta_env)
        abs_pose = controller.delta_to_abs_action(
            controller.scale_action(action[:6]), goal_update_mode="achieved"
        ).astype(np.float32)
        goal_pos = abs_pose[:3]
        goal_rot = abs_pose[3:6]
        open_fraction = np.float32((1.0 - np.clip(action[-1], -1.0, 1.0)) / 2.0)
        converted[idx] = np.concatenate((goal_pos, goal_rot, [open_fraction]))
    return converted


def quat_distance(q1, q2):
    return (Rotation.from_quat(q1) * Rotation.from_quat(q2).inv()).magnitude()


def validate_demo(delta_env, abs_env, states, actions, converted):
    pos_errors = []
    rot_errors = []
    state_errors = []
    for state, delta_action, abs_action in zip(states, actions, converted):
        delta_env.reset_to({"states": state})
        reset_arm_goal(delta_env)
        delta_obs, _, _, _ = delta_env.step(delta_action)

        env_abs_action = abs_action.copy()
        env_abs_action[-1] = 1.0 - 2.0 * np.clip(env_abs_action[-1], 0.0, 1.0)
        abs_env.reset_to({"states": state})
        reset_arm_goal(abs_env)
        abs_obs, _, _, _ = abs_env.step(env_abs_action)

        pos_errors.append(
            np.linalg.norm(delta_obs["robot0_eef_pos"] - abs_obs["robot0_eef_pos"])
        )
        rot_errors.append(
            quat_distance(delta_obs["robot0_eef_quat"], abs_obs["robot0_eef_quat"])
        )
        state_errors.append(
            np.max(
                np.abs(
                    delta_env.get_state()["states"] - abs_env.get_state()["states"]
                )
            )
        )
    return {
        "num_steps": len(actions),
        "max_eef_pos_error": float(np.max(pos_errors)),
        "mean_eef_pos_error": float(np.mean(pos_errors)),
        "max_eef_rot_error": float(np.max(rot_errors)),
        "mean_eef_rot_error": float(np.mean(rot_errors)),
        "max_state_error": float(np.max(state_errors)),
    }


def worker(payload):
    dataset_path, demo_indices, validate_indices = payload
    delta_env, abs_env = create_envs(dataset_path)
    converted = {}
    validation = {}
    with h5py.File(dataset_path, "r") as src:
        for demo_idx in demo_indices:
            demo = src[f"data/demo_{demo_idx}"]
            states = demo["states"][:]
            actions = demo["actions"][:].astype(np.float32)
            abs_actions = convert_demo(delta_env, states, actions)
            converted[demo_idx] = abs_actions
            if demo_idx in validate_indices:
                validation[demo_idx] = validate_demo(
                    delta_env, abs_env, states, actions, abs_actions
                )
    return converted, validation


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        num_demos = len(src["data"])
    all_indices = np.arange(num_demos)
    chunks = [chunk.tolist() for chunk in np.array_split(all_indices, args.workers) if len(chunk)]
    validate_indices = set(args.validate_demos)
    payloads = [(str(input_path), chunk, validate_indices) for chunk in chunks]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(payloads)) as pool:
        results = pool.map(worker, payloads)

    shutil.copy2(input_path, output_path)
    validation = {}
    with h5py.File(output_path, "r+") as dst:
        for converted, worker_validation in results:
            for demo_idx, actions in converted.items():
                dst[f"data/demo_{demo_idx}/actions"][:] = actions
            validation.update(worker_validation)
        dst["data"].attrs["table3_action_space"] = (
            "world_eef_pos_axis_angle_plus_continuous_gripper_open_fraction"
        )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "num_demos": num_demos,
        "action_space": "world EEF position + axis-angle + gripper open fraction [0,1]",
        "validation": {str(key): value for key, value in sorted(validation.items())},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
