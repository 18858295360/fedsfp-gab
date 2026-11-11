#!/usr/bin/env python3
import argparse
import os as _sfp_os
import copy
import json
import random
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from streaming_flow_policy.robomimic_sfp.model_v21 import SFPConditionalUnet1D
from streaming_flow_policy.robomimic_sfp.table3_papergrid_dp_v1 import (
    OBS_KEYS,
    Table3Normalizer,
    model10_to_env7,
    raw7_to_model10,
    obs_vector,
)
from scripts.robomimic_sfp.convert_table3_absolute_actions import arm_controller


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-rollouts", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--carry-previous-action", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trace-rollout", type=int, default=None)
    return parser.parse_args()


def absolute_env_meta(path):
    meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(path))
    controller = meta["env_kwargs"]["controller_configs"]
    if "body_parts" in controller:
        controller = controller["body_parts"]["right"]
    controller["control_delta"] = False
    controller["input_type"] = "absolute"
    return meta


def current_model_action(env, gripper_source):
    controller = arm_controller(env)
    controller.update()
    controller.reset_goal(goal_update_mode="achieved")
    pose = controller.delta_to_abs_action(
        controller.scale_action(np.zeros(6, dtype=np.float32)),
        goal_update_mode="achieved",
    ).astype(np.float32)
    qpos = np.asarray(env.get_observation()["robot0_gripper_qpos"], dtype=np.float32)
    open_fraction = float(np.clip(np.mean(np.abs(qpos)) / 0.04, 0.0, 1.0))
    if gripper_source == "binary_command":
        open_fraction = float(open_fraction >= 0.5)
    return raw7_to_model10(np.concatenate((pose, [open_fraction]))[None])[0]


def load_model(checkpoint, device):
    config = checkpoint["model_config"]
    model = SFPConditionalUnet1D(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        obs_horizon=checkpoint["obs_horizon"],
        diffusion_step_embed_dim=config["diffusion_step_embed_dim"],
        down_dims=config["down_dims"],
        kernel_size=config["kernel_size"],
        n_groups=config["n_groups"],
        fc_timesteps=config["fc_timesteps"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if "ema_model_state" in checkpoint:
        model.load_state_dict(checkpoint["ema_model_state"])
    model.eval()
    return model


@torch.inference_mode()
def predict_chunk(model, normalizer, history, previous, horizon, pred_horizon, substeps, device):
    obs = torch.from_numpy(normalizer.normalize_obs(history)).unsqueeze(0).to(device)
    action = torch.from_numpy(normalizer.normalize_action(previous)).unsqueeze(0).to(device)
    dt = 1.0 / (pred_horizon * substeps)
    outputs = [normalizer.unnormalize_action(action[0].cpu().numpy())]
    # Algorithm 2 executes point 0 before integrating. The point after H
    # intervals is carried as the next chunk's initial action.
    for step in range(horizon * substeps):
        t = torch.full((1,), step * dt, dtype=torch.float32, device=device)
        action = action + dt * model(obs, action, t)
        action[:, -1].clamp_(0.0, 1.0)
        if (step + 1) % substeps == 0:
            outputs.append(normalizer.unnormalize_action(action[0].cpu().numpy()))
    return np.stack(outputs[:-1]), normalizer.unnormalize_action(action[0].cpu().numpy())


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.start_seed)
    np.random.seed(args.start_seed)
    torch.manual_seed(args.start_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    table3 = checkpoint.get("table3", {})
    if table3.get("dataset_variant") != "square_mh_dp_absolute_standard_foh_gripper_v1":
        raise ValueError("checkpoint is not a standard-FOH-gripper run")
    gripper_source = table3.get("gripper_source")
    if gripper_source not in {"next_qpos", "binary_command"}:
        raise ValueError("checkpoint has an unsupported gripper source")
    normalizer = Table3Normalizer.from_state_dict(checkpoint["normalizer"])
    model = load_model(checkpoint, device)
    substeps = int(_sfp_os.environ.get("SFP_SUBSTEPS_OVERRIDE", checkpoint["table3"]["integration_steps_per_action"]))

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": list(OBS_KEYS), "rgb": []}}
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=absolute_env_meta(args.dataset_path),
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    results = []
    for rollout_idx in range(args.num_rollouts):
        seed = args.start_seed + rollout_idx
        # RoboSuite 1.5 exposes its seed as a constructor field rather than a
        # callable method. Its randomized reset path uses NumPy's global RNG.
        random.seed(seed)
        np.random.seed(seed)
        env.reset()
        # RoboMimic Square reset returns stale object[:7]; refresh live observation before history seeding.
        obs = env.get_observation()
        history = deque(maxlen=checkpoint["obs_horizon"])
        first_obs = obs_vector(obs)
        for _ in range(checkpoint["obs_horizon"]):
            history.append(first_obs)
        max_reward = 0.0
        success = False
        steps = 0
        trace = []
        carried_previous = None
        while steps < args.max_steps and not success:
            previous = (
                carried_previous
                if args.carry_previous_action and carried_previous is not None
                else current_model_action(env, gripper_source)
            )
            chunk, next_previous = predict_chunk(
                model,
                normalizer,
                np.stack(history),
                previous,
                args.action_horizon,
                checkpoint["pred_horizon"],
                substeps,
                device,
            )
            if args.carry_previous_action:
                carried_previous = next_previous
            for model_action in chunk:
                env_action = model10_to_env7(model_action[None])[0]
                obs, reward, done, info = env.step(env_action)
                history.append(obs_vector(obs))
                max_reward = max(max_reward, float(reward))
                success_info = info.get("is_success", {})
                success = bool(success_info.get("task", False)) or bool(max_reward > 0.5)
                steps += 1
                if args.trace_rollout == rollout_idx:
                    trace.append({
                        "step": steps,
                        "anchor": previous.tolist(),
                        "model_action": model_action.tolist(),
                        "env_action": env_action.tolist(),
                        "eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                        "reward": float(reward),
                    })
                if success or done or steps >= args.max_steps:
                    break
        result = {
            "rollout": rollout_idx,
            "seed": seed,
            "score": max_reward,
            "success": success,
            "steps": steps,
            "trace": trace,
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    scores = np.asarray([result["score"] for result in results], dtype=np.float32)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_rollouts": len(results),
        "start_seed": args.start_seed,
        "action_horizon": args.action_horizon,
        "carry_previous_action": args.carry_previous_action,
        "integration_steps_per_action": substeps,
        "trajectory_grid_points": checkpoint["pred_horizon"] + 1,
        "trajectory_grid_intervals": checkpoint["pred_horizon"],
        "execution_slice": "points_0_through_H_minus_1",
        "gripper_source": gripper_source,
        "mean_score": float(scores.mean()),
        "success_rate": float(np.mean([result["success"] for result in results])),
        "results": results,
    }
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
