from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


OBS_KEYS = (
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def axis_angle_to_rotation_6d(rotvec: np.ndarray) -> np.ndarray:
    matrix = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    return matrix[:, :2, :].reshape(*rotvec.shape[:-1], 6).astype(np.float32)


def rotation_6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    original_shape = rot6d.shape[:-1]
    vectors = rot6d.reshape(-1, 2, 3)
    b1 = vectors[:, 0] / np.maximum(np.linalg.norm(vectors[:, 0], axis=-1, keepdims=True), 1e-8)
    a2 = vectors[:, 1]
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    matrix = np.stack((b1, b2, b3), axis=-2)
    return Rotation.from_matrix(matrix).as_rotvec().reshape(*original_shape, 3).astype(np.float32)


def raw7_to_model10(actions: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (actions[..., :3], axis_angle_to_rotation_6d(actions[..., 3:6]), actions[..., 6:7]),
        axis=-1,
    ).astype(np.float32)


def model10_to_env7(actions: np.ndarray) -> np.ndarray:
    gripper_command = 1.0 - 2.0 * np.clip(actions[..., 9:10], 0.0, 1.0)
    return np.concatenate(
        (actions[..., :3], rotation_6d_to_axis_angle(actions[..., 3:9]), gripper_command),
        axis=-1,
    ).astype(np.float32)


def obs_vector(obs: Dict[str, np.ndarray], keys: Sequence[str] = OBS_KEYS) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys],
        axis=-1,
    )


@dataclass
class Table3Normalizer:
    obs_scale: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray

    @classmethod
    def fit(cls, observations: np.ndarray, actions: np.ndarray):
        obs_max_abs = max(float(np.max(np.abs(observations))), 1e-7)
        obs_scale = np.full(observations.shape[-1], 1.0 / obs_max_abs, dtype=np.float32)
        action_min = actions.min(axis=0)
        action_max = actions.max(axis=0)
        action_scale = np.ones(actions.shape[-1], dtype=np.float32)
        action_offset = np.zeros(actions.shape[-1], dtype=np.float32)
        action_range = np.maximum(action_max[:3] - action_min[:3], 1e-7)
        action_scale[:3] = 2.0 / action_range
        action_offset[:3] = -1.0 - action_scale[:3] * action_min[:3]
        return cls(obs_scale, action_scale, action_offset)

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            np.asarray(state["obs_scale"], dtype=np.float32),
            np.asarray(state["action_scale"], dtype=np.float32),
            np.asarray(state["action_offset"], dtype=np.float32),
        )

    def state_dict(self):
        return {
            "kind": "robomimic_table3_abs",
            "obs_scale": self.obs_scale,
            "action_scale": self.action_scale,
            "action_offset": self.action_offset,
        }

    def normalize_obs(self, obs):
        return np.asarray(obs, dtype=np.float32) * self.obs_scale

    def normalize_action(self, action):
        return np.asarray(action, dtype=np.float32) * self.action_scale + self.action_offset

    def unnormalize_action(self, action):
        return (np.asarray(action, dtype=np.float32) - self.action_offset) / self.action_scale


class Table3PaperGridDPDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        sigma0: float = 0.4,
        stabilization_k: float = 15.0,
        integration_steps_per_action: int = 6,
        val_ratio: float = 0.02,
        split_seed: int = 42,
        max_train_demos: Optional[int] = None,
        normalizer: Optional[Table3Normalizer] = None,
        gripper_source: str = "next_qpos",
    ):
        self.dataset_path = dataset_path
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.sigma0 = float(sigma0)
        self.k = float(stabilization_k)
        self.integration_steps_per_action = int(integration_steps_per_action)
        if gripper_source not in {"next_qpos", "binary_command"}:
            raise ValueError(f"unsupported gripper_source: {gripper_source}")
        self.gripper_source = gripper_source
        self.episodes = self._load()

        rng = np.random.default_rng(split_seed)
        indices = np.arange(len(self.episodes))
        rng.shuffle(indices)
        num_val = int(round(len(indices) * val_ratio))
        train_indices = indices[num_val:]
        if max_train_demos is not None:
            train_indices = train_indices[:max_train_demos]
        self.train_indices = sorted(int(idx) for idx in train_indices)

        train_episodes = [self.episodes[idx] for idx in self.train_indices]
        all_obs = np.concatenate([episode["obs"] for episode in train_episodes], axis=0)
        all_actions = np.concatenate([episode["action"] for episode in train_episodes], axis=0)
        self.normalizer = normalizer or Table3Normalizer.fit(all_obs, all_actions)
        self.obs_dim = all_obs.shape[-1]
        self.action_dim = all_actions.shape[-1]
        self.indices: List[Tuple[int, int]] = []
        for episode_idx in self.train_indices:
            for step in range(len(self.episodes[episode_idx]["action"])):
                self.indices.append((episode_idx, step))

    def _load(self):
        episodes = []
        with h5py.File(self.dataset_path, "r") as source:
            names = sorted(source["data"], key=lambda name: int(name.split("_")[-1]))
            for name in names:
                demo = source[f"data/{name}"]
                obs = np.concatenate(
                    [demo[f"obs/{key}"][:].astype(np.float32) for key in OBS_KEYS],
                    axis=-1,
                )
                raw_action = demo["actions"][:].astype(np.float32)
                if self.gripper_source == "next_qpos":
                    next_gripper_qpos = demo["next_obs/robot0_gripper_qpos"][:].astype(np.float32)
                    open_fraction = np.clip(
                        np.mean(np.abs(next_gripper_qpos), axis=-1, keepdims=True) / 0.04,
                        0.0,
                        1.0,
                    )
                else:
                    # The DP-converted HDF5 already stores the raw binary
                    # command as open fraction in {0, 1}.
                    open_fraction = np.clip(raw_action[:, 6:7], 0.0, 1.0)
                raw_action[:, 6:7] = open_fraction
                action = raw7_to_model10(raw_action)
                episodes.append({"obs": obs, "action": action})
        return episodes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode_idx, current = self.indices[index]
        episode = self.episodes[episode_idx]
        obs_ids = np.clip(
            np.arange(current - self.obs_horizon + 1, current + 1),
            0,
            len(episode["obs"]) - 1,
        )
        action_ids = np.clip(
            np.arange(current - 1, current + self.pred_horizon),
            0,
            len(episode["action"]) - 1,
        )
        obs = self.normalizer.normalize_obs(episode["obs"][obs_ids])
        trajectory = self.normalizer.normalize_action(episode["action"][action_ids])

        t = np.float32(np.random.rand())
        scaled_t = t * self.pred_horizon
        segment = min(int(np.floor(scaled_t)), self.pred_horizon - 1)
        alpha = np.float32(scaled_t - segment)
        xi_t = trajectory[segment] + alpha * (trajectory[segment + 1] - trajectory[segment])
        xi_dot = (trajectory[segment + 1] - trajectory[segment]) * np.float32(self.pred_horizon)

        sigma_t = self.sigma0 * np.exp(-self.k * float(t))
        action_t = xi_t + np.float32(sigma_t) * np.random.randn(self.action_dim).astype(np.float32)
        velocity = xi_dot - np.float32(self.k) * (action_t - xi_t)

        return {
            "obs": torch.from_numpy(obs.astype(np.float32)),
            "a": torch.from_numpy(action_t.astype(np.float32)),
            "t": torch.tensor(t, dtype=torch.float32),
            "v": torch.from_numpy(velocity.astype(np.float32)),
        }
