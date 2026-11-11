"""Paper-grid dataset with targeted sampling of rare gripper switch segments."""

import numpy as np
import torch

from streaming_flow_policy.robomimic_sfp.table3_papergrid_dp_v1 import (
    Table3PaperGridDPDataset,
)


class Table3PaperGridSwitchAwareDataset(Table3PaperGridDPDataset):
    def __init__(self, *args, switch_probability=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.switch_probability = float(switch_probability)

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
        switches = np.flatnonzero(trajectory[:-1, -1] != trajectory[1:, -1])
        if len(switches) and np.random.rand() < self.switch_probability:
            segment = int(switches[0])
            t = np.float32((segment + np.random.rand()) / self.pred_horizon)

        scaled_t = t * self.pred_horizon
        segment = min(int(np.floor(scaled_t)), self.pred_horizon - 1)
        alpha = np.float32(scaled_t - segment)
        xi_t = trajectory[segment] + alpha * (
            trajectory[segment + 1] - trajectory[segment]
        )
        xi_dot = (trajectory[segment + 1] - trajectory[segment]) * np.float32(
            self.pred_horizon
        )
        sigma_t = self.sigma0 * np.exp(-self.k * float(t))
        # global-anchor broadening: with probability frac, centre the
        # noise draw on the round-start global's generated action at this
        # sample's flow-time segment instead of on the expert. The noise
        # radius is unchanged and the velocity target below still
        # restores toward the real xi_t.
        _anchor = xi_t
        _gt = getattr(self, 'global_anchor_table', None)
        _gf = float(getattr(self, 'global_anchor_frac', 0.0) or 0.0)
        if _gt is not None and _gf > 0.0 and np.random.rand() < _gf:
            _anchor = np.asarray(_gt[index, segment], dtype=np.float32)
        action_t = _anchor + np.float32(sigma_t) * np.random.randn(
            self.action_dim
        ).astype(np.float32)
        velocity = xi_dot - np.float32(self.k) * (action_t - xi_t)

        return {
            "obs": torch.from_numpy(obs.astype(np.float32)),
            "a": torch.from_numpy(action_t.astype(np.float32)),
            "t": torch.tensor(t, dtype=torch.float32),
            "v": torch.from_numpy(velocity.astype(np.float32)),
        }
