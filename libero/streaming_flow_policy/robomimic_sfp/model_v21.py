from typing import Sequence

import torch
import torch.nn as nn

from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D


class SFPConditionalUnet1D(nn.Module):
    """Official ConditionalUnet1D velocity field adapted to v_theta(a, t | h)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int = 2,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        fc_timesteps: int = 1,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.diffusion_step_embed_dim = int(diffusion_step_embed_dim)
        self.down_dims = tuple(int(dim) for dim in down_dims)
        self.kernel_size = int(kernel_size)
        self.n_groups = int(n_groups)
        self.fc_timesteps = int(fc_timesteps)
        self.velocity_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon,
            diffusion_step_embed_dim=self.diffusion_step_embed_dim,
            down_dims=list(self.down_dims),
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            fc_timesteps=self.fc_timesteps,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        obs_cond = obs.flatten(start_dim=1)
        sample = action.unsqueeze(1)
        pred = self.velocity_net(sample=sample, timestep=t * 100.0, global_cond=obs_cond)
        return pred.squeeze(1)
