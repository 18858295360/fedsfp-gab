import math
from typing import Sequence

import torch
import torch.nn as nn

from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D


class FourierTimeEmbedding(nn.Module):
    def __init__(self, num_frequencies: int = 8):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        frequencies = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)

    @property
    def dim(self) -> int:
        return 1 + 2 * self.num_frequencies

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1)
        angles = math.pi * t * self.frequencies.reshape(1, -1)
        return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)


class SFPMLP(nn.Module):
    """MLP velocity field v_theta(a, t | h) for low-dimensional RoboMimic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int = 2,
        hidden_dims: Sequence[int] = (512, 512, 512),
        time_frequencies: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.time_embed = FourierTimeEmbedding(num_frequencies=time_frequencies)

        input_dim = self.obs_dim * self.obs_horizon + self.action_dim + self.time_embed.dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        obs_flat = obs.flatten(start_dim=1)
        t_emb = self.time_embed(t)
        x = torch.cat([obs_flat, action, t_emb], dim=-1)
        return self.net(x)


class EEFBridgeMLP(nn.Module):
    """Supervised bridge from current/target EEF state to RoboMimic action."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        hidden_dims: Sequence[int] = (512, 512, 512),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplanGateMLP(nn.Module):
    """Binary gate that predicts whether to discard the current EEF target chunk."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
