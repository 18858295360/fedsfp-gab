"""Paper-aligned time conditioning for the RoboMimic SFP velocity field.

The released SFP implementation passes continuous flow time t in [0, 1]
directly to ConditionalUnet1D. Historical local runs multiplied t by 100.
This module is intentionally separate so old checkpoints keep their original
forward semantics.
"""

import torch

from streaming_flow_policy.robomimic_sfp.model_v21 import (
    SFPConditionalUnet1D as _LegacySFPConditionalUnet1D,
)


class SFPConditionalUnet1D(_LegacySFPConditionalUnet1D):
    """ConditionalUnet1D using the released SFP continuous-time convention."""

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = obs.flatten(start_dim=1)
        sample = action.unsqueeze(1)
        pred = self.velocity_net(
            sample=sample,
            timestep=t,
            global_cond=obs_cond,
        )
        return pred.squeeze(1)
