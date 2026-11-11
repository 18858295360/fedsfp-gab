"""Image-conditioned SFP with the encoder robomimic manipulation policies use.

Three changes from the first version, in decreasing order of how wrong the old
one was:

1. **One encoder per camera.** The old model concatenated agentview and
   robot0_eye_in_hand on the channel axis and ran a single conv stack over the
   six channels. A filter that is meaningful for a wrist camera is meaningless
   for a third-person view; they do not share geometry, and sharing kernels
   across them is a modelling error rather than a parameter saving.

2. **Spatial softmax instead of flatten + linear.** Keypoint pooling returns
   explicit (x, y) locations of salient features, which is the representation a
   nut-insertion task needs -- where the nut is, where the peg is. Flattening a
   feature map into a linear layer discards that structure and has to relearn
   it. This is the standard robomimic manipulation encoder.

3. **ResNet-18 with GroupNorm.** The old stack was four conv layers trained
   from scratch. GroupNorm rather than BatchNorm because federated clients hold
   different data and batch statistics do not transfer across aggregation --
   BatchNorm running stats are averaged into nonsense.

The flow backbone is unchanged: the UNet consumes a per-timestep conditioning
vector, so only the width of that vector changes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from streaming_flow_policy.robomimic_sfp.model_v22_paper_aligned import SFPConditionalUnet1D


def replace_bn_with_gn(module, groups=16):
    """BatchNorm cannot survive federated averaging: each client accumulates
    running statistics over its own distribution, and the server averages them
    into something that describes no client. GroupNorm has no running state."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name,
                    nn.GroupNorm(num_groups=min(groups, child.num_features),
                                 num_channels=child.num_features))
        else:
            replace_bn_with_gn(child, groups)
    return module


class CameraEncoder(nn.Module):
    """ResNet-18 trunk -> spatial softmax keypoints, for one camera."""

    def __init__(self, in_ch=3, num_kp=32, img_size=84, pretrained=False):
        super().__init__()
        from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
        self.backbone = ResNet18Conv(input_channel=in_ch, pretrained=pretrained,
                                     input_coord_conv=False)
        replace_bn_with_gn(self.backbone)
        shape = self.backbone.output_shape([in_ch, img_size, img_size])
        self.pool = SpatialSoftmax(input_shape=shape, num_kp=num_kp,
                                   temperature=1.0, learnable_temperature=False)
        self.out_dim = num_kp * 2

    def forward(self, x):            # x: [N, C, H, W] in [-0.5, 0.5]
        return self.pool(self.backbone(x)).flatten(1)


class SFPImagePolicyV2(nn.Module):
    def __init__(self, action_dim, obs_horizon=2, proprio_dim=23, num_cameras=2,
                 num_kp=32, img_size=84, cam_channels=3, pretrained=False,
                 down_dims=(256, 512, 1024), diffusion_step_embed_dim=256,
                 fc_timesteps=1, lang_dim=0, lang_proj_dim=64):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.proprio_dim = proprio_dim
        self.num_cameras = num_cameras
        self.cam_channels = cam_channels
        self.encoders = nn.ModuleList([
            CameraEncoder(cam_channels, num_kp, img_size, pretrained)
            for _ in range(num_cameras)])
        feat = sum(e.out_dim for e in self.encoders)
        # The instruction enters as a learned projection of a frozen sentence
        # embedding rather than as a task index, so the conditioning stays
        # defined for an instruction the model has not seen.
        self.lang_dim = int(lang_dim)
        self.lang_proj_dim = int(lang_proj_dim) if self.lang_dim > 0 else 0
        self.lang_proj = (nn.Sequential(nn.Linear(self.lang_dim, self.lang_proj_dim),
                                        nn.SiLU())
                          if self.lang_dim > 0 else None)
        self.unet = SFPConditionalUnet1D(
            obs_dim=feat + proprio_dim + self.lang_proj_dim, action_dim=action_dim,
            obs_horizon=obs_horizon, down_dims=list(down_dims),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            fc_timesteps=fc_timesteps)

    def encode_obs(self, imgs, proprio):
        """The conditioning vector the policy is built on: per-camera features
        concatenated with proprioception. Exposed separately so MOON can compare
        it across models without running the flow backbone."""
        B, H = imgs.shape[:2]
        x = imgs.reshape(B * H, *imgs.shape[2:]).permute(0, 3, 1, 2)
        feats = [enc(x[:, i * self.cam_channels:(i + 1) * self.cam_channels])
                 for i, enc in enumerate(self.encoders)]
        z = torch.cat(feats, dim=-1).reshape(B, H, -1)
        return torch.cat([z, proprio], dim=-1)

    def forward(self, imgs, proprio, action, t, lang=None):
        """imgs: [B, H, 84, 84, C*num_cameras] -- cameras stacked on channels by
        the dataset, split apart here so each gets its own encoder.

        lang is the sentence embedding of the task instruction, one vector per
        sample, broadcast over the observation history. It is joined after
        encode_obs so MOON's contrastive term keeps comparing vision alone."""
        obs = self.encode_obs(imgs, proprio)
        if self.lang_proj is not None:
            if lang is None:
                raise ValueError(
                    "this policy was built with lang_dim=%d but forward got no "
                    "lang; training with the instruction and evaluating without "
                    "it scores zero for no visible reason" % self.lang_dim)
            l = self.lang_proj(lang)
            obs = torch.cat([obs, l.unsqueeze(1).expand(-1, obs.shape[1], -1)], dim=-1)
        return self.unet(obs, action, t)
