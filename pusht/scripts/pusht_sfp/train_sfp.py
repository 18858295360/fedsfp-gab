import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streaming_flow_policy.pusht.dataset import (
    PushTImageStateDataset,
    PushTStateActionDataset,
    PushTStateDataset,
    PushTStateDatasetWithNextObsAsAction,
)
from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D, make_pusht_image_encoder
from streaming_flow_policy.pusht.sfpd import StreamingFlowPolicyDeterministic
from streaming_flow_policy.pusht.sfps import StreamingFlowPolicyStochastic


class SimpleEMA:
    def __init__(self, parameters, decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def step(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        for shadow, param in zip(self.shadow, params):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        for param, shadow in zip(params, self.shadow):
            param.copy_(shadow)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Push-T SFP baseline.")
    parser.add_argument("--model-type", choices=["sfpd", "sfps"], required=True)
    parser.add_argument("--imitation-mode", choices=["state", "action"], default="state", help="state uses next pusher state as target; action uses raw dataset env actions.")
    parser.add_argument("--obs-mode", choices=["state", "image"], default="state", help="Observation modality used for policy conditioning.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=0.1, help="SFPD training position noise.")
    parser.add_argument("--sigma0", type=float, default=0.1, help="SFPS sigma0.")
    parser.add_argument("--sigma1", type=float, default=0.1, help="SFPS sigma1.")
    parser.add_argument("--use-stabilized-objective", action="store_true", help="Use the paper stabilized SFPD velocity target.")
    parser.add_argument("--stabilization-k", type=float, default=1.0, help="Stabilizing gain k for the SFPD objective.")
    parser.add_argument("--use-time-decayed-noise", action="store_true", help="Use sigma_t = sigma * exp(-k*t) for stabilized SFPD sampled states.")
    parser.add_argument("--debug-stabilized-objective", action="store_true", help="Print one-batch SFPD target statistics.")
    parser.add_argument("--debug-dataset-info", action="store_true", help="Print one dataset/sample summary before training.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional cap on optimizer steps for smoke tests.")
    parser.add_argument("--init-ckpt-path", type=Path, default=None, help="Optional checkpoint used to initialize model weights before training.")
    parser.add_argument("--init-load-velocity-only", action="store_true", help="Load only velocity_net weights from --init-ckpt-path, leaving current config buffers such as sigma intact.")
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--image-cond-dim", type=int, default=256)
    parser.add_argument("--image-encoder", type=str, default="small_avgpool", help="Push-T image encoder variant: e0/small_avgpool, e1/conv_flatten, e2/spatial_softmax, e3/resnet18_spatial, e4/resnet18_spatial_softmax, final_v1/resnet18_gn_spatial_softmax.")
    parser.add_argument("--image-agent-pos-cond", action="store_true", help="For image observations, concatenate normalized agent_pos history to the image condition.")
    parser.add_argument("--image-key", type=str, default="img")
    parser.add_argument("--image-random-crop-size", type=int, default=None, help="Optional square random crop size for image training, e.g. 84 for 96x96 Push-T frames.")
    parser.add_argument("--episode-indices-json", type=Path, default=None, help="Optional JSON file containing episode-index lists.")
    parser.add_argument("--episode-indices-key", type=str, default=None, help="Key in --episode-indices-json used as the training subset.")
    parser.add_argument("--stats-episode-indices-key", type=str, default=None, help="Key in --episode-indices-json used for normalizer stats.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay used for saved evaluation checkpoints.")
    return parser.parse_args()


def build_policy(args):
    if args.image_agent_pos_cond and args.obs_mode != "image":
        raise ValueError("--image-agent-pos-cond is only valid with --obs-mode image.")

    obs_dim = 5
    action_dim = 2
    fc_timesteps = 1 if args.model_type == "sfpd" else 2
    obs_encoder = None
    agent_pos_condition_dim = 0
    global_cond_dim = obs_dim * args.obs_horizon
    if args.obs_mode == "image":
        obs_encoder = make_pusht_image_encoder(
            variant=args.image_encoder,
            obs_horizon=args.obs_horizon,
            feature_dim=args.image_cond_dim,
        )
        agent_pos_condition_dim = 2 * args.obs_horizon if args.image_agent_pos_cond else 0
        global_cond_dim = args.image_cond_dim + agent_pos_condition_dim

    args.agent_pos_condition_dim = agent_pos_condition_dim
    args.global_cond_dim = global_cond_dim

    velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        fc_timesteps=fc_timesteps,
    )
    velocity_net.to(args.device)
    if obs_encoder is not None:
        obs_encoder.to(args.device)

    if args.model_type == "sfpd":
        return StreamingFlowPolicyDeterministic(
            velocity_net=velocity_net,
            action_dim=action_dim,
            pred_horizon=args.pred_horizon,
            sigma=args.sigma,
            use_stabilized_objective=args.use_stabilized_objective,
            stabilization_k=args.stabilization_k,
            use_time_decayed_noise=args.use_time_decayed_noise,
            debug_stabilized_objective=args.debug_stabilized_objective,
            start_mode="state" if args.imitation_mode == "state" else "zero",
            obs_encoder=obs_encoder,
            agent_pos_condition_dim=agent_pos_condition_dim,
            device=args.device,
        )
    return StreamingFlowPolicyStochastic(
        velocity_net=velocity_net,
        action_dim=action_dim,
        σ0=args.sigma0,
        σ1=args.sigma1,
        pred_horizon=args.pred_horizon,
        use_stabilized_objective=args.use_stabilized_objective,
        stabilization_k=args.stabilization_k,
        use_time_decayed_noise=args.use_time_decayed_noise,
        debug_stabilized_objective=args.debug_stabilized_objective,
        start_mode="state" if args.imitation_mode == "state" else "zero",
        obs_encoder=obs_encoder,
        agent_pos_condition_dim=agent_pos_condition_dim,
        device=args.device,
    )


def _extract_state_dict(raw_checkpoint):
    if isinstance(raw_checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model_state", "policy_state_dict"):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return raw_checkpoint


def load_initial_checkpoint(policy, init_ckpt_path: Path, *, device: str, velocity_only: bool):
    raw_checkpoint = torch.load(init_ckpt_path, map_location=device)
    state_dict = _extract_state_dict(raw_checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint {init_ckpt_path} did not contain a state dict")

    if velocity_only:
        velocity_state = {
            key.removeprefix("velocity_net."): value
            for key, value in state_dict.items()
            if key.startswith("velocity_net.")
        }
        if not velocity_state:
            raise ValueError(f"No velocity_net.* weights found in {init_ckpt_path}")
        policy.velocity_net.load_state_dict(velocity_state, strict=True)
        print(f"Loaded velocity_net weights from {init_ckpt_path}")
    else:
        policy.load_state_dict(state_dict, strict=True)
        print(f"Loaded full policy state from {init_ckpt_path}")




def _load_episode_indices(args):
    if args.episode_indices_json is None:
        return None, None
    with args.episode_indices_json.open() as f:
        split_data = json.load(f)

    def read_key(key):
        if key is None:
            return None
        if key not in split_data:
            raise KeyError(f"{key!r} not found in {args.episode_indices_json}")
        return [int(x) for x in split_data[key]]

    return read_key(args.episode_indices_key), read_key(args.stats_episode_indices_key)


def build_dataset(args, policy):
    episode_indices, stats_episode_indices = _load_episode_indices(args)
    if args.obs_mode == "image":
        return PushTImageStateDataset(
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            transform_datum_fn=policy.TransformTrainingDatum,
            image_key=args.image_key,
            action_target_mode="next_state" if args.imitation_mode == "state" else "action",
            episode_indices=episode_indices,
            stats_episode_indices=stats_episode_indices,
            image_random_crop_size=args.image_random_crop_size,
        )
    dataset_cls = (
        PushTStateDatasetWithNextObsAsAction
        if args.imitation_mode == "state"
        else PushTStateActionDataset
    )
    return dataset_cls(
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        transform_datum_fn=policy.TransformTrainingDatum,
        episode_indices=episode_indices,
        stats_episode_indices=stats_episode_indices,
    )


def _unnormalize_np(ndata, stats):
    return ((ndata + 1.0) / 2.0) * (stats["max"] - stats["min"]) + stats["min"]


def print_dataset_debug(args, dataset):
    sample = dataset[0]
    obs = sample["obs"]
    action = sample.get("action")
    print("[dataset debug] imitation_mode", args.imitation_mode, flush=True)
    print("[dataset debug] sample keys", sorted(sample.keys()), flush=True)
    print("[dataset debug] obs shape", getattr(obs, "shape", None), "action target shape", getattr(action, "shape", None), flush=True)
    for key in ("obs", "action"):
        if key not in dataset.stats:
            continue
        stats = dataset.stats[key]
        print(f"[dataset debug] {key} normalizer min={stats['min']} max={stats['max']}", flush=True)
    state_obs = sample.get("state_obs", obs)
    raw_obs_last = _unnormalize_np(state_obs[-1], dataset.stats["obs"])
    if getattr(obs, "ndim", 0) == 4:
        print("[dataset debug] image obs min/max/mean", float(np.min(obs)), float(np.max(obs)), float(np.mean(obs)), flush=True)
        print("[dataset debug] state_obs shape", state_obs.shape, flush=True)
    print("[dataset debug] normalized last obs[:2]", state_obs[-1, :2], flush=True)
    print("[dataset debug] raw last obs[:2]", raw_obs_last[:2], flush=True)
    if action is not None:
        raw_action0 = _unnormalize_np(action[0], dataset.stats["action"])
        print("[dataset debug] normalized first target", action[0], flush=True)
        print("[dataset debug] raw first target", raw_action0, flush=True)
        print("[dataset debug] target differs from state-start", bool(not np.allclose(action[0], state_obs[-1, :2])), flush=True)
    if "x" in sample:
        print("[dataset debug] transformed x shape", sample["x"].shape, "v shape", sample["v"].shape, flush=True)
        print("[dataset debug] transformed xi shape", sample.get("xi", np.empty(())).shape, flush=True)
        print("[dataset debug] transformed x mean/std/min/max", float(np.mean(sample["x"])), float(np.std(sample["x"])), float(np.min(sample["x"])), float(np.max(sample["x"])), flush=True)
        print("[dataset debug] transformed v mean/std/min/max", float(np.mean(sample["v"])), float(np.std(sample["v"])), float(np.min(sample["v"])), float(np.max(sample["v"])), flush=True)
    if "a" in sample:
        print("[dataset debug] transformed a shape", sample["a"].shape, "va shape", sample["va"].shape, "vz shape", sample["vz"].shape, flush=True)


def save_checkpoint(policy, save_dir: Path, epoch: int, loss_history, ema: SimpleEMA | None = None):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"checkpoint_epoch_{epoch:04d}.pth"
    latest_path = save_dir / "latest.pth"
    backup = None
    if ema is not None:
        params = [p for p in policy.parameters() if p.requires_grad]
        backup = [p.detach().clone() for p in params]
        ema.copy_to(policy.parameters())
    torch.save(policy.state_dict(), ckpt_path)
    torch.save(policy.state_dict(), latest_path)
    if backup is not None:
        for param, old_param in zip(params, backup):
            param.data.copy_(old_param)
    with (save_dir / "loss_history.json").open("w") as f:
        json.dump(loss_history, f, indent=2)


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = build_policy(args)
    if args.init_ckpt_path is not None:
        load_initial_checkpoint(
            policy,
            args.init_ckpt_path,
            device=args.device,
            velocity_only=args.init_load_velocity_only,
        )

    dataset = build_dataset(args, policy)
    if args.debug_dataset_info:
        print_dataset_debug(args, dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    ema = SimpleEMA(policy.parameters(), decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        params=policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_train_steps = len(dataloader) * args.epochs
    if args.max_train_steps is not None:
        total_train_steps = min(total_train_steps, args.max_train_steps)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_train_steps,
    )

    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata["dataset_len"] = len(dataset)
    metadata["num_batches"] = len(dataloader)
    metadata["image_action_target_mode"] = getattr(dataset, "action_target_mode", None)
    metadata["agent_pos_condition_dim"] = getattr(args, "agent_pos_condition_dim", 0)
    metadata["global_cond_dim"] = getattr(args, "global_cond_dim", None)
    metadata["ema_type"] = "SimpleEMA"
    metadata["ema_decay"] = args.ema_decay
    metadata["checkpoint_weights"] = "ema_shadow"
    metadata["final_train_package"] = bool(
        args.obs_mode == "image"
        and args.model_type == "sfps"
        and args.imitation_mode == "state"
        and args.image_agent_pos_cond
    )
    metadata["episode_indices"] = (
        dataset.episode_indices.tolist()
        if getattr(dataset, "episode_indices", None) is not None
        else None
    )
    metadata["stats_episode_indices"] = (
        dataset.stats_episode_indices.tolist()
        if getattr(dataset, "stats_episode_indices", None) is not None
        else None
    )
    with (args.save_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    loss_history = []
    global_step = 0
    stop_training = False
    with tqdm(range(1, args.epochs + 1), desc=f"Train {args.model_type}") as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
                    loss = policy.Loss(nbatch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(policy.parameters())
                    global_step += 1

                    loss_cpu = float(loss.item())
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu, step=global_step)
                    if args.max_train_steps is not None and global_step >= args.max_train_steps:
                        stop_training = True
                        break

            mean_loss = float(np.mean(epoch_loss))
            loss_history.append({"epoch": epoch_idx, "loss": mean_loss, "global_step": global_step})
            tglobal.set_postfix(loss=mean_loss, step=global_step)

            if epoch_idx % args.save_every == 0 or epoch_idx == args.epochs:
                save_checkpoint(policy, args.save_dir, epoch_idx, loss_history, ema=ema)
            if stop_training:
                break

    ema.copy_to(policy.parameters())
    save_checkpoint(policy, args.save_dir, epoch_idx, loss_history)
    print(f"Saved final checkpoint to {args.save_dir / 'latest.pth'}")


if __name__ == "__main__":
    main()
