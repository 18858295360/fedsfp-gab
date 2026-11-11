"""Federated (raw FedAvg) trainer for robomimic table3 SFP — reuses the verified
v3-deterministic recipe (switch-aware dataset, v22 model, paper AdamW) via the
same monkey-patch chain, adds a client/round loop.

Protocol (mirrors our Push-T raw_reset_both):
- clients from --split-json {clients: {client_i: [demo indices]}}
- GLOBAL normalizer fit on the FULL dataset, injected into every client dataset
- per round: client loads global weights, FRESH paper-AdamW, per-round cosine
  (warmup 50), trains --local-epochs epochs; sample-weighted FedAvg
- checkpoints: global every --save-every-rounds; saved in their checkpoint
  format (model_state + ema_model_state[=model_state] + normalizer + args)
  so their eval tooling loads it unchanged. NO EMA in federated arms (noted).
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.robomimic_sfp import train_table3_v3_deterministic as v3  # applies patches
base = v3.base
from diffusers.optimization import get_scheduler


class ClientDataset(v3.AllDemoSwitchAwareDataset):
    def __init__(self, *a, client_demos=None, **k):
        super().__init__(*a, **k)
        if client_demos is not None:
            allowed = set(int(x) for x in client_demos)
            self.train_indices = [i for i in self.train_indices if i in allowed]
            self.indices = [(e, s) for (e, s) in self.indices if e in allowed]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--obs-keys", default=None,
                   help="comma-separated observation keys; default is the\n                         four-key Square layout, so omitting it reproduces\n                         previous runs exactly")
    p.add_argument("--split-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sigma0", type=float, default=0.4)
    p.add_argument("--k", type=float, default=10.0)
    p.add_argument("--integration-steps-per-action", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=58)
    p.add_argument("--save-every-rounds", type=int, default=5)
    p.add_argument("--equal-agg-weight", action="store_true",
                   help="aggregate with p_i = 1/N instead of sample-proportional weights")
    p.add_argument("--anchor-frac-inverse-size", type=float, default=0.0,
                   help="scale each client anchor fraction by ((1/N)/p_i)**alpha")
    p.add_argument("--fednova", action="store_true",
                   help="normalise each client update by its local step count "
                        "before averaging (Wang et al., NeurIPS 2020)")
    p.add_argument("--global-anchor-frac", type=float, default=0.0,
                   help="probability that a training anchor is centred on "
                        "the round-start global's generated action rather "
                        "than on the expert trajectory")
    p.add_argument("--global-anchor-refresh", type=int, default=5,
                   help="regenerate the anchor tables every this many rounds")
    p.add_argument("--gab-reset-moment", choices=("none", "m", "mv"),
                   default="none",
                   help="zero the persistent Adam moments after each "
                        "anchor refresh; raw already resets every round")
    p.add_argument("--gab-rescale-d", type=float, default=0.0,
                   help="rescale each anchor path about the expert so "
                        "mean ||A-xi|| equals this; 0 disables")
    p.add_argument("--gab-randomize-direction", type=float, default=0.0,
                   help="replace the anchor residual with an isotropic random "
                        "vector of this norm. The paired control for "
                        "--gab-rescale-d: rescale keeps the direction the "
                        "trained global chose and only fixes the length, this "
                        "keeps the length and destroys the direction. The "
                        "frozen table's residual is at chance alignment with "
                        "xi_dot (|cos| 0.259 against 0.252 expected for "
                        "isotropic in 10-D) while the dynamic table reaches "
                        "0.359, so direction is the one thing rescale could "
                        "not have restored; 0 disables")
    p.add_argument("--gab-anchor", choices=("global", "loo", "self"),
                   default="global",
                   help="which model generates the anchor table: the "
                        "aggregate (global, the original); for each client "
                        "the weighted average of the OTHER clients (loo), "
                        "which has no centralized counterpart so any gain "
                        "is federation-specific; or the client's OWN "
                        "previous-round weights (self), which matches loo "
                        "in everything but the cross-client information.")
    p.add_argument("--gripper-source", default="binary_command")
    p.add_argument("--steps-budget-json", default=None,
                   help="JSON {client_key: steps_per_round}; fixes optimizer steps per round")
    p.add_argument("--optimizer-mode", choices=["raw", "persistent"], default="raw",
                   help="raw = fresh AdamW + per-round cosine each round; persistent = "
                        "per-client AdamW state kept across rounds + continuous cosine over total steps")
    p.add_argument("--normalizer-dataset-path", default=None,
                   help="fit normalizer on this hdf5 instead (e.g. the raw 300-demo file, so "
                        "augmented arms share the raw arm's normalizer bit-for-bit)")
    return p.parse_args()


@torch.no_grad()
def _refresh_global_anchor_tables(args, clients, model, device):
    """Integrate the frozen global's flow over every client sample.

    One row per dataset index, pred_horizon+1 points per row, so a sample can
    look up table[index, segment] for whichever flow-time segment it drew.

    The starting action is the sample's own point 0 -- the previous action --
    perturbed by sigma0, which is the only stochasticity this single-branch model
    offers. Values are clamped to a sane range so an early ill-trained global
    cannot inject infinities into the table.

    Generation reads the trajectory directly rather than going through
    __getitem__, because __getitem__ draws a random flow time and would return
    one point rather than the whole path.
    """
    import numpy as _np
    model.eval()
    for c in clients:
        ds = c["ds"]
        H = int(ds.pred_horizon)
        n = len(ds.indices)
        tbl = _np.zeros((n, H + 1, int(ds.action_dim)), dtype=_np.float32)
        B = 256
        for lo in range(0, n, B):
            hi = min(lo + B, n)
            obs_b, a0_b, xi_b = [], [], []
            for i in range(lo, hi):
                ep_idx, current = ds.indices[i]
                ep = ds.episodes[ep_idx]
                obs_ids = _np.clip(
                    _np.arange(current - ds.obs_horizon + 1, current + 1),
                    0, len(ep["obs"]) - 1)
                action_ids = _np.clip(
                    _np.arange(current - 1, current + ds.pred_horizon),
                    0, len(ep["action"]) - 1)
                obs_b.append(ds.normalizer.normalize_obs(ep["obs"][obs_ids]))
                traj = ds.normalizer.normalize_action(ep["action"][action_ids])
                a0_b.append(traj[0])
                xi_b.append(traj)
            obs_t = torch.as_tensor(_np.asarray(obs_b, dtype=_np.float32), device=device)
            a = torch.as_tensor(_np.asarray(a0_b, dtype=_np.float32), device=device)
            a = a + float(ds.sigma0) * torch.randn_like(a)
            rows = [a.clone()]
            for j in range(H):
                tj = torch.full((a.shape[0],), j / H, device=device, dtype=torch.float32)
                v = model(obs_t, a, tj)
                a = a + v / H
                rows.append(a.clone())
            _blk = torch.stack(rows, dim=1).clamp(-5, 5)
            _target = float(getattr(args, "gab_rescale_d", 0.0) or 0.0)
            if _target > 0:
                # hold the supervision the restoring term supplies at a fixed
                # distance: scale the residual about the client's own expert,
                # which keeps the direction the current global chose
                _xi = torch.as_tensor(_np.asarray(xi_b, dtype=_np.float32),
                                      device=_blk.device)
                _res = _blk - _xi
                _nrm = _res.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                _blk = _xi + _res * (_target / _nrm)
            _rand = float(getattr(args, "gab_randomize_direction", 0.0) or 0.0)
            if _rand > 0:
                # same place, same tensor and same norm as the rescale control,
                # so the two arms differ in direction alone
                _xi = torch.as_tensor(_np.asarray(xi_b, dtype=_np.float32),
                                      device=_blk.device)
                _u = torch.randn_like(_blk)
                _u = _u / _u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                _blk = _xi + _u * _rand
            tbl[lo:hi] = _blk.cpu().numpy()
        ds.global_anchor_table = tbl
        _base = float(args.global_anchor_frac)
        _alpha = float(getattr(args, "anchor_frac_inverse_size", 0.0) or 0.0)
        if _alpha > 0.0 and _base > 0.0:
            _sizes = [int(_c.get("n_demos", _c.get("weight", 1)))
                      for _c in clients]
            _tot = float(sum(_sizes)) or 1.0
            _pi = float(len(ds.indices)) / _tot
            _q = _base * ((1.0 / len(_sizes)) / max(_pi, 1e-9)) ** _alpha
            _q = min(max(_q, 0.0), 1.0)
        else:
            _q = _base
        ds.global_anchor_frac = _q
        print("[gab] client %s: n=%d q=%.3f"
              % (c["key"], len(ds.indices), _q), flush=True)
        hit = float((abs(tbl) >= 5.0 - 1e-6).mean())
        print("[gab] %s: table %s refreshed, clamp-hit %.5f"
              % (c["key"], tbl.shape, hit), flush=True)
    model.train()


def save_global(path, model, full_ds, args, rnd, gstep):
    # mirrors base.save_checkpoint format so their eval tooling loads it unchanged
    torch.save({
        "model_state": model.state_dict(),
        "ema_model_state": model.state_dict(),  # eval-tool compat; NO real EMA in fed
        "optimizer_state": {}, "scheduler_state": {},
        "normalizer": full_ds.normalizer.state_dict(),
        "obs_dim": full_ds.obs_dim,
        "action_dim": full_ds.action_dim,
        "obs_horizon": full_ds.obs_horizon,
        "pred_horizon": full_ds.pred_horizon,
        "model_config": {
            "diffusion_step_embed_dim": base._SFP_EMBED,
            "down_dims": base._SFP_DOWN,
            "kernel_size": 5,
            "n_groups": 8,
            "fc_timesteps": 1,
        },
        "table3": {
            "dataset_variant": "square_mh_dp_absolute_standard_foh_gripper_v1",
            "num_total_demos": len(full_ds.episodes),
            "num_train_demos": len(full_ds.train_indices),
            "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            "action_space": "world EEF pos + rotation6d + gripper open fraction",
            "trajectory_grid_points": full_ds.pred_horizon + 1,
            "trajectory_grid_intervals": full_ds.pred_horizon,
            "trajectory_alignment": "point0=action[t-1], point1=action[t]",
            "gripper_velocity": "piecewise_linear_first_order_hold",
            "gripper_source": full_ds.gripper_source,
            "observation_source": "raw_hdf_observations",
            "controller_anchor_cache": False,
            "biased_gripper_sampling": True,
            "biased_gripper_prob": 0.5,
            "sigma0": args.sigma0,
            "k": args.k,
            "integration_steps_per_action": args.integration_steps_per_action,
            "action_horizon_square": 8,
        },
        # recipe metadata expected by rollout_..._v3_switchaware.validate_checkpoint —
        # truthful: fed clients use the identical v3 dataset/optimizer recipe, and the
        # global model sees all 300 demos across clients
        "args": {**vars(args),
                 "action_alignment": "previous_action_then_current_through_t_plus_15",
                 "time_conditioning": "continuous_t_in_0_1_unscaled",
                 "time_sampling": "uniform_0_1_with_switch_segment_oversampling_p0.5",
                 "train_demo_policy": "all_300",
                 "adam_betas": [0.9, 0.999],
                 "biased_gripper_sampling": True,
                 "biased_gripper_prob": 0.5},
        "epoch": rnd, "global_step": gstep,
        "federated": True,
    }, path)


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(args.split_json).read_text())
    clients_cfg = split["clients"]

    if args.obs_keys:
        _keys = tuple(k.strip() for k in args.obs_keys.split(',') if k.strip())
        print('[obs] custom keys (%d): %s' % (len(_keys), list(_keys)), flush=True)
    else:
        _keys = None
    ds_kw = dict(sigma0=args.sigma0, stabilization_k=args.k,
                 **({'obs_keys': _keys} if _keys else {}),
                 integration_steps_per_action=args.integration_steps_per_action,
                 gripper_source=args.gripper_source)
    full = v3.AllDemoSwitchAwareDataset(args.normalizer_dataset_path or args.dataset_path, **ds_kw)
    normalizer = full.normalizer
    print("[knobs] gabfrac=%s refresh=%s sigma0=%s k=%s rounds=%s "
          "local_epochs=%s seed=%s split=%s"
          % (getattr(args, "global_anchor_frac", "ABSENT"),
             getattr(args, "global_anchor_refresh", "ABSENT"),
             args.sigma0, args.k, args.rounds, args.local_epochs,
             args.seed, Path(args.split_json).name), flush=True)
    print(f"[fed] normalizer/full dataset demos={len(full.train_indices)} obs={full.obs_dim} act={full.action_dim}", flush=True)
    if _keys:
        print("[obs] built width %d from %d keys" % (full.obs_dim, len(_keys)), flush=True)
    steps_budget = json.loads(Path(args.steps_budget_json).read_text()) if args.steps_budget_json else None
    if steps_budget:
        print(f"[fed] steps budget: {steps_budget}", flush=True)

    clients = []
    for key in sorted(clients_cfg.keys()):
        ds = ClientDataset(args.dataset_path, normalizer=normalizer,
                           client_demos=clients_cfg[key], **ds_kw)
        # persistent workers snapshot the dataset at their first fork, so a
        # table refreshed in the parent after that never reaches training.
        # Only disable them when the anchor table is actually in use.
        _persist = (args.num_workers > 0
                    and float(getattr(args, "global_anchor_frac", 0.0) or 0.0) <= 0.0)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=_persist, drop_last=True)
        clients.append(dict(key=key, ds=ds, loader=loader,
                            weight=(1.0 if getattr(args, "equal_agg_weight", False)
                                    else len(ds.indices)),
                            n_demos=len(ds.indices)))
        print(f"[fed] {key}: demos={len(ds.train_indices)} samples={len(ds.indices)} "
              f"batches={len(loader)}", flush=True)

    def new_model():
        return base.SFPConditionalUnet1D(
            obs_dim=full.obs_dim, action_dim=full.action_dim,
            obs_horizon=full.obs_horizon,
            down_dims=base._SFP_DOWN, diffusion_step_embed_dim=base._SFP_EMBED,
            fc_timesteps=1).to(device)

    global_model = new_model()
    work = new_model()
    gstep = 0
    if args.optimizer_mode == "persistent":
        # per-client AdamW bound once to work's param tensors (load_state_dict copies
        # in-place, so param identity — and thus Adam state — survives across rounds)
        for c in clients:
            spr = (int(steps_budget[c["key"]]) if steps_budget and c["key"] in steps_budget
                   else len(c["loader"]) * args.local_epochs)
            c["opt"] = torch.optim.AdamW(work.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                         eps=1e-8, weight_decay=args.weight_decay)
            c["sched"] = get_scheduler("cosine", optimizer=c["opt"],
                                       num_warmup_steps=args.warmup_steps,
                                       num_training_steps=max(spr * args.rounds, 1))
    # loo/self need the individual client models, which only exist AFTER a
    # round of local training, so round r uses what round r-1 produced --
    # the same one-round lag the frozen round-start global already has.
    _loo_states = {}
    _loo_round = [0]
    for rnd in range(1, args.rounds + 1):
        gstate = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        if (float(getattr(args, "global_anchor_frac", 0.0) or 0.0) > 0
                and (rnd - 1) % max(int(getattr(args, "global_anchor_refresh", 5) or 5), 1) == 0):
            _mode = getattr(args, "gab_anchor", "global")
            _alt = (_mode in ("loo", "self")
                    and _loo_round[0] == rnd - 1
                    and len(_loo_states) == len(clients)
                    and (_mode == "self" or len(clients) > 1))
            if _alt:
                # one call per client with a single-element list: the
                # function's own loop then sees exactly one dataset and one
                # model, so nothing inside it changes. work carries the
                # weights, avoiding a second full model.
                for _c in clients:
                    _lo = {}
                    for _n in gstate:
                        # loo takes every OTHER client, self takes only this one
                        _vals = [(_w2, _sd2[_n])
                                 for _k2, (_w2, _sd2) in _loo_states.items()
                                 if ((_k2 != _c["key"]) == (_mode == "loo"))
                                 and _n in _sd2]
                        if not _vals:
                            _lo[_n] = gstate[_n]
                        elif _vals[0][1].dtype.is_floating_point:
                            _den = sum(_w2 for _w2, _ in _vals)
                            _a2 = None
                            for _w2, _v2 in _vals:
                                _t2 = _w2 * _v2.float()
                                _a2 = _t2 if _a2 is None else _a2 + _t2
                            _lo[_n] = (_a2 / _den).to(gstate[_n].dtype)
                        else:
                            _lo[_n] = _vals[0][1]
                    work.load_state_dict(_lo)
                    _refresh_global_anchor_tables(args, [_c], work, device)
                work.load_state_dict(gstate)
                print("[gab] anchor source: %s, round %d"
                      % ("leave-one-out" if _mode == "loo" else "own-previous",
                         rnd), flush=True)
            else:
                _refresh_global_anchor_tables(args, clients, global_model, device)
            _rm = getattr(args, "gab_reset_moment", "none")
            if _rm in ("m", "mv") and args.optimizer_mode == "persistent":
                _n = 0
                for _c in clients:
                    for _st in _c["opt"].state.values():
                        if "exp_avg" in _st:
                            _st["exp_avg"].zero_(); _n += 1
                        if _rm == "mv" and "exp_avg_sq" in _st:
                            _st["exp_avg_sq"].zero_()
                print("[gab] reset %s on %d tensors after refresh, round %d"
                      % (_rm, _n, rnd), flush=True)
                if _mode != "global":
                    print("[gab] anchor source: global (%s not available "
                          "at round %d)" % (_mode, rnd), flush=True)
        acc = None; wsum = 0.0
        nova_acc = None; nova_tau = 0.0
        round_losses = []
        for c in clients:
            work.load_state_dict(gstate)
            work.train()
            steps_total = (int(steps_budget[c["key"]]) if steps_budget and c["key"] in steps_budget
                           else len(c["loader"]) * args.local_epochs)
            if args.optimizer_mode == "persistent":
                opt, sched = c["opt"], c["sched"]
            else:
                # NB: v3 shim globally patches torch.optim.AdamW → paper_adamw (betas forced 0.9,0.999)
                opt = torch.optim.AdamW(work.parameters(), lr=args.lr,
                                        betas=(0.9, 0.999), eps=1e-8,
                                        weight_decay=args.weight_decay)
                sched = get_scheduler("cosine", optimizer=opt,
                                      num_warmup_steps=min(args.warmup_steps, steps_total // 10),
                                      num_training_steps=max(steps_total, 1))
            done_steps = 0
            while done_steps < steps_total:
                for batch in c["loader"]:
                    if done_steps >= steps_total:
                        break
                    obs = batch["obs"].to(device, non_blocking=True)
                    action = batch["a"].to(device, non_blocking=True)
                    t = batch["t"].to(device, non_blocking=True)
                    target = batch["v"].to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    loss = F.mse_loss(work(obs, action, t), target)
                    loss.backward()
                    opt.step(); sched.step()
                    gstep += 1
                    done_steps += 1
                    round_losses.append(float(loss.detach().cpu()))
            w = float(c["weight"]); wsum += w
            cs = work.state_dict()
            if getattr(args, "fednova", False):
                # tau_i is the number of optimizer steps this client actually
                # took, which is what makes a large site's update longer rather
                # than merely heavier
                _tau = max(int(steps_total), 1)
                nova_tau += w * _tau
                if nova_acc is None:
                    nova_acc = {k: (cs[k].detach().float()
                                    - gstate[k].detach().float()) * (w / _tau)
                                for k in cs}
                else:
                    for k in cs:
                        nova_acc[k] += (cs[k].detach().float()
                                        - gstate[k].detach().float()) * (w / _tau)
            if getattr(args, "gab_anchor", "global") in ("loo", "self"):
                # CPU copies: cs aliases work, which the next client overwrites
                _loo_states[c["key"]] = (
                    w, {_n: _v.detach().to("cpu", copy=True)
                        for _n, _v in cs.items()})
                _loo_round[0] = rnd
            if acc is None:
                acc = {k: cs[k].detach().clone().float() * w for k in cs}
            else:
                for k in cs:
                    acc[k] += cs[k].detach().float() * w
        _gsd = global_model.state_dict()
        if getattr(args, "fednova", False):
            # theta <- theta_g + tau_eff * sum_i p_i (theta_i - theta_g) / tau_i,
            # with p_i = w_i / wsum.  Integer buffers have no meaningful delta,
            # so they keep the plain weighted average.
            tau_eff = nova_tau / wsum
            agg = {}
            for k in acc:
                if _gsd[k].is_floating_point():
                    agg[k] = (gstate[k].detach().float()
                              + tau_eff * nova_acc[k] / wsum).to(dtype=_gsd[k].dtype)
                else:
                    agg[k] = (acc[k] / wsum).to(dtype=_gsd[k].dtype)
            if rnd == 1:
                print("[fednova] tau_eff=%.1f taus=%s" % (tau_eff, nova_tau), flush=True)
        else:
            agg = {k: (acc[k] / wsum).to(dtype=_gsd[k].dtype) for k in acc}
        global_model.load_state_dict(agg)
        rec = dict(round=rnd, global_step=gstep, loss=float(np.mean(round_losses)))
        print(json.dumps(rec), flush=True)
        with (out / "metrics.jsonl").open("a") as s:
            s.write(json.dumps(rec) + "\n")
        if rnd % args.save_every_rounds == 0 or rnd == args.rounds:
            save_global(out / f"global_round_{rnd:04d}.pt", global_model, full, args, rnd, gstep)
            save_global(out / "latest.pt", global_model, full, args, rnd, gstep)
    print("[fed] DONE", flush=True)


if __name__ == "__main__":
    main()
