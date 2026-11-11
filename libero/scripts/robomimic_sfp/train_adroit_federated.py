#!/usr/bin/env python3
"""Federated SFP trainer for the LIBERO-10 replication recipe.

The trainer uses window-weighted aggregation and fixed per-client
optimizer-step accounting. The image/action recipe uses the v22 UNet and
uniform-time conditional flow sampling.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from streaming_flow_policy.robomimic_sfp.model_v22_paper_aligned import SFPConditionalUnet1D

_SFP_DOWN = json.loads(os.environ.get('SFP_DOWN_DIMS', '[256, 512, 1024]'))
_SFP_EMBED = int(os.environ.get('SFP_EMBED', '256'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--biased-gripper-prob", type=float, default=0.0,
                    help="probability of landing the flow time inside a "
                         "gripper-switch segment; grasp transitions are "
                         "~1%% of steps and are otherwise never sampled")
    p.add_argument("--normalizer-kind", choices=["legacy", "official"],
                    default="legacy",
                    help="official = per-dim min-max on obs and actions, "
                         "the recipe the 64%% Square line records")
    p.add_argument("--libero-root",
        required=True,
        help="directory of LIBERO-Spatial hdf5 files, one per client")
    p.add_argument("--equal-agg-weight", action="store_true",
                   help="aggregate with p_i = 1/N instead of the window-proportional weight")
    p.add_argument("--anchor-frac-inverse-size", type=float, default=0.0,
                   help="scale each client anchor fraction by ((1/N)/p_i)**alpha")
    p.add_argument("--global-anchor-frac", type=float, default=0.0,
                   help="probability that a training anchor is centred on "
                        "the round-start global's generated action")
    p.add_argument("--global-anchor-subsample", type=float, default=0.25,
                   help="fraction of windows given a generated anchor row")
    p.add_argument("--global-anchor-refresh", type=int, default=5,
                   help="regenerate the anchor tables every this many rounds")
    p.add_argument("--gab-randdir", action="store_true",
                   help="replace each anchor residual with a random direction "
                        "of the SAME length, per window: the paired control "
                        "for the anchor source. Keeps the magnitude the "
                        "policy-generated table produced and destroys only "
                        "the direction, so a gain that survives it is not "
                        "about where the aggregate actually goes.")
    p.add_argument("--gab-anchor", choices=("global", "loo", "self"),
                   default="global",
                   help="which model generates the anchor table: the "
                        "aggregate (global, the original); for each client "
                        "the weighted average of the OTHER clients (loo), "
                        "which has no centralized counterpart so any gain "
                        "from it is federation-specific; or the client's "
                        "OWN previous-round weights (self), which matches "
                        "loo in everything except the cross-client "
                        "information and isolates it.")
    p.add_argument("--anchor-broaden", type=float, default=0.0,
                   help="widen the flow anchor by this multiple of sigma0 "
                        "during local training, so a client learns to "
                        "restore from starting points its own slice never "
                        "reaches")
    p.add_argument("--residual-gate", type=float, default=0.0,
                   help="weight each local sample by the frozen "
                        "round-start model's error on it, so a client "
                        "learns only what the global does not know; "
                        "0 disables, 1 uses the raw normalised error")
    p.add_argument("--flare-confidence-gate", action="store_true",
                   help="scale the FiLM by the gate's confidence, so a "
                        "uniform gate reduces the model to the shared "
                        "policy instead of applying a noisy expert")
    p.add_argument("--flare-experts", type=int, default=0,
                   help="build FLARE with this many experts; 0 keeps the "
                        "plain policy. An expert is a FiLM row on the "
                        "conditioning, active only below --flare-t-split")
    p.add_argument("--flare-t-split", type=float, default=0.5,
                   help="flow time below which experts apply; 0.50 beat "
                        "0.25 when the mixture was assembled from real "
                        "client models")
    p.add_argument("--flare-gate-weight", type=float, default=0.1)
    p.add_argument("--libero-partition", default="task",
                   choices=["task", "init", "mode"],
                   help="task: one client per LIBERO task, which is "
                        "task shift. init: one client per initial-state region, "
                        "every client holding every task in equal proportion, "
                        "which is support shift on the same data")
    p.add_argument("--libero-clients", type=int, default=3,
                   help="regions for --libero-partition init; 50 demos "
                        "per task allow about three")
    p.add_argument("--libero-lang", default="",
                   help="path to the instruction-embedding npz. LIBERO is "
                        "language-conditioned and every published number on "
                        "it comes from a model that is told the task")
    p.add_argument("--restore-cap", type=float, default=0.0,
                   help="saturate the restoring term at this many ordinary-"
                        "anchor sigmas. 0 disables. Aimed at global anchors, "
                        "whose displacement does not shrink with sigma(t) and "
                        "otherwise dominates the target by 10-17x late in the "
                        "flow")
    p.add_argument("--dump-clients-at-round", type=int, default=0,
                   help="save every client model just before aggregation at this round; their weighted mean is the next global "
                        "model, so averaging can be compared with its own inputs")
    p.add_argument("--policy", choices=["sfp", "dp"], default="sfp",
                   help="sfp: flow matching. dp: diffusion policy on the "
                        "same network, DDPM training and DDIM sampling")
    p.add_argument("--dp-steps", type=int, default=100,
                   help="diffusion steps used in training")
    p.add_argument("--libero-mode-clients", type=int, default=2,
                   help="how many clients the by-mode split makes")
    p.add_argument("--libero-mode-iid", type=int, default=0,
                   help="IID control for the client-count sweep: shuffle "
                        "which demonstrations land in which client while "
                        "keeping the client count and sizes the by-mode "
                        "split produced. 0 disables; any other value is the "
                        "shuffle seed.")
    p.add_argument("--libero-mode-feature", default="azimuth",
                   help="what a behavioural mode is measured in")
    p.add_argument("--libero-keep-client", type=int, default=-1,
                   help="train only this client index, alone: the local-only "
                        "baseline. The partition and datasets are unchanged; "
                        "only the number of participants differs, so the "
                        "contrast isolates aggregation")
    p.add_argument("--libero-merge-clients", action="store_true",
                   help="pool every LIBERO file into one client: the "
                        "centralized control, which removes aggregation and "
                        "leaves the data and sampling untouched")
    p.add_argument("--libero-demo-limit", type=int, default=None)
    p.add_argument("--dataset-class", choices=["libero"],
                    default="libero",
                    help="table3 = the reference pipeline: robomimic obs "
                         "groups, raw7_to_model10, gripper_source")
    p.add_argument("--gripper-source", choices=["next_qpos", "binary_command"],
                    default="binary_command")
    p.add_argument("--server-optimizer", choices=["fedavg", "fedadam", "fedavgm"],
                    default="fedavg",
                    help="fedadam: Adam on the server over (average - global), "
                         "damping directions the clients disagree about")
    p.add_argument("--server-lr", type=float, default=0.01)
    p.add_argument("--server-beta1", type=float, default=0.9)
    p.add_argument("--server-beta2", type=float, default=0.99)
    p.add_argument("--server-eps", type=float, default=1e-3)
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--split-json", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sigma0", type=float, default=0.4)
    p.add_argument("--k", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every-rounds", type=int, default=5)
    p.add_argument("--steps-budget-json", default=None)
    p.add_argument("--fmime-anchor-noise", action="store_true",
        help="draw a second independent anchor per client to size sampling noise")
    p.add_argument("--fmime-anchor-batches", type=int, default=6,
        help="batches per client used to build the flow-balanced anchor")
    p.add_argument("--fmime-no-correction", action="store_true",
        help="server-global frozen moments only; drop g - a_i + a_bar")
    p.add_argument("--fmime-bins", type=int, default=3,
        help="flow-time bins the anchor is balanced over")
    p.add_argument("--moon-mu", type=float, default=0.0,
        help="weight of the MOON contrastive term; 0 disables it")
    p.add_argument("--moon-tau", type=float, default=0.5)
    p.add_argument("--reset-first-moment",
                   choices=("none", "round", "refresh"), default="none",
                   help="zero Adam's first moment at the start of every "
                        "round while leaving the second moment and the "
                        "scheduler alone; with --optimizer-mode persistent "
                        "this is exactly Push-T's v_only")
    p.add_argument("--scaffold", action="store_true",
                   help="SCAFFOLD control variates (Karimireddy et al. 2020)")
    p.add_argument("--fedprox-mu", type=float, default=0.0,
                   help="FedProx proximal weight; 0 disables it")
    p.add_argument("--feddyn-alpha", type=float, default=0.0,
        help="FedDyn dynamic regularisation strength; 0 disables it")
    p.add_argument("--optimizer-mode",
                   choices=["raw", "persistent", "sched_only", "opt_only", "const", "tailanneal",
                            "vp", "pgs", "svr", "fmime"],
                   default="raw",
                   help="raw: fresh Adam + per-round cosine restarts; persistent: both kept; "
                        "sched_only: fresh Adam each round + ONE global cosine (isolates the "
                        "schedule component); opt_only: persistent Adam + per-round cosine "
                        "restarts (isolates the Adam-moment component)")
    p.add_argument("--task-variant", default="adroit_relocate_flat_v1")
    p.add_argument("--lr-shape", choices=["cosine", "linear"], default="cosine",
                   help="post-warmup profile of the global schedule. linear matches the "
                        "Push-T champion (global continuous linear decay to 0)")
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="fixed parameter updates per local epoch (canonical FTP3 uses 100, "
                        "giving 7*100=700 updates/client/round and a 35,000-step global LR "
                        "clock). 0 = iterate the whole loader (legacy behaviour).")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA decay; exactly one EMA update per batch. 0 disables. "
                        "Canonical FTP3 enables EMA and FedAvgs the EMA weights too.")
    p.add_argument("--fed-flow", choices=("none", "fca", "fcv", "fcc"), default="none",
                   help="federated-flow method on top of the per-bin split. "
                        "fcv: per-bin control variates (gradient units). "
                        "fcc: server reweights next-round flow-time sampling by "
                        "cross-client per-bin disagreement D_b.")
    p.add_argument("--obs-mode", choices=("flat", "image"), default="flat",
                   help="flat: a single obs vector (Adroit, Square low-dim). "
                        "image: rendered frames + proprio via SFPImagePolicy; "
                        "multi-camera views are stacked on the channel axis.")
    p.add_argument("--image-encoder", choices=["v1", "v2", "v3"], default="v1",
                    help="v2: per-camera ResNet-18 with spatial softmax "
                         "and GroupNorm, instead of one shared conv stack")
    p.add_argument("--num-kp", type=int, default=32)
    p.add_argument("--img-size", type=int, default=84)
    p.add_argument("--imitation-mode", choices=("action", "state"),
                   default="action",
                   help="action: imitate the recorded OSC delta command. "
                        "state: imitate the next end-effector pose, normalised, "
                        "with the gripper command as the seventh channel "
                        "(a gripper command is not recoverable from any state)")
    p.add_argument("--state-objects", action="store_true",
                   help="flat obs = proprio + 7*K object poses recovered from "
                        "the recorded simulator states; needs --object-slices")
    p.add_argument("--object-slices",
                   default="",
                   help="task -> free-joint qpos map from the slice generator")
    p.add_argument("--fcc-alpha", type=float, default=1.0)
    p.add_argument("--pgs-lambda", type=float, default=0.0,
                   help="PGS shrinkage: v_i <- v_bar + lam*(v_i - v_bar). "
                        "lam=0 is the Push-T champion point (shared v_bar).")
    p.add_argument("--svr-refresh-after", type=int, default=20,
                   help="SVR: zero the second moment from this round on.")
    p.add_argument("--fca-weight", choices=("binshare", "clientsize"),
                   default="binshare",
                   help="FCA aggregation weight. binshare is the method; "
                        "clientsize reduces exactly to FedAvg and is the control.")
    p.add_argument("--ftp-bins", type=int, default=0,
                   help="FedSFP-FTP: split each batch by flow time into K bins and give "
                        "each bin its own Adam state (client-persistent second moment, "
                        "first moment reset every round). 0 disables.")
    return p.parse_args()


@torch.no_grad()
def ema_update(ema_model, model, decay):
    """Exactly one EMA update per batch (never one per FTP bin)."""
    ep = dict(ema_model.named_parameters())
    for n, p in model.named_parameters():
        ep[n].mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    eb = dict(ema_model.named_buffers())
    for n, b in model.named_buffers():
        eb[n].copy_(b)



def _fwd_loss(model, batch, device, reduction="mean", weight=None):
    """Forward pass covering both obs layouts the trainer supports."""
    obs = (batch["imgs"] if "imgs" in batch else batch["obs"]).to(device, non_blocking=True)
    a = batch["a"].to(device, non_blocking=True)
    t = batch["t"].to(device, non_blocking=True)
    tgt = batch["v"].to(device, non_blocking=True)
    _lang = (batch["lang"].to(device, non_blocking=True)
             if "lang" in batch else None)
    if "proprio" in batch:
        pred = model(obs, batch["proprio"].to(device, non_blocking=True), a, t,
                     lang=_lang)
    else:
        pred = model(obs, a, t)
    if weight is None:
        return F.mse_loss(pred, tgt, reduction=reduction), t
    per = ((pred - tgt) ** 2).flatten(1).mean(1)
    return (per * weight).sum(), t


@torch.no_grad()
def _refresh_gab_tables(args, clients, model):
    """Integrate the round-start global's flow over every client window.

    One row per dataset index and per trajectory grid point, so a sample looks
    up table[i, segment]. The start is the window's own first action perturbed
    by sigma0; the model is the freshly aggregated global, before any client
    moves this round. Values are clamped so an early ill-trained global cannot
    seed the table with infinities -- the clamp-hit rate is printed and should
    stay at zero.
    """
    from scripts.robomimic_sfp.libero_dataset import _window_for_table
    import numpy as _np
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    for c in clients:
        ds = c["ds"]
        n = len(ds.index)
        H = int(ds.pred_horizon)
        B = 96
        sub = float(getattr(args, "global_anchor_subsample", 0.25) or 1.0)
        rng = _np.random.RandomState(1234 + len(ds.index))
        if sub < 1.0:
            # choose whole episodes, so the reads run in file order
            eps = sorted({(fi, key) for fi, key, _ in ds.index})
            keep = {e for e in eps if rng.rand() < sub}
            if not keep:                      # never leave a client with none
                keep = {eps[0]}
            mask = _np.array([(fi, key) in keep for fi, key, _ in ds.index])
        else:
            mask = _np.ones(n, dtype=bool)
        rows_wanted = _np.flatnonzero(mask)
        tbl = None
        for lo0 in range(0, len(rows_wanted), B):
            sel = rows_wanted[lo0:lo0 + B]
            imgs, props, langs, a0s = [], [], [], []
            for i in sel:
                im, pr, lg, a0 = _window_for_table(ds, i)
                imgs.append(im); props.append(pr); a0s.append(a0)
                if lg is not None:
                    langs.append(lg)
            obs_t = torch.as_tensor(_np.stack(imgs), device=dev)
            prop_t = torch.as_tensor(_np.stack(props), device=dev)
            lang_t = (torch.as_tensor(_np.stack(langs), device=dev)
                      if langs else None)
            a = torch.as_tensor(_np.stack(a0s), device=dev)
            if tbl is None:
                tbl = _np.zeros((n, H, a.shape[1]), dtype=_np.float32)
            a = a + float(ds.sigma0) * torch.randn_like(a)
            rows = [a.clone()]
            for j in range(H - 1):
                t = torch.full((a.shape[0],), j / (H - 1), device=dev)
                if getattr(ds, "state_only", False):
                    # proprioception is the observation; there is no second
                    # input, and the flat policy takes (obs, a, t)
                    pred = model(prop_t, a, t)
                elif lang_t is None:
                    pred = model(obs_t, prop_t, a, t)
                else:
                    try:
                        pred = model(obs_t, prop_t, a, t, lang=lang_t)
                    except TypeError:
                        pred = model(obs_t, prop_t, a, t)
                a = a + pred / (H - 1)
                rows.append(a.clone())
            tbl[sel] = torch.stack(rows, dim=1).clamp(-3, 3).cpu().numpy()
            if getattr(args, "gab_randdir", False):
                from scripts.robomimic_sfp.libero_dataset import (
                    _virtual_actions as _va)
                for _i in sel:
                    _fi, _key, _t0 = ds.index[_i]
                    _g = ds._file(_fi)["data"][_key]
                    if getattr(ds, "imit_state", False):
                        _tj = _va(ds, _g, _t0)
                    else:
                        _tj = _np.asarray(_g["actions"][_t0:_t0 + H],
                                          dtype=_np.float32)
                    if _tj.shape[0] < H:
                        _tj = _np.concatenate(
                            [_tj, _np.repeat(_tj[-1:], H - _tj.shape[0],
                                             axis=0)])
                    _res = tbl[_i] - _tj
                    _nrm = _np.linalg.norm(_res, axis=-1, keepdims=True)
                    _u = _np.random.randn(*_res.shape).astype(_np.float32)
                    _u /= _np.maximum(_np.linalg.norm(_u, axis=-1,
                                                      keepdims=True), 1e-6)
                    tbl[_i] = (_tj + _u * _nrm).astype(_np.float32)
        ds.global_anchor_table = tbl
        ds.global_anchor_mask = mask
        _base = float(args.global_anchor_frac)
        _alpha = float(getattr(args, "anchor_frac_inverse_size", 0.0) or 0.0)
        _sizes = getattr(args, "_client_sizes", None)
        if _alpha > 0.0 and _base > 0.0 and _sizes:
            _tot = float(sum(_sizes)) or 1.0
            _pi = float(getattr(ds, "_agg_n", _tot / len(_sizes))) / _tot
            _q = _base * ((1.0 / len(_sizes)) / max(_pi, 1e-9)) ** _alpha
            _q = min(max(_q, 0.0), 1.0)
        else:
            _q = _base
        ds.global_anchor_frac = _q
        print("[gab] client n=%d q=%.3f" % (getattr(ds, "_agg_n", -1), _q),
              flush=True)
        hit = float((_np.abs(tbl[mask]) >= 3.0 - 1e-6).mean()) if tbl is not None else 0.0
        if getattr(args, "gab_randdir", False):
            print("[gab] randdir ON: residual direction randomized, "
                  "per-window length preserved", flush=True)
        print("[gab] %s: %d of %d rows (%d episodes), effective anchor "
              "frac %.3f, clamp-hit %.5f"
              % (c.get("key", "?"), int(mask.sum()), n,
                 len({(fi, k2) for j, (fi, k2, _) in enumerate(ds.index)
                      if mask[j]}),
                 float(args.global_anchor_frac) * mask.mean(), hit), flush=True)
        # close the h5py handles this pass opened in the main process; a
        # forked loader worker inheriting them would share file offsets
        for k2 in range(len(ds.files)):
            try:
                if ds.files[k2] is not None:
                    ds.files[k2].close()
            except Exception:
                pass
            ds.files[k2] = None
    if was_training:
        model.train()


def flow_balanced_anchor(model, loader, device, n_batches, bins=3):
    """Gradient at the current parameters, balanced across flow time.

    Each sample is weighted by 1/(bins * count-in-its-bin) so every bin
    contributes equally regardless of how many samples landed in it. Binning is
    per-sample: a batch mean would sit at 0.5 for any reasonable batch size and
    collapse the whole scheme into the middle bin."""
    model.zero_grad(set_to_none=True)
    seen = [0] * bins
    used = 0
    it = iter(loader)
    while used < n_batches:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            try:
                batch = next(it)
            except StopIteration:
                break
        t = batch["t"].to(device, non_blocking=True).flatten()
        idx = (t * bins).long().clamp_(0, bins - 1)
        w = torch.zeros_like(t)
        for b in range(bins):
            sel = idx == b
            n = int(sel.sum())
            if n:
                w[sel] = 1.0 / float(bins * n * n_batches)
                seen[b] += n
        loss, _ = _fwd_loss(model, batch, device, weight=w)
        loss.backward()
        used += 1
    grad = {n: (q.grad.detach().clone() if q.grad is not None else torch.zeros_like(q))
            for n, q in model.named_parameters() if q.requires_grad}
    model.zero_grad(set_to_none=True)
    return grad, seen, used


def fmime_step(work, batch, lr, wd, device, mom, corr):
    """One local step preconditioned by the server's frozen moments.

    mom["m"], mom["v"] are read and never written: in MIME they advance only on
    the server, from the aggregated anchor. Updating them here would rebuild
    exactly the per-client optimizer drift the method removes."""
    work.zero_grad(set_to_none=True)
    loss, _ = _fwd_loss(work, batch, device)
    loss.backward()
    b1, eps = 0.9, 1e-8
    with torch.no_grad():
        for n, q in work.named_parameters():
            if q.grad is None:
                continue
            g = q.grad
            if corr is not None and n in corr:
                g = g + corr[n]           # g - a_i + a_bar
            mg, vg = mom["m"].get(n), mom["v"].get(n)
            if mg is None:                # round 1: no server statistics yet
                d = g
            else:
                # MIME freezes v for the round after seeding it from one
                # anchor, so entries whose anchor was near zero would otherwise
                # divide by ~eps. Floor the denominator at a fraction of this
                # tensor's own RMS so scale is preserved where v is healthy.
                rms = vg.mean().sqrt()
                vg_floor = (vg.sqrt()).clamp_min(0.01 * rms + eps)
                d = (b1 * mg + (1.0 - b1) * g) / (vg_floor + eps)
            # bound the step: a preconditioned update has no natural scale limit
            dn = d.norm()
            if dn > 10.0:
                d = d * (10.0 / dn)
            if wd:
                q.add_(q, alpha=-lr * wd)
            q.add_(d, alpha=-lr)
    lv = float(loss.detach())
    if lv != lv:
        raise SystemExit(
            "[fmime] loss became NaN -- refusing to continue and write NaN "
            "checkpoints; the preconditioner or the correction has diverged")
    return lv


def fmime_server_update(mom, abar, device):
    """Advance the server moments from the aggregated anchor.

    Round 1 seeds m=a_bar, v=a_bar^2 rather than starting at zero. A zero v
    would make the first preconditioned step divide by ~eps and blow up, and
    bias correction is the usual fix; seeding is the same thing without having
    to carry the step counter through every local update."""
    first = mom["step"] == 0
    b1, b2 = 0.9, 0.999
    for n, g in abar.items():
        gd = g.to(device)
        if first:
            mom["m"][n] = gd.clone()
            mom["v"][n] = gd * gd
        else:
            mom["m"][n].mul_(b1).add_(gd, alpha=1 - b1)
            mom["v"][n].mul_(b2).addcmul_(gd, gd, value=1 - b2)
    mom["step"] += 1



def moon_obs_loss(work, glob, prev, obs, proprio, tau):
    """Contrastive alignment of the observation representation.

    Requires the policy to expose encode_obs(); the state-only policy has no
    learned observation encoder at all, so callers must not enable MOON there --
    aligning a fixed, parameter-free quantity is a no-op that would silently
    report as a working run.
    """
    z = work.encode_obs(obs, proprio).flatten(1)
    with torch.no_grad():
        zg = glob.encode_obs(obs, proprio).flatten(1)
        zp = prev.encode_obs(obs, proprio).flatten(1)
    sim_g = F.cosine_similarity(z, zg, dim=-1) / tau
    sim_p = F.cosine_similarity(z, zp, dim=-1) / tau
    logits = torch.stack([sim_g, sim_p], dim=1)
    tgt = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, tgt)



def feddyn_terms(work, gvec, h_i, alpha):
    """The two FedDyn additions to the local loss.

    Returns -<h_i, theta> + (alpha/2)||theta - theta_g||^2. Both are computed
    over the trainable parameters only, in the order given by named_parameters,
    which is the same order h_i was built in.
    """
    lin = 0.0
    prox = 0.0
    for n, p in work.named_parameters():
        if not p.requires_grad:
            continue
        g = gvec.get(n)
        if g is None:
            continue
        lin = lin - (h_i[n] * p).sum()
        prox = prox + 0.5 * alpha * (p - g).pow(2).sum()
    return lin + prox


def scaffold_correct(work, c_glob, c_i):
    """g <- g - c_i + c, applied in place after backward."""
    for n, p in work.named_parameters():
        if p.grad is None:
            continue
        cg = c_glob.get(n)
        ci = c_i.get(n)
        if cg is None:
            cg = torch.zeros_like(p.grad); c_glob[n] = cg
        if ci is None:
            ci = torch.zeros_like(p.grad); c_i[n] = ci
        p.grad.add_(cg.to(p.grad.device) - ci.to(p.grad.device))


@torch.no_grad()
def scaffold_update_ci(work, gstate, c_glob, c_i, steps, lr):
    """c_i^+ = c_i - c + (x - y_i)/(K*eta); returns the per-name delta."""
    delta = {}
    denom = max(int(steps), 1) * max(float(lr), 1e-12)
    for n, p in work.named_parameters():
        x = gstate.get(n)
        if x is None:
            continue
        cg = c_glob.get(n)
        if cg is None:
            cg = torch.zeros_like(p); c_glob[n] = cg
        old = c_i.get(n)
        if old is None:
            old = torch.zeros_like(p); c_i[n] = old
        new = old - cg.to(p.device) + (x.to(p.device) - p.detach()) / denom
        delta[n] = (new - old).detach().clone()
        c_i[n] = new.detach().clone()
    return delta


def feddyn_update_h(work, gvec, h_i, alpha):
    """h_i <- h_i - alpha (theta_i - theta_g), after local training."""
    with torch.no_grad():
        for n, p in work.named_parameters():
            if not p.requires_grad or n not in h_i:
                continue
            h_i[n].add_(p.detach() - gvec[n], alpha=-alpha)


def feddyn_server(averaged, h_bar, alpha):
    """theta_g <- mean_i(theta_i) - (1/alpha) mean_i(h_i)."""
    out = dict(averaged)
    for n, h in h_bar.items():
        if n in out and torch.is_floating_point(out[n]):
            out[n] = out[n] - h.to(out[n].device) / alpha
    return out


def pair_conflict_stats(acc_i, acc_j, w_i=0.5, w_j=0.5):
    """D = 1 - cos(d_i, d_j) (>1 means anti-correlated) and the cancellation
    ratio C = 1 - |w_i d_i + w_j d_j| / (w_i|d_i| + w_j|d_j|)."""
    names = [n for n in acc_i if n in acc_j]
    if not names:
        return None
    dot = ni2 = nj2 = merged2 = 0.0
    for n in names:
        a = acc_i[n].float()
        b = acc_j[n].float()
        dot += float((a * b).sum())
        ni2 += float((a * a).sum())
        nj2 += float((b * b).sum())
        m = w_i * a + w_j * b
        merged2 += float((m * m).sum())
    ni, nj, merged = ni2 ** 0.5, nj2 ** 0.5, merged2 ** 0.5
    cos = dot / (ni * nj + 1e-12)
    return 1.0 - cos, 1.0 - merged / (w_i * ni + w_j * nj + 1e-12)


def ftp_step(work, batch, opt, states, K, wd, device, bin_counts=None,
             corr=None, gsum=None, gw=None, acc=None):
    """FedSFP-FTP: flow-time-conditioned preconditioning (Adroit port).

    Split the batch by flow time t into K bins; each bin runs its own
    forward/backward (total compute unchanged: sum n_k = B) and owns an
    independent Adam state. One parameter update per batch, weighted by each
    bin's sample share, so different flow-time regimes never share a
    second-moment scale. Mirrors the Push-T ftp_step exactly.
    """
    lr = opt.param_groups[0]["lr"]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    obs = batch["imgs"].to(device, non_blocking=True) if "imgs" in batch \
        else batch["obs"].to(device, non_blocking=True)
    prop = batch["proprio"].to(device, non_blocking=True) if "proprio" in batch else None
    lang = batch["lang"].to(device, non_blocking=True) if "lang" in batch else None
    action = batch["a"].to(device, non_blocking=True)
    t = batch["t"].to(device, non_blocking=True)
    target = batch["v"].to(device, non_blocking=True)
    tv = t.float().view(-1)
    B = int(tv.shape[0])
    bin_idx = torch.clamp((tv * K).long(), 0, K - 1)
    named = [(n, p) for n, p in work.named_parameters() if p.requires_grad]
    upd = {}
    total_loss = 0.0
    for k in range(K):
        idx = (bin_idx == k).nonzero(as_tuple=True)[0]
        nk = int(idx.numel())
        if nk == 0:
            continue
        if bin_counts is not None:
            bin_counts[k] += nk
        work.zero_grad(set_to_none=True)
        if prop is None:
            pred = work(obs[idx], action[idx], t[idx])
        else:
            pred = work(obs[idx], prop[idx], action[idx], t[idx],
                        lang=(lang[idx] if lang is not None else None))
        loss_k = F.mse_loss(pred, target[idx])
        loss_k.backward()
        w = nk / B
        total_loss += w * float(loss_k.detach().cpu())
        st_k = states[k]
        if gsum is not None:
            gw[k] += w
        for n, p in named:
            if p.grad is None:
                continue
            if gsum is not None:
                # FCV: accumulate the RAW bin gradient before any correction,
                # so the control variate lives in gradient units. The drift form
                # c=(x-y)/(K*lr) is in Adam-direction units (~100x larger) and
                # diverged at both 1.0 and 0.5 damping on Push-T.
                if n in gsum[k]:
                    gsum[k][n].add_(p.grad.detach(), alpha=w)
                else:
                    gsum[k][n] = p.grad.detach().clone().mul_(w)
            if corr is not None and n in corr[k]:
                p.grad.add_(corr[k][n])
            st = st_k.get(n)
            if st is None:
                st = {"m_step": 0, "v_step": 0,
                      "exp_avg": torch.zeros_like(p),
                      "exp_avg_sq": torch.zeros_like(p)}
                st_k[n] = st
            # m and v carry INDEPENDENT bias-correction clocks: m_step restarts
            # every round (m is zeroed), v_step runs continuously for the whole
            # run (v persists). Sharing one clock under-corrects the fresh m.
            st["m_step"] += 1
            st["v_step"] += 1
            g = p.grad
            st["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
            st["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            bc1 = 1 - beta1 ** st["m_step"]
            bc2 = 1 - beta2 ** st["v_step"]
            denom = (st["exp_avg_sq"] / bc2).sqrt_().add_(eps)
            direction = (st["exp_avg"] / bc1).div_(denom)
            if acc is not None:
                # FCA/FCC: this bin's share of the applied delta, so the server
                # can aggregate within matched bins / measure per-bin conflict.
                contrib = direction.mul(-lr * w)
                if wd:
                    contrib = contrib.add(p.detach(), alpha=-lr * wd * w)
                if n in acc[k]:
                    acc[k][n].add_(contrib)
                else:
                    acc[k][n] = contrib.clone()
            if n in upd:
                upd[n].add_(direction, alpha=w)
            else:
                upd[n] = direction.mul_(w)
    with torch.no_grad():
        for n, p in named:
            if n in upd:
                if wd:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(upd[n], alpha=-lr)
    work.zero_grad(set_to_none=True)
    return total_loss


def save_global(path, model, full_ds, args, rnd, gstep, ema_model=None):
    torch.save({
        "model_state": model.state_dict(),
        # real EMA shadow when EMA is enabled; falls back to the raw weights so
        # older evaluators that always read ema_model_state keep working
        "ema_model_state": (ema_model or model).state_dict(),
        "optimizer_state": {}, "scheduler_state": {},
        "normalizer": (full_ds.normalizer.state_dict()
                       if getattr(full_ds, "normalizer", None) is not None else None),
        "image_model_config": ({"version": "v2",
                                "num_cameras": max(getattr(full_ds, "img_channels", 3) // 3, 1),
                                "num_kp": getattr(args, "num_kp", 32),
                                "proprio_dim": getattr(full_ds, "proprio_dim", 23),
                                "img_size": getattr(args, "img_size", 84),
                                "cam_channels": 3,
                                "lang_dim": getattr(full_ds, "lang_dim", 0),
                                "flare_experts": getattr(args, "flare_experts", 0),
                                "flare_t_split": getattr(args, "flare_t_split", 0.5)}
                               if getattr(args, "image_encoder", "v1") == "v2" else
                               {"version": "v1",
                                "img_channels": getattr(full_ds, "img_channels", 3),
                                "proprio_dim": getattr(full_ds, "proprio_dim", 23),
                                "img_size": getattr(args, "img_size", 84)}),
        "obs_dim": getattr(full_ds, "obs_dim", None),
        # the evaluator inverts exactly what training applied; without these it
        # would decode poses to the wrong place and never raise
        "imit_target": getattr(full_ds, "imit_target", None),
        "pose_lo": (full_ds.pose_lo.tolist()
                    if getattr(full_ds, "pose_lo", None) is not None else None),
        "pose_hi": (full_ds.pose_hi.tolist()
                    if getattr(full_ds, "pose_hi", None) is not None else None),
        "action_dim": full_ds.action_dim,
        "obs_horizon": full_ds.obs_horizon, "pred_horizon": full_ds.pred_horizon,
        "model_config": {"diffusion_step_embed_dim": _SFP_EMBED, "down_dims": _SFP_DOWN,
                          "kernel_size": 5, "n_groups": 8, "fc_timesteps": 1},
        "table3": {"dataset_variant": args.task_variant,
                    "num_total_demos": len(full_ds.episodes),
                    "num_train_demos": len(full_ds.train_indices),
                    "sigma0": args.sigma0, "k": args.k,
                    "integration_steps_per_action": 1,
                    "gripper_source": "n/a"},
        "args": vars(args), "epoch": rnd, "global_step": gstep, "federated": True,
    }, path)


def main():
    args = parse_args()
    from diffusers.optimization import get_scheduler

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset_class == "libero":
        # the ten spatial configurations are both the data and the partition,
        # so there is no split file to read
        clients_cfg = {}
    else:
        if not args.split_json:
            raise SystemExit("--split-json is required for --dataset-class %s"
                             % args.dataset_class)
        split = json.loads(Path(args.split_json).read_text())
        clients_cfg = split["clients"]
    ds_kw = dict(sigma0=args.sigma0, stabilization_k=args.k, val_ratio=0.0,
                 integration_steps_per_action=1, normalizer_kind=args.normalizer_kind)

    if args.dataset_class == "libero":
        # libero stand-in: one client dataset supplies the dimensions the model
        # factory reads off `full`. It is the same class every client uses, so
        # the widths are correct by construction rather than by assumption.
        from scripts.robomimic_sfp.libero_dataset import build_clients as _lb
        from scripts.robomimic_sfp import libero_dataset as _libds_early
        _libds_early.STATE_ONLY = (args.obs_mode != "image")
        _libds_early.DP_MODE = (args.policy == "dp")
        _libds_early.DP_STEPS = int(args.dp_steps)
        _libds_early.STATE_OBJECTS = bool(args.state_objects)
        _libds_early.OBJECT_SLICES_JSON = args.object_slices
        _libds_early.IMIT_STATE = (args.imitation_mode == "state")
        if _libds_early.IMIT_STATE:
            print("[libero] imitation target: next ee pose (+gripper command)",
                  flush=True)
        if _libds_early.STATE_OBJECTS:
            print("[libero] state+objects: obs = proprio + 7*K object poses "
                  "from data/*/states", flush=True)
        if _libds_early.DP_MODE:
            print("[dp] diffusion policy: %d steps, epsilon target"
                  % args.dp_steps, flush=True)
        if _libds_early.STATE_ONLY:
            print("[libero] state-observation mode: proprio only, "
                  "no cameras", flush=True)
        _probe = _lb(args.libero_root, obs_horizon=2, pred_horizon=16,
                     demo_limit=2,
                     lang_npz=(args.libero_lang or None))[0]["ds"]
        full = _probe
        full.train_indices = list(range(len(_probe)))
        full.biased_gripper_prob = float(args.biased_gripper_prob)
        print("[libero] dims: action=%d proprio=%d img_channels=%d"
              % (full.action_dim, full.proprio_dim, full.img_channels), flush=True)
    elif args.obs_mode == 'image':
        from scripts.robomimic_sfp.adroit_image_dataset import AdroitImageDataset
        full = AdroitImageDataset(args.dataset_path,
                                  obs_horizon=ds_kw.get('obs_horizon', 2),
                                  pred_horizon=ds_kw.get('pred_horizon', 16),
                                  sigma0=ds_kw.get('sigma0', 0.4),
                                  stabilization_k=ds_kw.get('k', 10.0),
                                  normalizer_kind=args.normalizer_kind)
        full.biased_gripper_prob = float(args.biased_gripper_prob)
    else:
        from scripts.robomimic_sfp.adroit_dataset import AdroitFlatDataset, AdroitClientDataset
        from streaming_flow_policy.robomimic_sfp.table3_papergrid_dp_v1 import Table3PaperGridDPDataset
        if args.dataset_class == "table3":
            _kw = {k: v for k, v in ds_kw.items() if k != "normalizer_kind"}
            full = Table3PaperGridDPDataset(args.dataset_path,
                                            gripper_source=args.gripper_source, **_kw)
            if args.normalizer_kind == "official":
                # per-dim min-max on observations as well as actions. The legacy
                # normalizer uses one scalar for all 23 obs dims, which leaves
                # gripper_qpos (raw range 0.041) at ~1/35 the scale of object
                # and eef_pos -- the dimension that says whether the nut is held.
                from streaming_flow_policy.robomimic_sfp.table3_normalizer_official_v1 \
                    import Table3NormalizerOfficial
                import numpy as _np
                _eps = [full.episodes[i] for i in full.train_indices]
                full.normalizer = Table3NormalizerOfficial.fit(
                    _np.concatenate([e["obs"] for e in _eps], axis=0),
                    _np.concatenate([e["action"] for e in _eps], axis=0))
                print("[afed] official per-dim normalizer fitted on %d episodes"
                      % len(_eps), flush=True)
        else:
            full = AdroitFlatDataset(args.dataset_path, **ds_kw)
        full.biased_gripper_prob = float(args.biased_gripper_prob)
    # the image dataset scales proprio internally and exposes no Normalizer
    normalizer = getattr(full, 'normalizer', None)
    print(f"[afed] full demos={len(full.train_indices)} "
          f"obs={getattr(full, 'obs_dim', None) or ('img%d+prop%d' % (full.img_channels, full.proprio_dim))} "
          f"act={full.action_dim}", flush=True)
    steps_budget = json.loads(Path(args.steps_budget_json).read_text()) if args.steps_budget_json else None

    # FedAdam server moments, keyed like the model state dict
    server_m, server_v, server_t = {}, {}, [0]
    if args.dataset_class != "libero":
        missing = [n for n, v in (("--dataset-path", args.dataset_path),
                                  ("--split-json", args.split_json)) if not v]
        if missing:
            raise SystemExit(
                "%s required for --dataset-class %s (only the libero mode "
                "supplies its own data and split)" % (", ".join(missing),
                                                      args.dataset_class))
    clients = []
    if args.dataset_class == "libero":
        from scripts.robomimic_sfp import libero_dataset as _libds
        _libds.STATE_ONLY = (args.obs_mode != "image")
        _libds.DP_MODE = (args.policy == "dp")
        _libds.DP_STEPS = int(args.dp_steps)
        _libds.STATE_OBJECTS = bool(args.state_objects)
        _libds.OBJECT_SLICES_JSON = args.object_slices
        _libds.IMIT_STATE = (args.imitation_mode == "state")
        if _libds.STATE_ONLY:
            print("[libero] state-observation mode: proprio only, no cameras",
                  flush=True)
        from scripts.robomimic_sfp import libero_dataset as _libds
        _libds.STATE_ONLY = (args.obs_mode != "image")
        if _libds.STATE_ONLY:
            print("[libero] state-observation mode: proprio only, no cameras",
                  flush=True)
        from scripts.robomimic_sfp import libero_dataset as _libds
        _libds.STATE_ONLY = (args.obs_mode != "image")
        if _libds.STATE_ONLY:
            print("[libero] state-observation mode: proprio only, no cameras",
                  flush=True)
        # one client per spatial configuration: LIBERO's own partition, not a
        # split file. Sorted filename order keeps client indices stable.
        from scripts.robomimic_sfp.libero_dataset import (
            build_clients, build_clients_byinit)
        # merge is checked before the partition is read: pooling removes the
        # client split, so it cannot be combined with one. The "init" branch
        # below says the same thing; leaving it out here is what let every
        # centralized arm silently train two by-mode clients instead.
        if args.libero_merge_clients:
            print("[libero] centralized: pooling every file into one client, "
                  "partition %r ignored by request" % args.libero_partition,
                  flush=True)
            _mk = lambda **kw: build_clients(
                args.libero_root, demo_limit=args.libero_demo_limit,
                merge=True,
                obs_horizon=kw["obs_horizon"],
                pred_horizon=kw["pred_horizon"],
                lang_npz=kw.get("lang_npz"),
                sigma0=kw.get("sigma0", 0.4),
                stabilization_k=kw.get("stabilization_k", 10.0))
        elif args.libero_partition == "mode":
            from scripts.robomimic_sfp.libero_dataset import (
                build_clients_bymode)
            _mk = lambda **kw: build_clients_bymode(
                args.libero_root,
                obs_horizon=kw["obs_horizon"],
                pred_horizon=kw["pred_horizon"],
                lang_npz=kw.get("lang_npz"),
                n_modes=args.libero_mode_clients,
                mode_feature=args.libero_mode_feature,
                mode_iid=args.libero_mode_iid,
                sigma0=kw.get("sigma0", 0.4),
                stabilization_k=kw.get("stabilization_k", 10.0))
        elif args.libero_partition == "init":
            # merge is handled above for every partition now, so no guard here
            _mk = lambda **kw: build_clients_byinit(
                args.libero_root, n_clients=args.libero_clients,
                obs_horizon=kw["obs_horizon"],
                pred_horizon=kw["pred_horizon"],
                lang_npz=kw.get("lang_npz"),
                sigma0=kw.get("sigma0", 0.4),
                stabilization_k=kw.get("stabilization_k", 10.0))
        else:
            _mk = lambda **kw: build_clients(
                args.libero_root, demo_limit=args.libero_demo_limit,
                merge=args.libero_merge_clients, **kw)
        for c in _mk(obs_horizon=ds_kw.get("obs_horizon", 2),
                               pred_horizon=ds_kw.get("pred_horizon", 16),
                               lang_npz=(args.libero_lang or None),
                               sigma0=ds_kw.get("sigma0", 0.4),
                               stabilization_k=ds_kw.get("k", 10.0)):
            ds = c["ds"]
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=(args.num_workers > 0 and float(getattr(args, "global_anchor_frac", 0.0) or 0.0) <= 0.0),
                                drop_last=True)
            ds.restore_cap = float(getattr(args, "restore_cap", 0.0) or 0.0)
            clients.append(dict(key=c["key"], ds=ds, loader=loader,
                                weight=float(len(ds))))
            print("[libero] %-10s %-52s samples=%d batches=%d"
                  % (c["key"], c["task"][:52], len(ds), len(loader)), flush=True)
        if args.libero_keep_client >= 0:
            if args.libero_keep_client >= len(clients):
                raise SystemExit(
                    "--libero-keep-client %d but the partition made %d clients"
                    % (args.libero_keep_client, len(clients)))
            kept = clients[args.libero_keep_client]
            print("[libero] local-only: keeping client %d (%s), discarding %d"
                  % (args.libero_keep_client, kept["key"], len(clients) - 1),
                  flush=True)
            clients = [kept]
        if not clients:
            raise SystemExit("no LIBERO clients built from %s" % args.libero_root)

    for key in sorted(clients_cfg.keys()) if args.dataset_class != "libero" else []:
        if args.obs_mode == 'image':
            from scripts.robomimic_sfp.adroit_image_dataset import AdroitImageDataset
            ds = AdroitImageDataset(args.dataset_path,
                                    client_demos=clients_cfg[key],
                                    obs_horizon=ds_kw.get('obs_horizon', 2),
                                    pred_horizon=ds_kw.get('pred_horizon', 16),
                                    sigma0=ds_kw.get('sigma0', 0.4),
                                    stabilization_k=ds_kw.get('k', 10.0),
                                    proprio_scale=full.proprio_scale,
                                    normalizer=normalizer,
                                    normalizer_kind=args.normalizer_kind)
            ds.biased_gripper_prob = float(args.biased_gripper_prob)
        else:
            if args.dataset_class == "table3":
                _kw = {k: v for k, v in ds_kw.items() if k != "normalizer_kind"}
                ds = Table3PaperGridDPDataset(args.dataset_path, normalizer=normalizer,
                                              gripper_source=args.gripper_source, **_kw)
                allowed = set(int(x) for x in clients_cfg[key])
                ds.train_indices = [i for i in ds.train_indices if i in allowed]
                ds.indices = [(e, st) for (e, st) in ds.indices if e in allowed]
            else:
                ds = AdroitClientDataset(args.dataset_path, normalizer=normalizer,
                                         client_demos=clients_cfg[key], **ds_kw)
            ds.biased_gripper_prob = float(args.biased_gripper_prob)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0 and float(getattr(args, "global_anchor_frac", 0.0) or 0.0) <= 0.0), drop_last=True)
        ds._agg_n = len(ds.indices)
        clients.append(dict(key=key, ds=ds, loader=loader, weight=len(ds.indices)))
        print(f"[afed] {key}: demos={len(ds.train_indices)} samples={len(ds.indices)} "
              f"batches={len(loader)}", flush=True)

    # published once the loop has built every client, so the anchor
    # refresh can turn a client size into a share without reaching for a
    # name that is out of scope there
    args._client_sizes = [int(c["weight"]) for c in clients]

    def new_model():
        if args.obs_mode == 'image':
            ch = getattr(full, 'img_channels', 3)
            if args.image_encoder == "v3":
                from scripts.robomimic_sfp.sfp_image_model_v3 import SFPImagePolicyV3
                return SFPImagePolicyV3(
                    action_dim=full.action_dim, obs_horizon=2,
                    proprio_dim=full.proprio_dim,
                    num_cameras=max(ch // 3, 1), num_kp=args.num_kp,
                    img_size=args.img_size, cam_channels=3,
                    down_dims=_SFP_DOWN,
                    diffusion_step_embed_dim=_SFP_EMBED).to(device)
            if args.flare_experts > 0:
                from scripts.robomimic_sfp.flare_model import FLAREPolicy
                return FLAREPolicy(
                    action_dim=full.action_dim, obs_horizon=2,
                    proprio_dim=full.proprio_dim,
                    num_cameras=max(ch // 3, 1), num_kp=args.num_kp,
                    img_size=args.img_size, cam_channels=3,
                    lang_dim=getattr(full, "lang_dim", 0),
                    down_dims=_SFP_DOWN,
                    diffusion_step_embed_dim=_SFP_EMBED,
                    n_experts=args.flare_experts,
                    t_split=args.flare_t_split,
                    confidence_gate=args.flare_confidence_gate).to(device)
            if args.image_encoder == "v2":
                from scripts.robomimic_sfp.sfp_image_model_v2 import SFPImagePolicyV2
                return SFPImagePolicyV2(
                    action_dim=full.action_dim, obs_horizon=2,
                    proprio_dim=full.proprio_dim,
                    num_cameras=max(ch // 3, 1), num_kp=args.num_kp,
                    img_size=args.img_size, cam_channels=3,
                    lang_dim=getattr(full, "lang_dim", 0),
                    down_dims=_SFP_DOWN,
                    diffusion_step_embed_dim=_SFP_EMBED).to(device)
            from scripts.robomimic_sfp.adroit_image_model import SFPImagePolicy
            return SFPImagePolicy(
                action_dim=full.action_dim, obs_horizon=full.obs_horizon,
                proprio_dim=full.proprio_dim, img_size=args.img_size,
                img_channels=ch, down_dims=_SFP_DOWN,
                diffusion_step_embed_dim=_SFP_EMBED, fc_timesteps=1).to(device)
        return SFPConditionalUnet1D(obs_dim=full.obs_dim, action_dim=full.action_dim,
                                    obs_horizon=full.obs_horizon, down_dims=_SFP_DOWN,
                                    diffusion_step_embed_dim=_SFP_EMBED, fc_timesteps=1).to(device)

    global_model = new_model()
    if args.feddyn_alpha > 0:
        if args.optimizer_mode in ("persistent", "vp", "pgs", "svr", "opt_only"):
            raise SystemExit(
                "--feddyn-alpha with --optimizer-mode %s would keep two client "
                "memories at once (the FedDyn drift term and the Adam moments), "
                "and their effects cannot be separated afterwards; use raw or "
                "sched_only" % args.optimizer_mode)
        if args.ftp_bins > 0:
            raise SystemExit(
                "--feddyn-alpha with --ftp-bins > 0 has the same problem: FTP "
                "carries per-bin moments across rounds")
    moon_glob = moon_prev = None
    if args.moon_mu > 0:
        if args.obs_mode != "image":
            raise SystemExit(
                "--moon-mu requires --obs-mode image: the state policy feeds raw "
                "observations straight to the UNet, so there is no learned "
                "representation for MOON to align")
        moon_glob = new_model(); moon_prev = new_model()
        for _m in (moon_glob, moon_prev):
            _m.eval()
            for _p in _m.parameters():
                _p.requires_grad_(False)
    work = new_model()
    # EMA shadow of the global weights; each client keeps its own copy for the
    # round and the server FedAvgs them exactly like the raw weights
    global_ema = new_model() if args.ema_decay > 0 else None
    if global_ema is not None:
        global_ema.load_state_dict(global_model.state_dict())
    work_ema = new_model() if args.ema_decay > 0 else None
    gstep = 0
    import math

    def global_lr(step, total):
        # one warmup + a single global profile over the client's WHOLE run
        # (never restarted per round). linear reproduces the Push-T champion.
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        p = (step - args.warmup_steps) / max(total - args.warmup_steps, 1)
        p = min(max(p, 0.0), 1.0)
        if args.lr_shape == "linear":
            return args.lr * (1.0 - p)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p))

    for c in clients:
        if args.steps_per_epoch > 0:
            # canonical FTP3: every client takes the same fixed number of updates
            # per round, so the global LR clock is exactly rounds*epochs*steps
            c["spr"] = args.steps_per_epoch * args.local_epochs
        else:
            c["spr"] = (int(steps_budget[c["key"]]) if steps_budget and c["key"] in steps_budget
                        else len(c["loader"]) * args.local_epochs)
        c["gcount"] = 0
        if args.feddyn_alpha > 0:
            c["h"] = {n: torch.zeros_like(p) for n, p in work.named_parameters()
                      if p.requires_grad}
        if args.ftp_bins > 0:
            # client-local per-flow-bin Adam states, persistent across rounds
            c["ftp_states"] = [dict() for _ in range(args.ftp_bins)]
            c["fcv_ci"] = [dict() for _ in range(args.ftp_bins)]
            c["fca_acc"] = [dict() for _ in range(args.ftp_bins)]
        if args.optimizer_mode in ("persistent", "opt_only", "vp", "pgs", "svr"):
            c["opt"] = torch.optim.AdamW(work.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                         eps=1e-8, weight_decay=args.weight_decay)
        if args.optimizer_mode == "persistent":
            c["sched"] = get_scheduler("cosine", optimizer=c["opt"],
                                       num_warmup_steps=args.warmup_steps,
                                       num_training_steps=max(c["spr"] * args.rounds, 1))

    server_mom = {"m": {}, "v": {}, "step": 0}
    fcv_c = None
    fcc_q = None
    if args.fed_flow == 'fcc' and args.ftp_bins > 0:
        # shared-memory tensor so in-place server updates are visible inside
        # persistent dataloader workers
        fcc_q = torch.full((args.ftp_bins,), 1.0 / args.ftp_bins, dtype=torch.float32)
        fcc_q.share_memory_()
        for _c in clients:
            _c['ds'].flow_time_bin_weights = fcc_q
    _frozen = None
    scaffold_c = {}
    # leave-one-out anchoring needs the individual client models, which
    # only exist AFTER a round of local training; round r therefore uses
    # what round r-1 produced, the same one-round lag the frozen
    # round-start global already has. key -> (weight, cpu state dict).
    _loo_states = {}
    _loo_round = [0]
    for rnd in range(1, args.rounds + 1):
        gstate = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        scaffold_delta = {}
        if args.scaffold:
            for _c in clients:
                _c.setdefault("sc_ci", {})
        if (float(getattr(args, "global_anchor_frac", 0.0) or 0.0) > 0
                and (rnd - 1) % max(int(getattr(args, "global_anchor_refresh", 5) or 5), 1) == 0):
            _mode = getattr(args, "gab_anchor", "global")
            _use_loo = (_mode in ("loo", "self")
                        and _loo_round[0] == rnd - 1
                        and len(_loo_states) == len(clients)
                        and (_mode == "self" or len(clients) > 1))
            if _use_loo:
                # one call per client with a single-element list: the
                # function's own loop then sees exactly one dataset and
                # one model, so no edit inside it is needed. work is
                # reused as the carrier, avoiding a second 69M model.
                for _c in clients:
                    _lo = {}
                    for _n in gstate:
                        # loo takes every OTHER client; self takes only
                        # this client's own previous-round weights
                        _vals = [(_w2, _sd2[_n])
                                 for _k2, (_w2, _sd2) in _loo_states.items()
                                 if ((_k2 != _c["key"]) == (_mode == "loo"))
                                 and _n in _sd2]
                        if not _vals:
                            _lo[_n] = gstate[_n]
                        elif _vals[0][1].dtype.is_floating_point:
                            _den = sum(_w2 for _w2, _ in _vals)
                            _acc2 = None
                            for _w2, _v2 in _vals:
                                _t2 = _w2 * _v2.float()
                                _acc2 = _t2 if _acc2 is None else _acc2 + _t2
                            _lo[_n] = (_acc2 / _den).to(gstate[_n].dtype)
                        else:
                            _lo[_n] = _vals[0][1]
                    work.load_state_dict(_lo)
                    _refresh_gab_tables(args, [_c], work)
                work.load_state_dict(gstate)
                print("[gab] anchor source: %s, round %d"
                      % ("leave-one-out" if _mode == "loo" else "own-previous",
                         rnd), flush=True)
            else:
                _refresh_gab_tables(args, clients, global_model)
                if _mode != "global":
                    print("[gab] anchor source: global (%s not available "
                          "at round %d)" % (_mode, rnd), flush=True)
        _frozen = None
        gstate_ema = ({k: v.detach().clone() for k, v in global_ema.state_dict().items()}
                      if global_ema is not None else None)
        acc = None; wsum = 0.0; losses = []
        if args.feddyn_alpha > 0:
            gvec = {n: v.detach().clone().to(device)
                    for n, v in gstate.items()
                    if n in dict(global_model.named_parameters())}
        if args.moon_mu > 0:
            # the positive reference is this round's starting point, identical
            # for every client, so it is loaded once outside the client loop
            moon_glob.load_state_dict(gstate)

        if args.optimizer_mode == "fmime":
            # Anchors are all computed at gstate, before any client moves, so
            # a_i and a_bar refer to the same point in parameter space.
            anchors = []
            for c in clients:
                work.load_state_dict(gstate)
                work.train()
                a_i, seen, used = flow_balanced_anchor(
                    work, c["loader"], device, args.fmime_anchor_batches, args.fmime_bins)
                anchors.append({n: g.detach().cpu() for n, g in a_i.items()})
                if args.fmime_anchor_noise:
                    # second draw at the SAME parameters: any difference here is
                    # estimator noise, since the client and the point are identical
                    a2, _s2, _u2 = flow_balanced_anchor(
                        work, c["loader"], device, args.fmime_anchor_batches,
                        args.fmime_bins)
                    _nz = float(sum(float((a2[n].detach().cpu() - anchors[-1][n]).pow(2).sum())
                                    for n in anchors[-1]) ** 0.5)
                    _an = float(sum(float(v.pow(2).sum()) for v in anchors[-1].values()) ** 0.5)
                    print("[fmime] round=%d client=%s anchor NOISE ||a1-a2||=%.4e "
                          "||a1||=%.4e ratio=%.3f"
                          % (rnd, c["key"], _nz, _an, _nz / max(_an, 1e-12)), flush=True)
                    del a2
                print("[fmime] round=%d client=%s anchor batches=%d bin_counts=%s"
                      % (rnd, c["key"], used, seen), flush=True)
                if used == 0 or min(seen) == 0:
                    raise SystemExit(
                        "[fmime] client %s produced an empty or unbalanced anchor "
                        "(batches=%d bins=%s); refusing to train on it"
                        % (c["key"], used, seen))
            tw = sum(float(c["weight"]) for c in clients)
            abar = {n: sum(float(c["weight"]) * a[n] for c, a in zip(clients, anchors)) / tw
                    for n in anchors[0]}
            _bn = float(sum(float(v.pow(2).sum()) for v in abar.values()) ** 0.5)
            for c, a_i in zip(clients, anchors):
                c["fmime_corr"] = {n: (abar[n] - a_i[n]) for n in abar}
                _dn = float(sum(float(v.pow(2).sum())
                                for v in c["fmime_corr"].values()) ** 0.5)
                print("[fmime] round=%d client=%s anchor drift ||a_i-abar||=%.4e "
                      "||abar||=%.4e ratio=%.3f"
                      % (rnd, c["key"], _dn, _bn, _dn / max(_bn, 1e-12)), flush=True)
            del anchors
        acc_ema = None
        for c in clients:
            work.load_state_dict(gstate)
            work.train()
            if args.moon_mu > 0:
                # negative: this client's weights from last round. On round 1
                # that is the shared start, so the term begins at log(2) and
                # only bites once the clients have actually drifted apart.
                moon_prev.load_state_dict(c.get("moon_prev", gstate))
            if work_ema is not None:
                work_ema.load_state_dict(gstate_ema)
            steps_total = c["spr"]
            sched = None
            if args.optimizer_mode == "persistent":
                opt, sched = c["opt"], c["sched"]
                _rfm = str(getattr(args, "reset_first_moment", "none"))
                _rref = max(int(getattr(args, "global_anchor_refresh", 10) or 10), 1)
                if _rfm == "round" or (_rfm == "refresh"
                                       and (rnd - 1) % _rref == 0):
                    # v_only: aggregation makes the direction estimate stale
                    # but not the per-parameter scale, so drop m and keep v
                    _z = 0
                    for _st in opt.state.values():
                        if "exp_avg" in _st:
                            _st["exp_avg"].zero_()
                            _z += 1
                    if rnd <= 2 or rnd % 10 == 0:
                        print("[vonly] %s round %d client %s: zeroed m on %d tensors"
                              % (_rfm, rnd, c.get("key", "?"), _z), flush=True)
            elif args.optimizer_mode == "opt_only":
                opt = c["opt"]
                sched = get_scheduler("cosine", optimizer=opt,
                                      num_warmup_steps=min(args.warmup_steps, steps_total // 10),
                                      num_training_steps=max(steps_total, 1))
            elif args.optimizer_mode == "fmime":
                # no per-client optimizer at all: the moments live on the
                # server and the correction lives in c["fmime_corr"]
                opt = None
                fm_corr = (None if args.fmime_no_correction else
                           {n: g.to(device) for n, g in c["fmime_corr"].items()})
            elif args.optimizer_mode in ("vp", "pgs", "svr"):
                # split-moment family: keep the client optimizer (so v
                # persists) but zero the first moment every round, and drive
                # the LR from the global clock rather than a fresh scheduler.
                opt = c["opt"]
                for _st in opt.state.values():
                    if "exp_avg" in _st:
                        _st["exp_avg"].zero_()
                    if "step" in _st and args.optimizer_mode != "persistent":
                        pass
                if args.optimizer_mode == "svr" and rnd >= args.svr_refresh_after:
                    for _st in opt.state.values():
                        if "exp_avg_sq" in _st:
                            _st["exp_avg_sq"].zero_()
            elif args.optimizer_mode in ("sched_only", "const", "tailanneal"):
                # const: literature-convention fixed LR (one-time warmup, then flat);
                # differs from sched_only ONLY in the post-warmup profile
                opt = torch.optim.AdamW(work.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                        eps=1e-8, weight_decay=args.weight_decay)
            else:
                opt = torch.optim.AdamW(work.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                        eps=1e-8, weight_decay=args.weight_decay)
                sched = get_scheduler("cosine", optimizer=opt,
                                      num_warmup_steps=min(args.warmup_steps, steps_total // 10),
                                      num_training_steps=max(steps_total, 1))
            if args.ftp_bins > 0:
                # round boundary: m zeroed and its clock restarted; v and its
                # clock carry over untouched (canonical FTP3 state rule)
                for _stk in c["ftp_states"]:
                    for _st in _stk.values():
                        _st["exp_avg"].zero_()
                        _st["m_step"] = 0
                c["fca_acc"] = [dict() for _ in range(args.ftp_bins)]
                c["gsum"] = [dict() for _ in range(args.ftp_bins)]
                c["gw"] = [0.0] * args.ftp_bins
                c["corr"] = None
                if args.fed_flow == "fcv" and fcv_c is not None:
                    c["corr"] = [
                        {n: (fcv_c[b][n] - c["fcv_ci"][b].get(
                            n, torch.zeros_like(fcv_c[b][n]))).to(device)
                         for n in fcv_c[b]}
                        for b in range(args.ftp_bins)]
                print("[gate] round=%d client=%s boundary %s" % (
                    rnd, c["key"],
                    " | ".join(
                        "b%d:m=%.3e m_step=%d v=%.3e v_step=%d n=%d" % (
                            _b,
                            sum(float(s["exp_avg"].abs().sum()) for s in _stk.values()),
                            max((s["m_step"] for s in _stk.values()), default=0),
                            sum(float(s["exp_avg_sq"].abs().sum()) for s in _stk.values()),
                            max((s["v_step"] for s in _stk.values()), default=0),
                            c.get("bin_counts", [0] * args.ftp_bins)[_b])
                        for _b, _stk in enumerate(c["ftp_states"]))),
                    flush=True)
                c["bin_counts"] = [0] * args.ftp_bins
            done = 0
            while done < steps_total:
                for batch in c["loader"]:
                    if done >= steps_total:
                        break
                    if args.optimizer_mode == "fmime":
                        lr_now = global_lr(c["gcount"], c["spr"] * args.rounds)
                    elif args.optimizer_mode in ("sched_only", "vp", "pgs", "svr"):
                        lr_now = global_lr(c["gcount"], c["spr"] * args.rounds)
                        for g in opt.param_groups:
                            g["lr"] = lr_now
                    elif args.optimizer_mode == "const":
                        lr_now = (args.lr * c["gcount"] / max(args.warmup_steps, 1)
                                  if c["gcount"] < args.warmup_steps else args.lr)
                        for g in opt.param_groups:
                            g["lr"] = lr_now
                    elif args.optimizer_mode == "tailanneal":
                        # area-matched: global cosine envelope g(r) x per-round
                        # tail shape h(k) (h=1 until 90%, linear->0), normalized
                        # by mean(h)=0.95 so per-round MEAN lr == raw-global's.
                        genv = global_lr(c["gcount"], c["spr"] * args.rounds)
                        tail = max(1, int(0.1 * steps_total))
                        hshape = (1.0 if done < steps_total - tail
                                  else (steps_total - done) / tail)
                        lr_now = genv * hshape / 0.95
                        for g in opt.param_groups:
                            g["lr"] = lr_now
                    if args.ftp_bins > 0:
                        loss_val = ftp_step(work, batch, opt, c["ftp_states"],
                                            args.ftp_bins, args.weight_decay, device,
                                            bin_counts=c.setdefault(
                                                "bin_counts", [0] * args.ftp_bins),
                                            corr=c.get("corr"),
                                            gsum=(c["gsum"] if args.fed_flow == "fcv" else None),
                                            gw=(c["gw"] if args.fed_flow == "fcv" else None),
                                            acc=(c["fca_acc"] if args.fed_flow in ("fca", "fcc") else None))
                        if work_ema is not None:
                            ema_update(work_ema, work, args.ema_decay)
                        gstep += 1; done += 1; c["gcount"] += 1
                        losses.append(loss_val)
                        continue
                    if args.optimizer_mode == "fmime":
                        loss_val = fmime_step(work, batch, lr_now, args.weight_decay,
                                              device, server_mom, fm_corr)
                        if work_ema is not None:
                            ema_update(work_ema, work, args.ema_decay)
                        gstep += 1; done += 1; c["gcount"] += 1
                        losses.append(loss_val)
                        continue
                    obs = (batch["imgs"] if "imgs" in batch else batch["obs"]).to(
                        device, non_blocking=True)
                    action = batch["a"].to(device, non_blocking=True)
                    t = batch["t"].to(device, non_blocking=True)
                    target = batch["v"].to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    lang_b = (batch['lang'].to(device, non_blocking=True)
                              if 'lang' in batch else None)
                    _rw = None
                    if args.residual_gate > 0:
                        # the frozen model is the round's starting point,
                        # so 'what the global already knows' is fixed for
                        # the whole round rather than chasing the client
                        if _frozen is None:
                            import copy as _copy
                            _frozen = _copy.deepcopy(work).eval()
                            for _q in _frozen.parameters():
                                _q.requires_grad_(False)
                        with torch.no_grad():
                            _pb = batch['proprio'].to(device, non_blocking=True)
                            _lb = (batch['lang'].to(device, non_blocking=True)
                                   if 'lang' in batch else None)
                            _fp = _frozen(obs, _pb, action, t, lang=_lb)
                            _e = ((_fp - target) ** 2).flatten(1).mean(1)
                            _rw = 1.0 + args.residual_gate * (
                                _e / (_e.mean() + 1e-8) - 1.0)
                            _rw = _rw.clamp(min=0.0)
                    if args.flare_experts > 0:
                        from scripts.robomimic_sfp.flare_model import gate_loss
                        prop_b = batch['proprio'].to(device, non_blocking=True)
                        # the client index is the routing label the
                        # split hands out for free
                        eidx = torch.full(
                            (obs.shape[0],), int(c['key'].split('_')[-1]),
                            device=device, dtype=torch.long)
                        _p, _g = work(obs, prop_b, action, t, lang=lang_b,
                                      expert=eidx, return_logits=True)
                        loss = (F.mse_loss(_p, target)
                                + args.flare_gate_weight * gate_loss(_g, eidx))
                    elif 'proprio' in batch:
                        prop_b = batch['proprio'].to(device, non_blocking=True)
                        _pred = work(obs, prop_b, action, t, lang=lang_b)
                        if _rw is not None:
                            _per = ((_pred - target) ** 2).flatten(1).mean(1)
                            loss = (_per * _rw).sum() / (_rw.sum() + 1e-8)
                        else:
                            loss = F.mse_loss(_pred, target)
                    else:
                        prop_b = None
                        loss = F.mse_loss(work(obs, action, t), target)
                    total_loss = loss
                    if args.fedprox_mu > 0:
                        _p = sum(((p - gstate[_n].to(p.device)) ** 2).sum()
                                 for _n, p in work.named_parameters()
                                 if _n in gstate)
                        total_loss = total_loss + 0.5 * args.fedprox_mu * _p
                    if args.feddyn_alpha > 0:
                        total_loss = total_loss + feddyn_terms(
                            work, gvec, c["h"], args.feddyn_alpha)
                    if args.moon_mu > 0:
                        total_loss = total_loss + args.moon_mu * moon_obs_loss(
                            work, moon_glob, moon_prev, obs, prop_b, args.moon_tau)
                    total_loss.backward()
                    if args.scaffold:
                        scaffold_correct(work, scaffold_c, c["sc_ci"])
                    opt.step()
                    if sched is not None:
                        sched.step()
                    if work_ema is not None:
                        ema_update(work_ema, work, args.ema_decay)
                    gstep += 1; done += 1; c["gcount"] += 1
                    losses.append(float(loss.detach().cpu()))
            if args.feddyn_alpha > 0:
                feddyn_update_h(work, gvec, c["h"], args.feddyn_alpha)
            if args.scaffold:
                _d = scaffold_update_ci(work, gstate, scaffold_c,
                                        c["sc_ci"], done, args.lr)
                for _k, _v in _d.items():
                    scaffold_delta[_k] = scaffold_delta.get(_k, 0) + _v
            w = (1.0 if getattr(args, "equal_agg_weight", False)
                 else float(c["weight"]))
            wsum += w
            if args.moon_mu > 0:
                c["moon_prev"] = {k: v.detach().clone()
                                  for k, v in work.state_dict().items()}
            cs = work.state_dict()
            if getattr(args, "gab_anchor", "global") in ("loo", "self"):
                # CPU copies: cs aliases work, which the next client
                # overwrites. ~278 MB per client for the image model.
                _loo_states[c["key"]] = (
                    w, {_n: _v.detach().to("cpu", copy=True)
                        for _n, _v in cs.items()})
                _loo_round[0] = rnd
            if args.dump_clients_at_round and rnd == args.dump_clients_at_round:
                import os as _os
                _p = _os.path.join(args.output_dir,
                                   "clientdump_r%d_%s.pt" % (rnd, c["key"]))
                save_global(_p, work, full, args, rnd, 0, None)
                print("[dump] wrote %s" % _p, flush=True)
            if acc is None:
                acc = {k: cs[k].detach().clone().float() * w for k in cs}
            else:
                for k in cs:
                    acc[k] += cs[k].detach().float() * w
            if work_ema is not None:
                es = work_ema.state_dict()
                if acc_ema is None:
                    acc_ema = {k: es[k].detach().clone().float() * w for k in es}
                else:
                    for k in es:
                        acc_ema[k] += es[k].detach().float() * w
        if args.fed_flow == 'fcv' and args.ftp_bins > 0:
            # gradient-unit control variates (SCAFFOLD Option-I):
            # c_ib = mean raw bin gradient, c_b = cross-client mean
            for c in clients:
                c['fcv_ci'] = [
                    {n: (g / max(c['gw'][b], 1e-12)).detach().cpu()
                     for n, g in c['gsum'][b].items()}
                    for b in range(args.ftp_bins)]
            fcv_c = []
            for b in range(args.ftp_bins):
                names = set()
                for c in clients:
                    names.update(c['fcv_ci'][b].keys())
                fcv_c.append({n: sum(c['fcv_ci'][b][n] for c in clients
                                     if n in c['fcv_ci'][b]) / max(len(clients), 1)
                              for n in names})
        if args.fed_flow == 'fcc' and args.ftp_bins > 0:
            # D_b = 1 - mean pairwise cos of the per-bin client deltas;
            # q_b propto clip(D_b^alpha) drives next round's t sampling
            D = []
            for b in range(args.ftp_bins):
                vals = []
                for i in range(len(clients)):
                    for j in range(i + 1, len(clients)):
                        st = pair_conflict_stats(clients[i]['fca_acc'][b],
                                                 clients[j]['fca_acc'][b])
                        if st is not None:
                            vals.append(st[0])
                D.append(sum(vals) / len(vals) if vals else 0.0)
            raw = [max(d, 1e-6) ** args.fcc_alpha for d in D]
            q = [r / sum(raw) for r in raw]
            lo, hi = 0.5 / args.ftp_bins, 2.0 / args.ftp_bins
            q = [min(max(x, lo), hi) for x in q]
            q = [x / sum(q) for x in q]
            fcc_q.copy_(torch.tensor(q, dtype=torch.float32))
            rec_fcc = {'D': [round(x, 4) for x in D], 'q': [round(x, 4) for x in q]}
            print('[fcc] round=%d %s' % (rnd, rec_fcc), flush=True)
        if args.optimizer_mode == "fmime":
            fmime_server_update(server_mom, abar, device)
            print("[fmime] round=%d server moments advanced (step=%d)"
                  % (rnd, server_mom["step"]), flush=True)
        if args.optimizer_mode == 'pgs':
            # v_bar = mean_i v_i; each client's next-round v = v_bar + lam(v_i - v_bar)
            named = [[(n, p) for n, p in c['policy'].named_parameters()]
                     if 'policy' in c else None for c in clients]
            vlists = []
            for c in clients:
                opt = c['opt']
                vlists.append({id(p): opt.state[p]['exp_avg_sq']
                               for p in opt.state if 'exp_avg_sq' in opt.state[p]})
            keys = list(vlists[0].keys()) if vlists else []
            if keys:
                mats = [[vl[k] for vl in vlists if k in vl] for k in keys]
                for k, tensors in zip(keys, mats):
                    if len(tensors) != len(clients):
                        continue
                    vbar = torch.stack(tensors, dim=0).mean(dim=0)
                    for t in tensors:
                        t.copy_(vbar + args.pgs_lambda * (t - vbar))
        if args.fed_flow == 'fca' and args.ftp_bins > 0:
            # aggregate clients WITHIN matched flow bins, then combine bins.
            # sum_b acc_b == the client's whole delta, so the clientsize
            # weighting reduces exactly to FedAvg (the control).
            totals = [sum(c['bin_counts'][b] for c in clients)
                      for b in range(args.ftp_bins)]
            wsum_all = sum(float(c['weight']) for c in clients)
            merged = {k: v.detach().clone().float() for k, v in gstate.items()}
            for b in range(args.ftp_bins):
                for c in clients:
                    if args.fca_weight == 'binshare':
                        wb = c['bin_counts'][b] / max(totals[b], 1)
                    else:
                        wb = float(c['weight']) / max(wsum_all, 1.0)
                    if wb == 0.0:
                        continue
                    for n, d in c['fca_acc'][b].items():
                        if n in merged and torch.is_floating_point(merged[n]):
                            merged[n] = merged[n] + wb * d.to(merged[n].device)
            acc = {k: merged[k] * wsum for k in merged}
        averaged = {k: acc[k] / wsum for k in acc}
        if args.flare_experts > 0:
            # only client i writes row i, so the mean returns
            # g + (w_i/wsum)*delta_i and shrinks each client's own update
            # by its weight share; divide that share back out
            for _k in averaged:
                if not _k.startswith(("expert_scale", "expert_shift")):
                    continue
                _g = gstate[_k].detach().float().to(averaged[_k].device)
                for _i, _c in enumerate(clients):
                    if _i >= averaged[_k].shape[0]:
                        break
                    _s = wsum / max(float(_c["weight"]), 1e-12)
                    averaged[_k][_i] = _g[_i] + _s * (averaged[_k][_i] - _g[_i])
        if args.feddyn_alpha > 0:
            h_bar = {n: sum(c["h"][n] for c in clients) / max(len(clients), 1)
                     for n in clients[0]["h"]}
            averaged = feddyn_server(averaged, h_bar, args.feddyn_alpha)
            print("[feddyn] round=%d ||h_bar||=%.4e ||theta-theta_g||=%.4e" % (
                rnd,
                float(sum(float(v.pow(2).sum()) for v in h_bar.values()) ** 0.5),
                float(sum(float((averaged[n].to(device) - gvec[n]).pow(2).sum())
                          for n in gvec if n in averaged) ** 0.5)), flush=True)
        if args.server_optimizer == "fedavgm":
            # server momentum over the pseudo-gradient (average - global)
            cur = global_model.state_dict()
            for k in averaged:
                if not torch.is_floating_point(averaged[k]):
                    continue
                d = averaged[k] - cur[k].detach().float()
                if k not in server_m:
                    server_m[k] = torch.zeros_like(d)
                server_m[k].mul_(args.server_beta1).add_(d)
                averaged[k] = cur[k].detach().float() \
                    + args.server_lr * server_m[k]
        if args.server_optimizer == "fedadam":
            # (average - global) is the pseudo-gradient; only float tensors
            # participate, since integer buffers have no meaningful moments
            cur = global_model.state_dict()
            server_t[0] += 1
            b1, b2 = args.server_beta1, args.server_beta2
            bc1 = 1 - b1 ** server_t[0]
            bc2 = 1 - b2 ** server_t[0]
            for k in averaged:
                if not torch.is_floating_point(averaged[k]):
                    continue
                d = averaged[k] - cur[k].detach().float()
                if k not in server_m:
                    server_m[k] = torch.zeros_like(d)
                    server_v[k] = torch.zeros_like(d)
                server_m[k].mul_(b1).add_(d, alpha=1 - b1)
                server_v[k].mul_(b2).addcmul_(d, d, value=1 - b2)
                step = args.server_lr * (server_m[k] / bc1) / (
                    (server_v[k] / bc2).sqrt() + args.server_eps)
                averaged[k] = cur[k].detach().float() + step
        if args.scaffold and scaffold_delta:
            _n = max(len(clients), 1)
            for _k, _v in scaffold_delta.items():
                scaffold_c[_k] = scaffold_c.get(_k, torch.zeros_like(_v)) + _v / _n
        global_model.load_state_dict({k: averaged[k].to(
            dtype=global_model.state_dict()[k].dtype) for k in averaged})
        if global_ema is not None and acc_ema is not None:
            global_ema.load_state_dict({k: (acc_ema[k] / wsum).to(dtype=global_ema.state_dict()[k].dtype)
                                        for k in acc_ema})
        rec = dict(round=rnd, global_step=gstep, loss=float(np.mean(losses)))
        print(json.dumps(rec), flush=True)
        with (out / "metrics.jsonl").open("a") as s:
            s.write(json.dumps(rec) + "\n")
        if rnd % args.save_every_rounds == 0 or rnd == args.rounds:
            save_global(out / f"global_round_{rnd:04d}.pt", global_model, full, args, rnd, gstep,
                        ema_model=global_ema)
            save_global(out / "latest.pt", global_model, full, args, rnd, gstep,
                        ema_model=global_ema)
    print("[afed] DONE", flush=True)


if __name__ == "__main__":
    main()
