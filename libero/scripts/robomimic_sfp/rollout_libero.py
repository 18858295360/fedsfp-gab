#!/usr/bin/env python3
"""Rollout evaluator for LIBERO SFP checkpoints.

Each query integrates the velocity field from the previous chunk endpoint
using the latest observation history. The evaluator clips commands sent to
the environment and carries the unclipped endpoint into the next query.

An episode succeeds if LIBERO's BDDL goal predicate is reached at any step.
The evaluator reports per-task and pooled success rates.
"""
import argparse
import json
import os
from collections import deque

import numpy as np
import torch

STATE_OBS = [False]
OBJ_PAD = [False]
FLAT_LANG = [None]      # the task's embedding, appended to flat observations
IMIT = {'state': False, 'lo': None, 'hi': None,
        'gain': None, 'bias': None}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--libero-root", default="")
    p.add_argument("--rollouts-per-task", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--action-horizon", type=int, default=8)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--start-seed", type=int, default=100000)
    p.add_argument("--dump-traj", default="",
                   help="npz path: record the end-effector trajectory of every "
                        "rollout. Two runs with the same seeds start from the "
                        "same designated states, so their divergence is the "
                        "policy difference alone")
    p.add_argument("--shuffle-lang", action="store_true",
                   help="give each task the next task's "
                        "instruction; the score must collapse "
                        "if language is what is doing the work")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--benchmark", default="",
                   help="libero_spatial|libero_object|libero_goal|libero_10; "
                        "empty means read it from the checkpoint")
    p.add_argument("--lang-npz", default="",
                   help="instruction embeddings keyed by task name; read "
                        "only when the checkpoint carries lang_dim")
    return p.parse_args()


_FLAT_MODES = ("flat", "state", "lowdim")
_IMAGE_MODES = ("image",)


def build_model(ck, device):
    """Rebuild whatever the checkpoint describes, image or state.

    STATE_OBS is set here so construction and the later forward call can never
    disagree about which one this is.
    """
    _a = ck.get("args") or {}
    _om = str(_a.get("obs_mode", "image"))
    if _om in _FLAT_MODES:
        STATE_OBS[0] = True
    elif _om in _IMAGE_MODES:
        STATE_OBS[0] = False
    else:
        raise SystemExit(
            "checkpoint declares obs_mode=%r, which this evaluator does not "
            "recognise. Known: %s (state), %s (image). Refusing to guess."
            % (_om, "/".join(_FLAT_MODES), "/".join(_IMAGE_MODES)))

    if STATE_OBS[0]:
        mc = ck.get("model_config", {}) or {}
        from streaming_flow_policy.robomimic_sfp.model_v22_paper_aligned \
            import SFPConditionalUnet1D
        kw = dict(obs_dim=ck["obs_dim"], action_dim=ck["action_dim"],
                  obs_horizon=ck.get("obs_horizon", 2),
                  down_dims=mc.get("down_dims", [256, 512, 1024]),
                  diffusion_step_embed_dim=mc.get(
                      "diffusion_step_embed_dim", 256),
                  fc_timesteps=mc.get("fc_timesteps", 1))
        print("  rebuilding flat policy (obs_mode=%s): %s" % (_om, kw),
              flush=True)
        m = SFPConditionalUnet1D(**kw).to(device)
        m.load_state_dict(ck.get("ema_model_state") or ck["model_state"])
        m.eval()
        return m

    cfg = ck.get("image_model_config", {}) or {}
    ver = cfg.get("version", "v2")
    # every image-model dimension lives in image_model_config; only action_dim,
    # obs_horizon and pred_horizon are at the top level
    kw = dict(action_dim=ck["action_dim"],
              obs_horizon=ck.get("obs_horizon", 2),
              proprio_dim=cfg["proprio_dim"],
              num_cameras=cfg.get("num_cameras", 2),
              num_kp=cfg.get("num_kp", 32),
              img_size=cfg.get("img_size", 128),
              cam_channels=cfg.get("cam_channels", 3),
              lang_dim=cfg.get("lang_dim", 0))
    print("  rebuilding %s encoder: %s" % (ver, kw), flush=True)
    if cfg.get("flare_experts", 0):
        from scripts.robomimic_sfp.flare_model import FLAREPolicy as M
        kw["n_experts"] = cfg["flare_experts"]
        kw["t_split"] = cfg.get("flare_t_split", 0.5)
        m = M(**kw).to(device)
        m.load_state_dict(ck.get("ema_model_state") or ck["model_state"])
        m.eval()
        print("  FLARE: %d experts, t_split %.2f, gate routes at inference"
              % (kw["n_experts"], kw["t_split"]), flush=True)
        return m
    if ver == "v3":
        from scripts.robomimic_sfp.sfp_image_model_v3 import SFPImagePolicyV3 as M
    else:
        from scripts.robomimic_sfp.sfp_image_model_v2 import SFPImagePolicyV2 as M
    m = M(**kw).to(device)
    m.load_state_dict(ck.get("ema_model_state") or ck["model_state"])
    m.eval()
    return m


def _aa_to_R(v):
    th = float(np.linalg.norm(v))
    if th < 1e-9:
        return np.eye(3, dtype=np.float64)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
                 dtype=np.float64)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _R_to_aa(R):
    c = float(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
                 dtype=np.float64)
    n = float(np.linalg.norm(v))
    return (v / n * th) if n > 1e-9 else np.zeros(3)


def _norm_pose(cur_pose):
    """Current absolute pose -> the normalised space the flow was trained in.
    Exactly the inverse of the un-normalisation at the top of _decode_pose."""
    lo, hi = IMIT["lo"], IMIT["hi"]
    n = 2.0 * (np.asarray(cur_pose[:6], dtype=np.float32) - lo) / (hi - lo) - 1.0
    return np.concatenate([n, np.zeros(1, dtype=np.float32)]).astype(np.float32)


def _decode_pose(pred, cur_pose):
    """Normalised next pose (7) + current absolute pose (6) -> action (7)."""
    lo, hi = IMIT["lo"], IMIT["hi"]
    absn = (pred[:6] + 1.0) * 0.5 * (hi - lo) + lo
    d = np.empty(6, dtype=np.float64)
    d[:3] = absn[:3] - cur_pose[:3]
    d[3:] = _R_to_aa(_aa_to_R(absn[3:6]) @ _aa_to_R(cur_pose[3:6]).T)
    arm = IMIT["gain"] * d + IMIT["bias"]
    return np.concatenate([arm, pred[6:7]]).astype(np.float32)


@torch.inference_mode()
def predict_chunk(model, imgs, prop, previous, horizon, pred_horizon, device,
                  lang=None):
    """Euler-integrate the velocity field from the previous action."""
    I = torch.from_numpy(imgs).unsqueeze(0).float().to(device)
    P = torch.from_numpy(prop).unsqueeze(0).float().to(device)
    a = torch.from_numpy(previous).unsqueeze(0).float().to(device)
    dt = 1.0 / pred_horizon
    out = []
    for step in range(horizon):
        t = torch.full((1,), step * dt, dtype=torch.float32, device=device)
        if STATE_OBS[0]:
            a = a + dt * model(P, a, t)
        else:
            a = a + dt * model(I, P, a, t, lang=lang)
        out.append(a[0].cpu().numpy().copy())
    return np.stack(out), a[0].cpu().numpy()


def main():
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import sys
    if args.libero_root:
        sys.path.insert(0, args.libero_root)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ck, device)

    # checkpoint contract: everything that changes how a policy must be run is
    # read from the checkpoint, printed, and refused if unsupported
    _ck_all = {}
    try:
        _ck_all = ck.get("args") or {}
    except Exception:
        _ck_all = {}
    _obs_mode = str(_ck_all.get("obs_mode", "image"))
    _policy = str(_ck_all.get("policy", "sfp"))
    print("  checkpoint: obs_mode=%s policy=%s lang=%s"
          % (_obs_mode, _policy, os.path.basename(
              str(_ck_all.get("libero_lang") or "none"))), flush=True)
    if _policy != "sfp":
        raise SystemExit(
            "this checkpoint was trained with policy=%s; the evaluator only "
            "integrates flows. Refusing to score it with the wrong sampler."
            % _policy)
    # STATE_OBS was set by build_model from the same field; not reassigned
    if str(_ck_all.get("imitation_mode", "action")) == "state":
        _tgt = ck.get("imit_target")
        if _tgt not in ("virtual_action_v2", "virtual_action_v3"):
            raise SystemExit(
                "imitation_mode=state checkpoint without the virtual_action_v2 "
                "marker (found %r). The v1 absolute-pose target is broken by "
                "construction -- sigma0 0.4 buries the ~0.03 per-step pose "
                "delta, and its policies measured worse than stay-put on every "
                "dimension -- so scoring it can only produce a plausible zero. "
                "Retrain under v2 instead." % _tgt)
        print("  imitation target v2: virtual actions from pose transitions; "
              "standard action-space rollout, no decoding", flush=True)
    _prop_dim_ck = int((ck.get("image_model_config") or {}).get("proprio_dim", 15))
    _state_objects = bool(_ck_all.get("state_objects")) and STATE_OBS[0]
    if STATE_OBS[0]:
        # The object block width comes from the slice map, which records how
        # many free objects each suite's tasks carry. image_model_config is not
        # usable here: flat arms pass no --image-encoder, so the trainer writes
        # its v1 form, which has no lang_dim key and silently reads as zero.
        def _K_from_slices(bench):
            import json as _j
            p = str(_ck_all.get("object_slices") or "")
            if not os.path.exists(p):
                return None
            ent = _j.load(open(p)).get(bench) or {}
            ks = {len(v["objects"]) for v in ent.values()}
            if not ks:
                return None
            # a suite whose tasks differ trains on K_max padded slots plus a
            # K_max presence mask; a uniform suite keeps the bare 7*K layout
            OBJ_PAD[0] = (len(ks) > 1)
            return max(ks)

        _obj_dims_ck = int(ck["obs_dim"]) - _prop_dim_ck
        # OBJ_PAD is decided for every state+objects checkpoint, language or
        # not: it describes the object block's layout, and the arms that need
        # padding most (libero_10) carry no instruction at all.
        if _state_objects:
            _bench_pad = os.path.basename(
                str(_ck_all.get("libero_root", ""))) or "libero_spatial"
            _K_pad = _K_from_slices(_bench_pad)
            if _K_pad is not None and OBJ_PAD[0]:
                _obj_dims_ck = 8 * _K_pad
        if _state_objects and str(_ck_all.get("libero_lang") or ""):
            _bench_guess = os.path.basename(
                str(_ck_all.get("libero_root", ""))) or "libero_spatial"
            _K = _K_from_slices(_bench_guess)
            if _K is None:
                raise SystemExit(
                    "cannot determine the object count for %r from the slice "
                    "map, so the language block in a %d-dim observation cannot "
                    "be separated from the object block"
                    % (_bench_guess, int(ck["obs_dim"])))
            _obj_dims_ck = (8 * _K) if OBJ_PAD[0] else (7 * _K)
            FLAT_LANG[0] = int(ck["obs_dim"]) - _prop_dim_ck - _obj_dims_ck
            if FLAT_LANG[0] <= 0:
                raise SystemExit(
                    "obs_dim %d leaves %d dims for the instruction after "
                    "proprio %d and %d object poses; the widths disagree"
                    % (int(ck["obs_dim"]), FLAT_LANG[0], _prop_dim_ck, _K))
            print("  flat observation: proprio %d + %d objects x%s + %d-dim "
                  "instruction"
                  % (_prop_dim_ck, _K, "7+mask" if OBJ_PAD[0] else "7",
                     FLAT_LANG[0]), flush=True)
        _unit = 8 if OBJ_PAD[0] else 7
        if _state_objects and (_obj_dims_ck <= 0 or _obj_dims_ck % _unit):
            raise SystemExit(
                "checkpoint declares state_objects but obs_dim %d - proprio %d "
                "is not a positive multiple of %d"
                % (ck["obs_dim"], _prop_dim_ck, _unit))
        if not _state_objects and _obj_dims_ck != 0:
            raise SystemExit(
                "checkpoint obs_dim %d exceeds proprio %d without declaring "
                "state_objects; refusing to guess the layout"
                % (ck["obs_dim"], _prop_dim_ck))
        if _state_objects:
            print("  state+objects: %d object slots will come from the live "
                  "sim's qpos%s"
                  % (_obj_dims_ck // (8 if OBJ_PAD[0] else 7),
                     " (padded, with presence mask)" if OBJ_PAD[0] else ""),
                  flush=True)
    if STATE_OBS[0]:
        print("  state observations: proprioception only, cameras unused",
              flush=True)

    _ck_args = {}
    try:
        _ck_args = ck.get("args") or {}
    except Exception:
        _ck_args = {}
    _ck_lang = str(_ck_args.get("libero_lang") or "")
    if _ck_lang:
        args.lang_npz = args.lang_npz or _ck_lang
        if not os.path.isfile(args.lang_npz):
            raise SystemExit("Pass --lang-npz pointing to the same embeddings used for training.")
    else:
        print("  checkpoint used no language; ignoring --lang-npz", flush=True)
        args.lang_npz = ""
    oh = ck.get("obs_horizon", 2)
    ph = ck.get("pred_horizon", 16)

    # v1 image_model_config carries no lang_dim, which is what every flat arm
    # gets; fall back to the width the contract resolved from the slice map
    lang_dim = (ck.get("image_model_config", {}) or {}).get("lang_dim", 0) \
        or (FLAT_LANG[0] or 0)
    lang_table = None
    if lang_dim:
        import numpy as _np
        z = _np.load(args.lang_npz, allow_pickle=True)
        lang_table = {str(t): v.astype("float32")
                      for t, v in zip(z["tasks"], z["emb"])}
        print("  language conditioning on, %d instructions"
              % len(lang_table), flush=True)

    _bench = args.benchmark
    if not _bench:
        _root = ""
        try:
            _root = str((ck.get("args") or {}).get("libero_root", ""))
        except Exception:
            _root = ""
        for _cand in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            if _cand in _root:
                _bench = _cand
                break
        _bench = _bench or "libero_spatial"
        print("  benchmark from checkpoint: %s" % _bench, flush=True)
    suite = benchmark.get_benchmark_dict()[_bench]()
    per_task, all_flags = {}, []

    if args.shuffle_lang:
        if lang_table is None:
            raise SystemExit("--shuffle-lang on a checkpoint with no language")
        _names = [os.path.basename(suite.get_task(i).bddl_file).replace(".bddl", "")
                  for i in range(min(args.tasks, suite.n_tasks))]
        _rot = {n: lang_table[_names[(i + 1) % len(_names)]]
                for i, n in enumerate(_names)}
        lang_table = dict(lang_table)
        lang_table.update(_rot)
        print("  SHUFFLED: each task receives the next task's instruction",
              flush=True)

    for ti in range(min(args.tasks, suite.n_tasks)):
        task = suite.get_task(ti)
        task_lang = None
        if lang_table is not None:
            key = os.path.basename(task.bddl_file).replace(".bddl", "")
            if key not in lang_table:
                raise SystemExit(
                    "no instruction embedding for task %r; the lookup is "
                    "keyed by the bddl basename" % key)
            task_lang = torch.from_numpy(
                lang_table[key]).unsqueeze(0).to(device)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=args.img_size,
                                 camera_widths=args.img_size)

        # the benchmark's designated evaluation states, as its own evaluator
        # uses. torch 2.6 defaults weights_only True and cannot unpickle the
        # numpy array they ship, so it is forced off for this one call.
        _orig_load = torch.load
        torch.load = lambda *a, **k: _orig_load(
            *a, **{**k, "weights_only": False})
        try:
            init_states = suite.get_task_init_states(ti)
        finally:
            torch.load = _orig_load
        if init_states is None or len(init_states) == 0:
            raise SystemExit(
                "no designated init states for task %d; refusing to fall back "
                "to env.reset(), which is an easier protocol" % ti)
        if ti == 0:
            print("  designated init states: %d for task 0 (official protocol)"
                  % len(init_states), flush=True)
        obj_adr = []
        if _state_objects:
            _m = env.env.sim.model
            obj_adr = sorted(int(_m.jnt_qposadr[j]) for j in range(_m.njnt)
                             if _m.jnt_type[j] == 0)
            if OBJ_PAD[0]:
                _kmax = _obj_dims_ck // 8
                if len(obj_adr) > _kmax:
                    raise SystemExit(
                        "task %d: live scene has %d free objects but the "
                        "checkpoint pads to %d slots; refusing to drop one"
                        % (ti, len(obj_adr), _kmax))
            elif 7 * len(obj_adr) != _obj_dims_ck:
                raise SystemExit(
                    "task %d: live scene has %d free objects (%d dims) but the "
                    "checkpoint's object block is %d dims; refusing"
                    % (ti, len(obj_adr), 7 * len(obj_adr), _obj_dims_ck))
            if ti == 0:
                print("  object qpos addresses (live sim): %s" % obj_adr,
                      flush=True)
        flags = []
        _TRAJ = globals().setdefault("_TRAJ_STORE", [])
        for r in range(args.rollouts_per_task):
            _tr = []
            env.seed(args.start_seed + ti * 1000 + r)
            env.reset()
            obs = env.set_init_state(init_states[r % len(init_states)])
            for _ in range(5):  # let the scene settle before acting
                obs, _, _, _ = env.step(
                    np.zeros(ck["action_dim"], dtype=np.float32))
            hist = deque(maxlen=oh)
            prev = np.zeros(ck["action_dim"], dtype=np.float32)
            done_ok, steps = False, 0

            def frame(o):
                cams = [o["agentview_image"], o["robot0_eye_in_hand_image"]]
                return np.concatenate(
                    [c.astype(np.float32) / 255.0 - 0.5 for c in cams], axis=-1)

            def proprio(o):
                # The env exposes robosuite names; the model was trained on the
                # dataset's layout. ee_states is [pos, axis-angle], established
                # by setting the env to a recorded state and matching the
                # conversions numerically -- axis-angle 7.4e-03, euler 8.9e-02.
                import robosuite.utils.transform_utils as _T
                need = {"robot0_eef_pos": 3, "robot0_eef_quat": 4,
                        "robot0_gripper_qpos": 2, "robot0_joint_pos": 7}
                for k, w in need.items():
                    if k not in o:
                        raise KeyError(
                            "env observation lacks %r; cannot build the "
                            "proprio vector the model was trained on" % k)
                    if np.asarray(o[k]).shape[-1] != w:
                        raise ValueError(
                            "env %r has width %d, dataset expects %d"
                            % (k, np.asarray(o[k]).shape[-1], w))
                ee = np.concatenate([
                    np.asarray(o["robot0_eef_pos"], dtype=np.float32),
                    _T.quat2axisangle(np.asarray(o["robot0_eef_quat"])).astype(np.float32),
                ])
                v = np.concatenate([
                    ee,                                                   # ee_states
                    np.asarray(o["robot0_gripper_qpos"], dtype=np.float32),
                    np.asarray(o["robot0_joint_pos"], dtype=np.float32),
                ])
                if v.shape[-1] != 15:
                    raise ValueError("proprio is %d dims, model expects 15"
                                     % v.shape[-1])
                return v

            def curpose(o):
                import robosuite.utils.transform_utils as _T
                return np.concatenate([
                    np.asarray(o["robot0_eef_pos"], dtype=np.float64),
                    _T.quat2axisangle(
                        np.asarray(o["robot0_eef_quat"])).astype(np.float64)])

            def stateobs(o):
                v = proprio(o)
                if obj_adr:
                    q = env.env.sim.data.qpos
                    poses = [np.asarray(q[adr:adr + 7], dtype=np.float32)
                             for adr in obj_adr]
                    if OBJ_PAD[0]:
                        _kmax = _obj_dims_ck // 8
                        blk = np.zeros(7 * _kmax + _kmax, dtype=np.float32)
                        if poses:
                            blk[:7 * len(poses)] = np.concatenate(poses)
                        blk[7 * _kmax:7 * _kmax + len(poses)] = 1.0
                        v = np.concatenate([v, blk])
                    else:
                        v = np.concatenate([v] + poses)
                if FLAT_LANG[0] is not None:
                    if task_lang is None:
                        raise SystemExit(
                            "checkpoint expects a %d-dim instruction in the "
                            "flat observation but no embedding was resolved "
                            "for this task" % FLAT_LANG[0])
                    v = np.concatenate(
                        [v, task_lang[0].cpu().numpy().astype(np.float32)])
                if STATE_OBS[0] and len(v) != int(ck["obs_dim"] or 0):
                    raise SystemExit(
                        "flat observation is %d dims, checkpoint expects %d"
                        % (len(v), int(ck["obs_dim"])))
                return v

            for _ in range(oh):
                hist.append((frame(obs), stateobs(obs)))

            while steps < args.max_steps and not done_ok:
                imgs = np.stack([h[0] for h in hist])
                prop = np.stack([h[1] for h in hist])
                if IMIT["state"]:
                    # the flow lives in pose space, so it must start from the
                    # pose the arm is actually at; zeros would mean the middle
                    # of the workspace
                    prev = _norm_pose(curpose(obs))
                chunk, prev = predict_chunk(model, imgs, prop, prev,
                                            args.action_horizon, ph, device,
                                            lang=task_lang)
                for a in chunk:
                    if IMIT["state"]:
                        a = _decode_pose(a, curpose(obs))
                    obs, _rew, done, _info = env.step(np.clip(a, -1.0, 1.0))
                    if args.dump_traj:
                        _tr.append(np.concatenate([
                            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                            np.asarray(obs["robot0_eef_quat"], dtype=np.float32)]))
                    hist.append((frame(obs), stateobs(obs)))
                    steps += 1
                    if env.check_success():
                        done_ok = True
                    if done_ok or done or steps >= args.max_steps:
                        break
            flags.append(bool(done_ok))
            if args.dump_traj and _tr:
                _TRAJ.append((ti, r, bool(done_ok), np.stack(_tr)))
        env.close()
        if args.dump_traj:
            _st = globals().get("_TRAJ_STORE") or []
            np.savez_compressed(
                args.dump_traj,
                meta=np.array([[t, r, int(ok), len(a)] for t, r, ok, a in _st]),
                **{"t%d_r%d" % (t, r): a for t, r, ok, a in _st})
            print("  trajectories: %d rollouts -> %s"
                  % (len(_st), args.dump_traj), flush=True)
        rate = float(np.mean(flags))
        per_task[task.name] = rate
        all_flags.extend(flags)
        print("  %-58s %.2f" % (task.name[:58], rate), flush=True)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "benchmark": _bench,
        "rollouts_per_task": args.rollouts_per_task,
        "num_tasks": len(per_task),
        "num_rollouts": len(all_flags),
        "success_rate": float(np.mean(all_flags)) if all_flags else float("nan"),
        "per_task": per_task,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print("pooled success_rate=%.4f over %d rollouts -> %s"
          % (summary["success_rate"], len(all_flags), args.output))


if __name__ == "__main__":
    main()
