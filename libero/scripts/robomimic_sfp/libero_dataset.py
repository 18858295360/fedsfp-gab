"""LIBERO dataset for the federated SFP trainer.

LIBERO ships one robomimic-format HDF5 file per task. The LIBERO-10
replication uses a demo membership file to assign examples within each task.

The image-mode interface returns a batch of
{imgs, proprio, a, t, v}, with cameras stacked on the channel axis and the SFPS
stabilized velocity target computed on a piecewise-linear action path.

Two LIBERO specifics the Square path does not have:
  - images are uint8 [H, W, 3] per camera under obs/agentview_rgb and
    obs/eye_in_hand_rgb, not a pre-stacked float array
  - proprioception is split across ee_states, gripper_states and joint_states
    rather than a single vector
"""
import numpy as np
import torch
from torch.utils.data import Dataset

CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")

# set by the trainer before clients are built;
# True makes every dataset emit proprio alone as "obs"
STATE_ONLY = False

# diffusion-policy targets instead of flow targets; set by the trainer
DP_MODE = False
DP_STEPS = 100
PROPRIO = ("ee_states", "gripper_states", "joint_states")

# object poses from the recorded simulator states; set by the trainer.
# A states row is [time, qpos, qvel] -- hence the +1 in every slice below.
STATE_OBJECTS = False

# imitation target: "action" = the recorded OSC delta command (default),
# "state" = the next end-effector pose, normalised, with the gripper command
# carried along as its seventh channel because it is a command, not a state.
IMIT_STATE = False
OBJECT_SLICES_JSON = ""


class LiberoSpatialDataset(Dataset):
    def __init__(self, hdf5_paths, obs_horizon=2, pred_horizon=16,
                 sigma0=0.1, sigma1=0.01, stabilization_k=5.0,
                 img_size=84, demo_limit=None, client_of=None,
                 lang_npz=None, allow=None):
        import h5py
        self.paths = list(hdf5_paths)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.sigma0, self.sigma1, self.k = float(sigma0), float(sigma1), float(stabilization_k)
        self.img_size = int(img_size)
        self.client_of = client_of

        self.index = []          # (file_idx, demo_key, t0)
        self.episodes = []       # (file_idx, demo_key) -- what checkpoint
                                 # metadata records as the demonstrations seen
        self.train_indices = []
        self.files = [None] * len(self.paths)
        self._h5py = h5py
        self.action_dim = None
        self.proprio_dim = None

        for fi, p in enumerate(self.paths):
            with h5py.File(p, "r") as f:
                keys = list(f["data"].keys())
                if demo_limit:
                    keys = keys[:demo_limit]
                if allow is not None:
                    keys = [k for k in keys if (fi, k) in allow]
                for k in keys:
                    g = f["data"][k]
                    self.episodes.append((fi, k))
                    n = g["actions"].shape[0]
                    if self.action_dim is None:
                        self.action_dim = int(g["actions"].shape[1])
                        self.proprio_dim = sum(
                            int(g["obs"][c].shape[1]) for c in PROPRIO if c in g["obs"])
                    # every start whose full prediction window fits
                    for t0 in range(0, max(n - self.pred_horizon, 0) + 1):
                        self.index.append((fi, k, t0))
        # the instruction vectors, looked up by task name so the trainer's
        # sorted-filename client order and the evaluator's benchmark order
        # resolve to the same row without either knowing the other's indexing
        self.lang = None
        self.lang_dim = 0
        if lang_npz:
            import numpy as _np
            import os as _os
            z = _np.load(lang_npz, allow_pickle=True)
            table = {str(t): v for t, v in zip(z["tasks"], z["emb"])}
            rows = []
            for p in self.paths:
                key = _os.path.basename(p).replace("_demo.hdf5", "")
                if key not in table:
                    raise SystemExit(
                        "[libero] no instruction embedding for %r in %s; the "
                        "lookup is keyed by the hdf5 basename without _demo"
                        % (key, lang_npz))
                rows.append(table[key])
            self.lang = _np.stack(rows).astype("float32")
            self.lang_dim = int(self.lang.shape[1])
        self.anchor_broaden = 0.0   # the trainer overwrites this
        self.img_channels = 3 * len(CAMERAS)
        self.state_only = bool(STATE_ONLY)
        if self.state_only and self.proprio_dim:
            self.obs_dim = int(self.proprio_dim)
        # flat_lang: the instruction rides in the observation vector, because
        # SFPConditionalUnet1D has no lang_dim and conditions only through obs
        self.flat_lang = bool(self.state_only and self.lang is not None)
        if self.flat_lang:
            self.obs_dim = int(self.obs_dim) + int(self.lang_dim)
        self.state_objects = bool(STATE_OBJECTS) and self.state_only
        self.obj_adr = None
        if self.state_objects:
            import json as _json
            import os as _os
            if not OBJECT_SLICES_JSON or not _os.path.exists(OBJECT_SLICES_JSON):
                raise SystemExit(
                    "[libero] STATE_OBJECTS is on but the slice map %r does "
                    "not exist; run the generator first" % OBJECT_SLICES_JSON)
            with open(OBJECT_SLICES_JSON) as _fh:
                _sl = _json.load(_fh)
            _flat = {}
            for _suite in _sl.values():
                _flat.update(_suite)
            self.obj_adr, _counts = [], set()
            for _p in self.paths:
                _key = _os.path.basename(_p).replace("_demo.hdf5", "")
                if _key not in _flat:
                    raise SystemExit(
                        "[libero] no object-slice entry for task %r in %s"
                        % (_key, OBJECT_SLICES_JSON))
                _e = _flat[_key]
                if str(_e.get("check", "")) != "PASS":
                    raise SystemExit(
                        "[libero] slice map marks %r as %r; only PASS entries "
                        "may train -- a SKIP was never identity-validated"
                        % (_key, _e.get("check")))
                _adr = sorted(int(a) for _, a in _e["objects"])
                _counts.add(len(_adr))
                with h5py.File(_p, "r") as _f:
                    _k0 = list(_f["data"])[0]
                    _w = int(_f["data"][_k0]["states"].shape[1])
                if _w != 1 + int(_e["nq"]) + int(_e["nv"]):
                    raise SystemExit(
                        "[libero] %s: states width %d != 1+nq+nv=%d; the "
                        "[time,qpos,qvel] layout assumption fails here"
                        % (_key, _w, 1 + int(_e["nq"]) + int(_e["nv"])))
                self.obj_adr.append(_adr)
            # Uniform K keeps the original layout byte-for-byte, so every
            # arm already trained stays comparable. Only a suite whose tasks
            # genuinely differ (libero_10: 2..8) pays for padding, and it pays
            # with an explicit mask -- zero-padding alone would claim an absent
            # object sits at the origin, which is a pose the policy can
            # legitimately see.
            # suite-wide object count, never the counts of whichever files
            # this instance holds: the model-width probe builds a dataset over
            # ONE file, and deciding padding from that gave a model 50 dims
            # narrower than the data it was about to be trained on.
            _suite_key = _os.path.basename(_os.path.normpath(
                _os.path.dirname(self.paths[0])))
            _suite_tasks = _sl.get(_suite_key) or {}
            _suite_counts = {len(_v["objects"]) for _v in _suite_tasks.values()}
            if not _suite_counts:
                raise SystemExit(
                    "[libero] no slice-map entry for suite %r, so the padded "
                    "observation width cannot be fixed" % _suite_key)
            self.obj_pad = (len(_suite_counts) != 1)
            self.obj_k_max = max(_suite_counts)
            _counts = _suite_counts
            if self.obj_pad:
                print("[libero] object counts differ across tasks (%s); "
                      "padding to K_max=%d with a %d-dim presence mask"
                      % (sorted(_counts), self.obj_k_max, self.obj_k_max),
                      flush=True)
            # 7 pose dims per slot over obj_k_max slots, plus the mask when
            # padding is on -- the same two fields _obj_rows builds from, so a
            # declared width cannot drift from the produced one
            self.obj_dim = (8 if self.obj_pad else 7) * self.obj_k_max
            self.obs_dim = int(self.proprio_dim) + self.obj_dim
            if getattr(self, "flat_lang", False):
                self.obs_dim = int(self.obs_dim) + int(self.lang_dim)
        self.imit_state = bool(IMIT_STATE)
        self.pose_lo = self.pose_hi = None
        if self.imit_state:
            import os as _os
            import numpy as _np
            _mins, _maxs = [], []
            for _p in self.paths:
                with h5py.File(_p, "r") as _f:
                    for _k in list(_f["data"]):
                        _e = _np.asarray(_f["data"][_k]["obs"]["ee_states"][:, :6],
                                         dtype=_np.float64)
                        _mins.append(_e.min(0)); _maxs.append(_e.max(0))
            if not _mins:
                raise SystemExit("[libero] imitation-mode state: no ee_states")
            self.pose_lo = _np.min(_np.stack(_mins), 0).astype("float32")
            self.pose_hi = _np.max(_np.stack(_maxs), 0).astype("float32")
            # a degenerate dimension would divide by zero and produce inf
            _span = self.pose_hi - self.pose_lo
            if float(_span.min()) < 1e-6:
                raise SystemExit(
                    "[libero] pose dimension %d is constant (span %.2e); the "
                    "[-1,1] mapping would divide by zero"
                    % (int(_span.argmin()), float(_span.min())))
            import json as _json
            _icp = _os.environ.get("LIBERO_INVERSE_CONTROLLER_JSON", "")
            if not _os.path.exists(_icp):
                raise SystemExit(
                    "[libero] imitation-mode state needs the conditioned "
                    "inverse controller at %s" % _icp)
            with open(_icp) as _fh:
                _ic = _json.load(_fh)
            _suite = _os.path.basename(_os.path.normpath(
                _os.path.dirname(self.paths[0])))
            if _suite not in _ic["per_suite"]:
                raise SystemExit(
                    "[libero] no inverse model for suite %r; refusing to fall "
                    "back to another suite's map" % _suite)
            self.ic_W = _np.asarray(_ic["per_suite"][_suite]["W"],
                                    dtype=_np.float64)
            self.imit_target = "virtual_action_v3"
            print("[libero] imitation target v3: virtual actions through the "
                  "%s inverse model (held-out R2 %.3f)"
                  % (_suite, _ic["per_suite"][_suite]["heldout_r2"]),
                  flush=True)
        self.train_indices = list(range(len(self.episodes)))

    def _file(self, fi):
        if self.files[fi] is None:                     # opened lazily, per worker
            self.files[fi] = self._h5py.File(self.paths[fi], "r")
        return self.files[fi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, key, t0 = self.index[i]
        g = self._file(fi)["data"][key]
        obs = g["obs"]

        # observation history, clamped at the episode start
        hs = [max(t0 - (self.obs_horizon - 1 - j), 0) for j in range(self.obs_horizon)]
        if getattr(self, "state_only", False):
            imgs = np.zeros((len(hs), 1, 1, 1), dtype=np.float32)
        else:
            imgs = np.stack([
                np.concatenate([np.asarray(obs[c][h], dtype=np.float32) / 255.0 - 0.5
                                for c in CAMERAS if c in obs], axis=-1)
                for h in hs])                                        # [H, 84, 84, 6]
        prop = np.stack([
            np.concatenate([np.asarray(obs[c][h], dtype=np.float32)
                            for c in PROPRIO if c in obs])
            for h in hs])                                        # [H, P]
        if getattr(self, "state_objects", False):
            prop = np.concatenate([prop, _obj_rows(self, g, fi, hs)], axis=1)

        if getattr(self, "imit_state", False):
            # v2: virtual actions from the demo's own pose transitions -- the
            # motion that happened, expressed in the space the environment
            # consumes, so every SFP scale (sigma0, k, clamps, seeds) applies
            traj = _virtual_actions(self, g, t0)
        else:
            traj = np.asarray(g["actions"][t0:t0 + self.pred_horizon], dtype=np.float32)
        if traj.shape[0] < self.pred_horizon:                    # pad by repeating the last
            traj = np.concatenate(
                [traj, np.repeat(traj[-1:], self.pred_horizon - traj.shape[0], axis=0)])

        # SFPS stabilized target on the piecewise-linear path through traj
        t = np.float32(np.random.rand())
        u = t * (self.pred_horizon - 1)
        lo = min(int(np.floor(u)), self.pred_horizon - 2)
        alpha = np.float32(u - lo)
        xi = traj[lo] + alpha * (traj[lo + 1] - traj[lo])
        xi_dot = (self.pred_horizon - 1) * (traj[lo + 1] - traj[lo])
        sigma_t = self.sigma0 * np.exp(-self.k * t)
        # anchor broadening: v is computed from a below, so a wider draw
        # keeps the target correct and teaches restoration from starting
        # points this client never reaches
        _sc = sigma_t + float(getattr(self, "anchor_broaden", 0.0)) * self.sigma0
        # global-anchor broadening: with probability frac the draw is
        # centred on the round-start global's generated action at this
        # window's segment; the restoring target below still uses the
        # true xi, so the global decides WHERE to train and the expert
        # decides WHAT to output.
        _centre = xi
        _gt = getattr(self, "global_anchor_table", None)
        _gf = float(getattr(self, "global_anchor_frac", 0.0) or 0.0)
        _gm = getattr(self, "global_anchor_mask", None)
        if (_gt is not None and _gf > 0.0 and np.random.rand() < _gf
                and (_gm is None or _gm[i])):
            _centre = _gt[i, lo]
        a = _centre + _sc * np.random.randn(*xi.shape).astype(np.float32)
        _dev = a - xi
        _r = -self.k * _dev
        # A displaced anchor -- a global one, whose offset does not shrink with
        # sigma(t) -- otherwise produces a restoring velocity an order of
        # magnitude above the ordinary target. Saturate it, in units of what an
        # ordinary draw at this flow time would give, so the field stays one
        # consistent function and ordinary samples are untouched.
        _cap = float(getattr(self, "restore_cap", 0.0) or 0.0)
        if _cap > 0.0:
            _lim = _cap * self.k * _sc * np.sqrt(float(_dev.size))
            _nr = float(np.linalg.norm(_r))
            if _nr > _lim:
                _r = _r * (_lim / _nr)
        v = xi_dot + _r

        if getattr(self, "state_only", False):
            if getattr(self, "flat_lang", False):
                prop = np.concatenate(
                    [prop, np.repeat(self.lang[fi][None, :], prop.shape[0],
                                     axis=0)], axis=1).astype(np.float32)
            out = {"obs": torch.from_numpy(prop),
                   "a": torch.from_numpy(a.astype(np.float32)),
                   "t": torch.tensor(t, dtype=torch.float32),
                   "v": torch.from_numpy(v.astype(np.float32))}
            if self.lang is not None:
                out["lang"] = torch.from_numpy(self.lang[fi])
            return out
        if DP_MODE:
            # a clean action chunk, a diffusion step, and the noise to predict
            k = int(np.random.randint(DP_STEPS))
            ab = _dp_alpha_bar(k)
            eps = np.random.randn(*xi.shape).astype(np.float32)
            noisy = (np.sqrt(ab) * xi + np.sqrt(1.0 - ab) * eps).astype(np.float32)
            out = {"imgs": torch.from_numpy(imgs),
                   "proprio": torch.from_numpy(prop),
                   "a": torch.from_numpy(noisy),
                   "t": torch.tensor(k / float(DP_STEPS), dtype=torch.float32),
                   "v": torch.from_numpy(eps)}
            if self.lang is not None:
                out["lang"] = torch.from_numpy(self.lang[fi])
            return out
        out = {"imgs": torch.from_numpy(imgs),
                "proprio": torch.from_numpy(prop),
                "a": torch.from_numpy(a.astype(np.float32)),
                "t": torch.tensor(t, dtype=torch.float32),
                "v": torch.from_numpy(v.astype(np.float32))}
        if self.lang is not None:
            out["lang"] = torch.from_numpy(self.lang[fi])
        return out



def _dp_alpha_bar(k, steps=None):
    """Cumulative alpha for the squared-cosine schedule, the one Diffusion
    Policy uses. Computed in closed form so no scheduler object has to be
    carried into dataloader workers."""
    import math
    T = steps or DP_STEPS
    f = lambda x: math.cos((x / T + 0.008) / 1.008 * math.pi / 2) ** 2
    return max(min(f(k) / f(0), 0.9999), 1e-4)


def _aa_to_R(v):
    """axis-angle -> rotation matrices, batched over the leading axis."""
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.divide(v, th, out=np.zeros_like(v), where=th > 1e-9)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def _R_to_aa(R):
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2],
                  R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], axis=-1)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9) * th[..., None]


def _virtual_actions(ds, g, t0):
    """The v2 pose target for one window: each demo pose transition mapped
    through the fitted inverse controller, plus the recorded gripper command.

    Built here and ONLY here -- __getitem__ takes the whole window and the GAB
    table builder takes row zero, so the two can never drift apart. Rotation
    deltas are relative SO(3), never axis-angle subtraction.
    """
    ee = np.asarray(g["obs"]["ee_states"][:, :6], dtype=np.float64)
    gw = np.asarray(g["obs"]["gripper_states"], dtype=np.float64)
    n = len(ee)
    ci = [min(t0 + j, n - 1) for j in range(ds.pred_horizon)]
    ni = [min(t0 + j + 1, n - 1) for j in range(ds.pred_horizon)]
    cur, nxt = ee[ci], ee[ni]
    d = np.empty((ds.pred_horizon, 6))
    d[:, :3] = nxt[:, :3] - cur[:, :3]
    d[:, 3:] = _R_to_aa(_aa_to_R(nxt[:, 3:])
                        @ np.transpose(_aa_to_R(cur[:, 3:]), (0, 2, 1)))
    # the command depends on the configuration it was issued from, not on the
    # displacement alone -- that dependence is what the diagonal model dropped
    feat = np.concatenate([d, cur, gw[ci], np.ones((ds.pred_horizon, 1))],
                          axis=1)
    va = np.clip(feat @ ds.ic_W, -1.0, 1.0)
    gi = [min(t0 + j, g["actions"].shape[0] - 1)
          for j in range(ds.pred_horizon)]
    grip = np.asarray(g["actions"][:, 6:7], dtype=np.float32)[gi]
    return np.concatenate([va.astype(np.float32), grip], axis=1)


def _obj_rows(ds, g, fi, hs):
    """Object pose block for the given history frames, straight from the
    recorded simulator states. The +1 skips the leading time element; the
    offset was established by restoring a state and reading qpos back."""
    st = g["states"]
    # row h+1, not h: measured on all three suites, states[t] equals the state
    # obs[t-1] was rendered from EXACTLY (err 0.0; one step off is ~2e-2, ten
    # steps ~3e-1). The column +1 below is unrelated -- it skips the leading
    # time element of the [time, qpos, qvel] row layout.
    adrs = ds.obj_adr[fi]
    block = np.stack([
        np.concatenate([np.asarray(st[min(h + 1, len(st) - 1)][1 + adr:1 + adr + 7],
                                   dtype=np.float32)
                        for adr in adrs])
        for h in hs])
    if not getattr(ds, "obj_pad", False):
        return block
    # pad the pose slots, then state which of them are real
    kmax = ds.obj_k_max
    n = len(hs)
    out = np.zeros((n, 7 * kmax + kmax), dtype=np.float32)
    out[:, :7 * len(adrs)] = block
    out[:, 7 * kmax:7 * kmax + len(adrs)] = 1.0
    return out


def _window_for_table(ds, i):
    """Observation stack, proprio, language and the window's first action --
    everything the table generator needs, built exactly as __getitem__ builds
    its observation half so the two can never drift apart silently."""
    fi, key, t0 = ds.index[i]
    g = ds._file(fi)["data"][key]
    obs = g["obs"]
    hs = [max(t0 - (ds.obs_horizon - 1 - j), 0) for j in range(ds.obs_horizon)]
    imgs = np.stack([
        np.concatenate([np.asarray(obs[c][h], dtype=np.float32) / 255.0 - 0.5
                        for c in CAMERAS if c in obs], axis=-1)
        for h in hs])
    prop = np.stack([
        np.concatenate([np.asarray(obs[c][h], dtype=np.float32)
                        for c in PROPRIO if c in obs])
        for h in hs])
    if getattr(ds, "state_objects", False):
        prop = np.concatenate([prop, _obj_rows(ds, g, fi, hs)], axis=1)
    if getattr(ds, "flat_lang", False):
        prop = np.concatenate(
            [prop, np.repeat(ds.lang[fi][None, :], prop.shape[0], axis=0)],
            axis=1).astype(np.float32)
    if getattr(ds, "imit_state", False):
        a0 = _virtual_actions(ds, g, t0)[0]
    else:
        a0 = np.asarray(g["actions"][t0], dtype=np.float32)
    lang = ds.lang[fi] if ds.lang is not None else None
    return imgs, prop, lang, a0


def build_clients(root, obs_horizon=2, pred_horizon=16, demo_limit=None,
                  merge=False, **kw):
    """One client per spatial configuration -- LIBERO's own partition.

    Returned in sorted filename order so the client index is stable across runs
    and across machines; a shuffled assignment would make two arms incomparable
    for reasons nothing in the logs would reveal.
    """
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    if not paths:
        raise SystemExit("no LIBERO hdf5 files under %s" % root)
    if merge:
        # every file in one dataset: the centralized control, where the client
        # partition is removed but the data and the sampling are unchanged
        ds = LiberoSpatialDataset(paths, obs_horizon=obs_horizon,
                                  pred_horizon=pred_horizon,
                                  demo_limit=demo_limit, client_of=0, **kw)
        return [{"key": "pooled", "task": "all_%d_tasks" % len(paths),
                 "ds": ds, "weight": float(len(ds))}]

    clients = []
    for i, p in enumerate(paths):
        ds = LiberoSpatialDataset([p], obs_horizon=obs_horizon,
                                  pred_horizon=pred_horizon,
                                  demo_limit=demo_limit, client_of=i, **kw)
        clients.append({"key": "client_%d" % i,
                        "task": os.path.basename(p).replace("_demo.hdf5", ""),
                        "ds": ds, "weight": float(len(ds))})
    return clients


if __name__ == "__main__":
    import os
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/datasets/libero/libero_spatial")
    cs = build_clients(root, demo_limit=5)
    print("clients: %d" % len(cs))
    for c in cs[:3]:
        print("  %-10s %-58s samples=%d" % (c["key"], c["task"][:58], len(c["ds"])))
    b = cs[0]["ds"][0]
    print("\nbatch element shapes:")
    for k, v in b.items():
        print("  %-8s %s %s" % (k, tuple(v.shape), v.dtype))
    print("\naction_dim=%d proprio_dim=%d img_channels=%d"
          % (cs[0]["ds"].action_dim, cs[0]["ds"].proprio_dim, cs[0]["ds"].img_channels))


def _within_task_groups(paths, n_groups):
    """Sort each task's demos along its own leading initial-state component.

    Per task, not globally: a global component is dominated by which task a demo
    belongs to, and splitting on it reproduces the by-task partition.
    """
    import h5py
    import numpy as np
    out = [[] for _ in range(n_groups)]
    for fi, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            X = np.stack([np.asarray(f["data"][k].attrs["init_state"],
                                     dtype=np.float64).ravel() for k in keys])
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        order = np.argsort(Xc @ Vt[0])
        for gi, g in enumerate(np.array_split(order, n_groups)):
            for d in g:
                out[gi].append((fi, keys[d]))
    return [set(g) for g in out]


def build_clients_byinit(root, n_clients=3, obs_horizon=2, pred_horizon=16,
                         lang_npz=None, **kw):
    """One client per initial-state region, every client holding every task.

    The counterpart to build_clients, which gives each client one task. Here
    task composition is identical across clients by construction, so whatever
    heterogeneity remains is support shift and nothing else.
    """
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    if not paths:
        raise SystemExit("no LIBERO hdf5 files under %s" % root)
    groups = _within_task_groups(paths, n_clients)

    clients = []
    for i, allow in enumerate(groups):
        ds = LiberoSpatialDataset(paths, obs_horizon=obs_horizon,
                                  pred_horizon=pred_horizon, client_of=i,
                                  lang_npz=lang_npz, allow=allow, **kw)
        clients.append({"key": "region_%d" % i,
                        "task": "all_%d_tasks_region_%d" % (len(paths), i),
                        "ds": ds, "weight": float(len(ds))})
    return clients


def _mode_kmeans(X, k, seed=0):
    """k-means with farthest-point seeding, so a split is reproducible."""
    import numpy as np
    k = int(k)
    if k <= 1 or len(X) <= k:
        return np.zeros(len(X), dtype=int)
    rng = np.random.RandomState(seed)
    c = [X[rng.randint(len(X))]]
    while len(c) < k:
        d = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(X[int(np.argmax(d))])
    c = np.stack(c)
    lab = np.full(len(X), -1)
    for _ in range(30):
        new = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(2), 1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                c[j] = X[lab == j].mean(0)
    return lab


def _traj_feature(f, keys, n_grid=12):
    """Each demo's action path, resampled to a common length and flattened."""
    import numpy as np
    rows = []
    for k in keys:
        a = np.asarray(f["data"][k]["actions"], dtype=np.float64)
        idx = np.linspace(0, len(a) - 1, n_grid)
        rows.append(np.stack([np.interp(idx, np.arange(len(a)), a[:, d])
                              for d in range(a.shape[1])], axis=1).ravel())
    X = np.asarray(rows)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def _mode_groups(paths, k=2, feature="azimuth"):
    # bound to a name the body does not reuse: `k` below is a demo key
    n_clients = int(k)
    feature = str(feature)
    print("[libero] mode split: %d clients, feature=%s" % (n_clients, feature),
          flush=True)
    """Split each task's demos by the object-frame approach azimuth at grasp.

    Per task, because each scene has its own geometry, and in the object's
    frame, because the object's position is randomised and a world-frame
    heading would mix the two.
    """
    import h5py
    import numpy as np
    groups = [[] for _ in range(n_clients)]
    for fi, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            X = np.stack([np.asarray(f["data"][k].attrs["init_state"],
                                     dtype=np.float64).ravel() for k in keys])
            Xc = X - X.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            d = int(np.argmax(np.abs(Vt[0])))
            s = max(d - 1, 0)
            for cand in (d - 2, d - 1, d):
                if cand >= 0 and cand + 2 < X.shape[1] and \
                        np.abs(Vt[0][cand:cand + 3]).sum() > 0.8 * np.abs(Vt[0]).sum():
                    s = cand
                    break
            ang = []
            for i, k in enumerate(keys):
                g = f["data"][k]["obs"]["gripper_states"][:]
                ee = f["data"][k]["obs"]["ee_states"][:]
                st = np.abs(g[:, 0] - g[:, 1]) if g.shape[1] > 1 else g[:, 0]
                t = min(max(int(np.argmin(st)) if len(st) else len(ee) // 2, 1),
                        len(ee) - 1)
                v = ee[t, :2] - X[i, s:s + 2]
                ang.append(np.arctan2(v[1], v[0]))
        if feature == "traj":
            with h5py.File(p, "r") as f2:
                lab = _mode_kmeans(_traj_feature(f2, keys), n_clients, seed=fi)
        elif n_clients == 2:
            a = np.array(ang)
            v = np.stack([np.cos(a), np.sin(a)], 1)
            c = np.stack([v[np.argmax(v[:, 0])], v[np.argmin(v[:, 0])]])
            for _ in range(15):
                lab = np.argmax(v @ c.T, 1)
                for j in (0, 1):
                    if (lab == j).any():
                        m = v[lab == j].mean(0)
                        c[j] = m / (np.linalg.norm(m) + 1e-12)
        else:
            a = np.array(ang)
            lab = _mode_kmeans(np.stack([np.cos(a), np.sin(a)], 1), n_clients,
                               seed=fi)
        for i, k in enumerate(keys):
            groups[int(lab[i])].append((fi, k))
    return [set(g) for g in groups]


def build_clients_bymode(root, obs_horizon=2, pred_horizon=16, lang_npz=None,
                         **kw):
    """One client per behavioural mode, every task present in both.

    The counterpart to the task and region builders. Task and instruction are
    identical across clients, so nothing in the input announces which mode an
    episode belongs to -- a gate has to find it, which is the point.
    """
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    if not paths:
        raise SystemExit("no LIBERO hdf5 files under %s" % root)
    n_modes = int(kw.pop("n_modes", 2) or 2)
    feature = str(kw.pop("mode_feature", "azimuth") or "azimuth")
    if feature.startswith("nested") or feature.startswith("file:"):
        # The paper's K=2 by-mode split as the root, refined by the same
        # antipodal 2-means on the same per-task approach azimuth. Raising K
        # only subdivides -- every client at K sits inside one at K/2 -- so the
        # sweep varies client count and not where the boundary was drawn.
        # Re-clustering independently does not have that property: at K=2 the
        # shipped bespoke branch and the generic k-means branch agree on 292 of
        # 500 demonstrations, which is below chance for a 337/163 partition.
        import json as _json
        if not feature.startswith("file:"):
            raise ValueError("Provide the published partition with file:<path>.")
        _p = feature[5:]
        _d = _json.load(open(_p))
        groups = [set((int(x.split("|", 1)[0]), x.split("|", 1)[1]) for x in v)
                  for v in _d["clients"].values()]
        print("[libero] NESTED by-mode split loaded from %s, sizes %s"
              % (_p.rsplit("/", 1)[-1], [len(g) for g in groups]), flush=True)
    else:
        groups = _mode_groups(paths, k=n_modes, feature=feature)
    iid = int(kw.pop("mode_iid", 0) or 0)
    if iid:
        # IID control for the client-count sweep: identical client count and
        # identical client sizes, membership drawn at random instead of by
        # behaviour mode.  Holding the size profile fixed is what makes this
        # a control rather than a second intervention.
        pool = sorted(set().union(*groups)) if groups else []
        rng = np.random.RandomState(iid)
        rng.shuffle(pool)
        sizes = [len(g) for g in groups]
        out, at = [], 0
        for s in sizes:
            out.append(set(pool[at:at + s]))
            at += s
        groups = out
        print("[libero] BY-MODE IID CONTROL seed %d: membership shuffled, "
              "sizes held at %s" % (iid, sizes), flush=True)
    print("[libero] by-mode: %d clients, feature=%s, sizes %s"
          % (n_modes, feature, [len(g) for g in groups]), flush=True)
    clients = []
    for i, allow in enumerate(groups):
        ds = LiberoSpatialDataset(paths, obs_horizon=obs_horizon,
                                  pred_horizon=pred_horizon, client_of=i,
                                  lang_npz=lang_npz, allow=allow, **kw)
        clients.append({"key": "mode_%d" % i,
                        "task": "all_%d_tasks_mode_%d" % (len(paths), i),
                        "ds": ds, "weight": float(len(ds))})
    return clients
