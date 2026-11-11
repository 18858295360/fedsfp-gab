"""Trajectory and matched-action diagnostics for RoboMimic Square, Can and Lift.

The five levels of a ladder use one shared 60-draw reference distribution
that preserves the demonstration pool and client sizes. The reported
exceedance counts describe this reference rather than formal p-values.
"""

import argparse
import collections
import json
import os
from pathlib import Path
import random
import statistics
import sys

import h5py
import numpy as np

TSUB = 15
L_PHASE = 11
NULL_DRAWS = int(os.environ.get("RM_NULL_DRAWS", "60"))
NULL_SEED = 11


def qmul(a, b):
    """Hamilton product, robosuite's (x, y, z, w) order."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], axis=1)


def qconj(q):
    return np.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], axis=1)


def resample(P, L):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-9:
        return np.repeat(P[:1], L, axis=0)
    s = s / s[-1]
    q = np.linspace(0, 1, L)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(P.shape[1])], axis=1)


def feats(path, wanted, lift=False):
    """lift=True remaps the object columns; every other line is dcurve_rm.py's."""
    f = h5py.File(path, "r")
    O = {}
    A = {}
    X = {}
    for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
        i = int(k.split("_")[1])
        if i not in wanted:
            continue
        g = f["data"][k]
        ee = np.asarray(g["obs"]["robot0_eef_pos"][:], dtype=np.float32)
        eq = np.asarray(g["obs"]["robot0_eef_quat"][:], dtype=np.float32)
        gq = np.asarray(g["obs"]["robot0_gripper_qpos"][:], dtype=np.float32)
        if lift:
            rp = np.asarray(g["obs"]["object"][:, 7:10], dtype=np.float32)
            rq = qmul(qconj(np.asarray(g["obs"]["robot0_eef_quat"][:], dtype=np.float64)),
                      np.asarray(g["obs"]["object"][:, 3:7], dtype=np.float64))
            ob = np.concatenate([rp, rq.astype(np.float32)], axis=1)
        else:
            ob = np.asarray(g["obs"]["object"][:, :7], dtype=np.float32)
        a = np.asarray(g["actions"][:], dtype=np.float32)
        T = len(a)
        idx = np.linspace(0, T - 1, min(TSUB, T)).astype(int)
        O[i] = np.concatenate([ee, eq, gq, ob], axis=1)[idx]
        A[i] = a[idx]
        obp = np.asarray(g["obs"]["object"][:, 0:3] if lift
                         else g["obs"]["object"][:, 7:10], dtype=float)
        P = np.concatenate([np.asarray(g["obs"]["robot0_eef_pos"][:], dtype=float), obp], axis=1)
        X[i] = resample(P, L_PHASE)
    f.close()
    M = np.concatenate([O[i] for i in sorted(O)], axis=0)
    mu = M.mean(0)
    sd = M.std(0)
    sd[sd < 1e-8] = 1.0
    for i in O:
        O[i] = (O[i] - mu) / sd
    return O, A, X


def gap(Oq, Aq, Dq, Ot, At, Dt):
    out = []
    B = 512
    for s in range(0, len(Oq), B):
        q = Oq[s:s + B]
        d = ((q[:, None, :] - Ot[None, :, :]) ** 2).sum(-1)
        d = np.where(Dq[s:s + B][:, None] == Dt[None, :], np.inf, d)
        j = np.argmin(d, axis=1)
        m = np.isfinite(d[np.arange(len(q)), j])
        out.append(np.linalg.norm(Aq[s:s + B][m] - At[j][m], axis=1))
    return np.concatenate(out)


def dcond(lab, O, A):
    cl = collections.defaultdict(list)
    for i, c in lab.items():
        cl[c].append(i)
    pack = {}
    for c, ii in cl.items():
        ii = sorted(ii)
        pack[c] = (np.concatenate([O[i] for i in ii], axis=0),
                   np.concatenate([A[i] for i in ii], axis=0),
                   np.concatenate([np.full(len(O[i]), i) for i in ii]))
    ks = sorted(pack)
    wn = []
    ac = []
    for i in ks:
        Oi, Ai, Di = pack[i]
        wn.append(gap(Oi, Ai, Di, Oi, Ai, Di).mean())
        for j in ks:
            if j == i:
                continue
            Oj, Aj, Dj = pack[j]
            ac.append(gap(Oi, Ai, Di, Oj, Aj, Dj).mean())
    num = statistics.mean(ac)
    den = statistics.mean(wn)
    return num / den - 1.0


def dpath(lab, X):
    cl = collections.defaultdict(list)
    for i, c in lab.items():
        cl[c].append(i)
    cs = sorted(cl)
    mus = {}
    spreads = []
    for c in cs:
        P = np.stack([X[i] for i in sorted(cl[c])])
        m = P.mean(0)
        mus[c] = m
        spreads.append(np.linalg.norm(P - m, axis=2).mean(0).mean())
    dists = []
    for a in range(len(cs)):
        for b in range(a + 1, len(cs)):
            dists.append(np.linalg.norm(mus[cs[a]] - mus[cs[b]], axis=1).mean())
    num = statistics.mean(dists)
    den = statistics.mean(spreads)
    return float(num / max(den, 1e-9))


def read_split(path):
    d = json.load(open(path))
    if "clients" in d and isinstance(d["clients"], dict):
        d = d["clients"]
    keys = sorted([k for k in d if str(k).startswith("client_")],
                  key=lambda k: int(str(k).split("_")[1]))
    lab = {}
    sizes = []
    for ci, k in enumerate(keys):
        mem = [int(x) for x in d[k]]
        sizes.append(len(mem))
        for e in mem:
            lab[e] = ci
    return lab, sizes


def null_set(sizes, ids, O, A, X, draws, seed):
    rng = random.Random(seed)
    vc = []
    vp = []
    for _ in range(draws):
        p = list(ids)
        rng.shuffle(p)
        lab = {}
        t = 0
        for ci, n in enumerate(sizes):
            for _k in range(n):
                lab[p[t]] = ci
                t += 1
        vc.append(dcond(lab, O, A))
        vp.append(dpath(lab, X))
    return vc, vp


def ladder(name, h5, files, lift=False):
    levels = []
    for tag, f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        lab, sizes = read_split(f)
        levels.append((tag, lab, sizes))
    base_sizes = levels[0][2]
    base_ids = sorted(levels[0][1])
    for tag, lab, sizes in levels:
        if sorted(lab) != base_ids:
            raise ValueError(f"{tag}: split changes the demonstration pool")
        if sizes != base_sizes:
            raise ValueError(f"{tag}: client sizes {sizes} differ from {base_sizes}")
    print("=== %s: pool %d, client sizes %s, %d levels, one shared %d-draw null ==="
          % (name, len(base_ids), base_sizes, len(levels), NULL_DRAWS))
    O, A, X = feats(h5, set(base_ids), lift=lift)
    vc, vp = null_set(base_sizes, base_ids, O, A, X, NULL_DRAWS, NULL_SEED)
    mc = statistics.mean(vc)
    sc = statistics.stdev(vc)
    mp = statistics.mean(vp)
    sp = statistics.stdev(vp)
    print("  null: D_cond %+0.4f+-%.4f   D_traj %.4f+-%.4f" % (mc, sc, mp, sp))
    for tag, lab, sizes in levels:
        dc = dcond(lab, O, A)
        dp = dpath(lab, X)
        ec = 0
        for x in vc:
            if x > dc:
                ec += 1
        ep = 0
        for x in vp:
            if x > dp:
                ep += 1
        print("  s=%-5s D_cond %+0.4f (%+.2f sd, %2d of %d)   D_traj %.4f (%+.2f sd, %2d of %d)"
              % (tag, dc, (dc - mc) / sc, ec, NULL_DRAWS,
                 dp, (dp - mp) / sp, ep, NULL_DRAWS))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=("square", "can", "lift"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--clients", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--split-dir", type=Path, default=Path(__file__).with_name("splits"))
    args = parser.parse_args()

    levels = [("0", "000"), ("0.25", "025"), ("0.5", "050"), ("0.75", "075"), ("1", "100")]
    paths = [(level, args.split_dir / args.task / f"k{args.clients}_s{tag}.json")
             for level, tag in levels]
    ladder(f"{args.task} K={args.clients}", args.dataset, paths, lift=args.task == "lift")


if __name__ == "__main__":
    main()
