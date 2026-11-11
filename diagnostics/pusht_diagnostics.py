"""D_traj(s) and D_cond(s) along the frozen assignment path, Push-T.

Derived from ptexact.py, which produced the paper tab:dcond Push-T row.
Three deliberate changes, each stated:

  1. dpath generalises to K>2 as the mean over client PAIRS of the phase-aligned
     mean-path distance, over the mean over clients of the within-client spread.
     At K=2 this is algebraically the published formula, so the K=2 numbers stay
     comparable with the published ones.
  2. The path preserves the pool and every client size, so a size-matched
     randomisation null is the SAME null at every level. It is therefore drawn
     ONCE per corpus and client count and shared by all five points, which makes
     the five points directly comparable and lets the draw count rise cheaply.
  3. NULL_DRAWS defaults to 60 rather than 12. A 12-draw denominator was shown
     to be unstable: on the same data a fresh draw moved D_cond z from +17.12 to
     +9.18 while the raw value moved 4 per cent. Raw D and the null band are the
     primary report; z is printed for continuity with the paper.

POSITIVE CONTROL: run with --control to reproduce the published s=1 K=2 numbers
(12 draws, seed 3) before any curve is trusted.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import random
import statistics
import numpy as np, zarr

TS = 15
L_PHASE = 11
NULL_DRAWS = 60
NULL_SEED = 11


def resample(P, L):
    dd = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(dd)])
    if s[-1] < 1e-9:
        return np.repeat(P[:1], L, axis=0)
    s = s / s[-1]
    q = np.linspace(0, 1, L)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(P.shape[1])], axis=1)


def read_arrays(root):
    return (np.asarray(root["data"]["state"]),
            np.asarray(root["data"]["action"]),
            np.asarray(root["meta"]["episode_ends"]))


def load_corpus(dataset, union):
    if dataset.is_dir():
        root = zarr.open_group(str(dataset), mode="r")
        state, act, ends = read_arrays(root)
    else:
        with zarr.ZipStore(str(dataset), mode="r") as store:
            state, act, ends = read_arrays(zarr.open_group(store=store, mode="r"))
    starts = np.concatenate([[0], ends[:-1]])
    O = {}
    A = {}
    X = {}
    for n, e in enumerate(union):
        a, b = starts[e], ends[e]
        idx = np.linspace(0, b - a - 1, min(TS, b - a)).astype(int)
        O[n] = state[a:b][idx].astype(np.float32)
        A[n] = act[a:b][idx].astype(np.float32)
        X[n] = resample(state[a:b, :4].astype(float), L_PHASE)
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
        dd = ((q[:, None, :] - Ot[None, :, :]) ** 2).sum(-1)
        dd = np.where(Dq[s:s + B][:, None] == Dt[None, :], np.inf, dd)
        j = np.argmin(dd, axis=1)
        m = np.isfinite(dd[np.arange(len(q)), j])
        out.append(np.linalg.norm(Aq[s:s + B][m] - At[j][m], axis=1))
    return np.concatenate(out)


def dcond(lab, O, A):
    cl = collections.defaultdict(list)
    for k, c in lab.items():
        cl[c].append(k)
    pack = {}
    for c, ks in cl.items():
        ks = sorted(ks)
        pack[c] = (np.concatenate([O[k] for k in ks], axis=0),
                   np.concatenate([A[k] for k in ks], axis=0),
                   np.concatenate([np.full(len(O[k]), k) for k in ks]))
    cs = sorted(pack)
    wn = []
    ac = []
    for i in cs:
        Oi, Ai, Di = pack[i]
        wn.append(gap(Oi, Ai, Di, Oi, Ai, Di).mean())
        for j in cs:
            if j == i:
                continue
            Oj, Aj, Dj = pack[j]
            ac.append(gap(Oi, Ai, Di, Oj, Aj, Dj).mean())
    num = statistics.mean(ac)
    den = statistics.mean(wn)
    return num / den - 1.0


def dpath(lab, X):
    cl = collections.defaultdict(list)
    for k, c in lab.items():
        cl[c].append(k)
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
    with path.open(encoding="utf-8") as handle:
        d = json.load(handle)
    d = d.get("clients", d)
    keys = sorted([k for k in d if k.startswith("client_")],
                  key=lambda k: int(k.split("_")[1]))
    union = d.get("union_200")
    if union is None:
        acc = set()
        for k in keys:
            acc = acc.union(set(d[k]))
        union = sorted(acc)
    union = sorted(set(union))
    pos = {}
    for n, e in enumerate(union):
        pos[e] = n
    lab = {}
    for ci, k in enumerate(keys):
        for e in d[k]:
            if e in pos:
                lab[pos[e]] = ci
    sizes = []
    for k in keys:
        sizes.append(len(d[k]))
    return union, lab, sizes


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("splits", type=Path, nargs="*")
    args = parser.parse_args()
    if args.control and args.splits:
        parser.error("--control uses the included K=2 split")
    if args.control:
        split = Path(__file__).with_name("splits") / "pusht" / "rotation_k2.json"
        union, lab, sizes = read_split(split)
        O, A, X = load_corpus(args.dataset, union)
        ids = sorted(O)
        vc, vp = null_set(sizes, ids, O, A, X, 12, 3)
        dc = dcond(lab, O, A)
        dp = dpath(lab, X)
        mc = statistics.mean(vc)
        sc = statistics.stdev(vc)
        mp = statistics.mean(vp)
        sp = statistics.stdev(vp)
        print("POSITIVE CONTROL. Published: D_cond 0.2668 -> +9.18 sd, D_traj 0.3176 -> +4.43 sd")
        print("  sizes %s" % sizes)
        print("  D_cond %.4f  null %.4f+-%.4f  -> %+.2f sd" % (dc, mc, sc, (dc - mc) / sc))
        print("  D_traj %.4f  null %.4f+-%.4f  -> %+.2f sd" % (dp, mp, sp, (dp - mp) / sp))
        ok = abs(dc - 0.2668) < 0.0005 and abs(dp - 0.3176) < 0.0005
        print("  RAW VALUES REPRODUCE: %s" % ok)
        if not ok:
            raise SystemExit(1)
        return
    if not args.splits:
        parser.error("provide one or more split files, or use --control")
    files = args.splits
    levels = []
    for f in files:
        tag = f.stem
        union, lab, sizes = read_split(f)
        levels.append((tag, union, lab, sizes))
    base_union = levels[0][1]
    base_sizes = levels[0][3]
    for tag, u, lab, sizes in levels:
        if u != base_union:
            print("ABORT: %s has a different union, the null cannot be shared" % tag)
            return
        if sizes != base_sizes:
            print("ABORT: %s sizes %s against %s, the null cannot be shared"
                  % (tag, sizes, base_sizes))
            return
    print("pool %d episodes, client sizes %s, shared across all %d levels"
          % (len(base_union), base_sizes, len(levels)))
    O, A, X = load_corpus(args.dataset, base_union)
    ids = sorted(O)
    vc, vp = null_set(base_sizes, ids, O, A, X, NULL_DRAWS, NULL_SEED)
    mc = statistics.mean(vc)
    sc = statistics.stdev(vc)
    mp = statistics.mean(vp)
    sp = statistics.stdev(vp)
    print("null over %d size-matched draws (shared): D_cond %.4f+-%.4f   D_traj %.4f+-%.4f"
          % (NULL_DRAWS, mc, sc, mp, sp))
    print("")
    print("  %-28s %-30s %s" % ("level", "D_cond (z, exceed)", "D_traj (z, exceed)"))
    for tag, u, lab, sizes in levels:
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
        print("  %-28s %.4f (%+.2f sd, %2d of %d)     %.4f (%+.2f sd, %2d of %d)"
              % (tag, dc, (dc - mc) / sc, ec, NULL_DRAWS,
                 dp, (dp - mp) / sp, ep, NULL_DRAWS))


if __name__ == "__main__":
    main()
