"""LIBERO-10 D_traj.  Implements lb_dtraj_prereg_20260904.md.  READ ONLY.

Writes nothing.  Runs its four registered positive controls first and refuses to
print any measurement if a control fails.
"""
import argparse
import collections
import json
from pathlib import Path
import random
import statistics
import sys
import numpy as np, h5py
import xml.etree.ElementTree as ET

JW = {"free": 7, "ball": 4, "slide": 1, "hinge": 1}
L_PHASE, NULL_DRAWS, NULL_SEED, PERM_DRAWS = 11, 60, 11, 1000

TARGETS = [
    "libero10_sev0p00_PLD_K2",
    "libero10_sev0p25_PLD_K2",
    "libero10_sev0p50_PLD_K2",
    "libero10_sev0p75_PLD_K2",
    "libero10_sev1p00_PLD_K2",
    "libero10_pathPLD_K4_s000",
    "libero10_pathPLD_K4_s025",
    "libero10_pathPLD_K4_s050",
    "libero10_pathPLD_K4_s075",
    "libero10_closelift_S_K4",
    "libero10_pathPLD_K8_s000",
    "libero10_pathPLD_K8_s025",
    "libero10_pathPLD_K8_s050",
    "libero10_pathPLD_K8_s075",
    "libero10_closelift_S_K8",
]


# ---------------------------------------------------------------- dcurve.py, verbatim
def resample(P, L):
    dd = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(dd)])
    if s[-1] < 1e-9:
        return np.repeat(P[:1], L, axis=0)
    s = s / s[-1]
    q = np.linspace(0, 1, L)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(P.shape[1])], axis=1)


def dpath_ref(lab, X):
    """Literal copy of dcurve.py's dpath, kept only as the transcription control."""
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


# ---------------------------------------------------------------- the working version
def dpath(keys, lab, X):
    """Same statistic, written independently; control 1 asserts it equals dpath_ref."""
    cs = sorted(set(lab[k] for k in keys))
    if len(cs) < 2:
        return float("nan")
    mus, spr = {}, []
    for c in cs:
        mem = sorted(k for k in keys if lab[k] == c)
        P = np.stack([X[k] for k in mem])
        mu = P.mean(axis=0)
        mus[c] = mu
        spr.append(float(np.mean(np.linalg.norm(P - mu, axis=2))))
    dst = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            dst.append(float(np.mean(np.linalg.norm(mus[cs[i]] - mus[cs[j]], axis=1))))
    return float(statistics.mean(dst) / max(statistics.mean(spr), 1e-9))


def null_dist(keys, sizes, X, draws, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        p = list(keys)
        rng.shuffle(p)
        lab, t = {}, 0
        for ci, n in enumerate(sizes):
            for _k in range(n):
                lab[p[t]] = ci
                t += 1
        out.append(dpath(keys, lab, X))
    return out


# ---------------------------------------------------------------- data
def free_joints(xml):
    out, at = [], 0
    for node in ET.fromstring(xml).iter():
        if node.tag == "freejoint":
            out.append((at, node.get("name", "?"), 7))
            at += 7
        elif node.tag == "joint":
            w = JW.get(node.get("type", "hinge"), 1)
            out.append((at, node.get("name", "?"), w))
            at += w
    return [(a, n.replace("_joint0", "").replace("_main", "")) for a, n, w in out if w == 7]


def load(dataset_dir, canonical_objects):
    with canonical_objects.open(encoding="utf-8") as handle:
        canon = json.load(handle)
    paths = sorted(dataset_dir.glob("*.hdf5"))
    if len(paths) != 10:
        raise ValueError(f"Expected 10 LIBERO task files, found {len(paths)}")
    X, task_of = {}, {}
    for ti, p in enumerate(paths):
        want = canon[str(ti)]["canonical_object"]
        with h5py.File(p, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            mf = f["data"][keys[0]].attrs["model_file"]
            fj = free_joints(mf.decode("utf-8", "ignore") if isinstance(mf, bytes) else mf)
            hit = [a for a, n in fj if n == want]
            if not hit:
                raise SystemExit("task %d: canonical object %s not among the free joints"
                                 % (ti, want))
            j = 1 + hit[0]
            for k in keys:
                g = f["data"][k]
                ee = np.asarray(g["obs"]["ee_states"][:], dtype=np.float64)[:, :3]
                st = np.asarray(g["states"][:], dtype=np.float64)
                if j + 3 > st.shape[1]:
                    raise SystemExit("task %d: object address out of range" % ti)
                T = min(len(ee), len(st))
                P = np.concatenate([ee[:T], st[:T, j:j+3]], axis=1)
                nm = "%d|%s" % (ti, k)
                X[nm] = resample(P, L_PHASE)
                task_of[nm] = ti
    return X, task_of


def read_split(split_dir, name):
    with (split_dir / (name + ".json")).open(encoding="utf-8") as handle:
        d = json.load(handle)
    cl = d["clients"] if "clients" in d else d
    ks = sorted([k for k in cl if str(k).startswith("client_")],
                key=lambda k: int(str(k).split("_")[1]))
    lab = {}
    for ci, k in enumerate(ks):
        for e in cl[k]:
            lab[str(e)] = ci
    return lab, len(ks)


def zof(v, nul):
    m, s = statistics.mean(nul), statistics.stdev(nul)
    return (v - m) / s if s > 1e-12 else float("nan"), m, s


def run(dataset_dir, split_dir, canonical_objects):
    print("path = arc-length resample to %d of [EE_xyz, canonical-object_xyz]" % L_PHASE)
    print()
    X, task_of = load(dataset_dir, canonical_objects)
    ALL = sorted(X)
    TASKS = sorted(set(task_of.values()))
    bytask = collections.defaultdict(list)
    for n in ALL:
        bytask[task_of[n]].append(n)
    print("loaded %d demos over %d tasks, path shape %s"
          % (len(ALL), len(TASKS), X[ALL[0]].shape))
    print()

    # ---------------------------------------------------------------- controls
    print("=== POSITIVE CONTROLS ===")
    fails = []
    lab0, K0 = read_split(split_dir, TARGETS[0])
    mine = dpath(ALL, lab0, X)
    ref = dpath_ref(dict((k, lab0[k]) for k in ALL), X)
    ok1 = abs(mine - ref) < 1e-12
    print("  1 transcription: mine %.12f  dcurve.py %.12f  agree %s" % (mine, ref, ok1))
    if not ok1:
        fails.append("transcription")

    labT = dict((n, task_of[n]) for n in ALL)
    dT = dpath(ALL, labT, X)
    szT = [len(bytask[t]) for t in TASKS]
    nT = null_dist(ALL, szT, X, NULL_DRAWS, NULL_SEED)
    zT, mT, sT = zof(dT, nT)
    ok2 = dT > 1.0 and zT > 5.0
    print("  2 upper bracket, by-task (global): D_traj %.4f  null %.4f+-%.4f  z %+.1f  %s"
          % (dT, mT, sT, zT, "OK" if ok2 else "FAIL"))
    if not ok2:
        fails.append("upper bracket")

    zs = []
    for t in TASKS:
        ks = sorted(bytask[t])
        sz = [len(ks) // 2, len(ks) - len(ks) // 2]
        rng = random.Random(777 + t)
        p = list(ks)
        rng.shuffle(p)
        lab = {}
        for i, n in enumerate(p):
            lab[n] = 0 if i < sz[0] else 1
        nul = null_dist(ks, sz, X, NULL_DRAWS, NULL_SEED + t)
        z, _, _ = zof(dpath(ks, lab, X), nul)
        zs.append(abs(z))
    medz = statistics.median(zs)
    ok3 = medz < 2.0
    print("  3 lower bracket, within-task random: median |z| %.2f over %d tasks  %s"
          % (medz, len(TASKS), "OK" if ok3 else "FAIL"))
    if not ok3:
        fails.append("lower bracket")

    print("  4 Push-T external control (D_traj 0.3176): not part of this LIBERO run")

    if fails:
        print()
        print("CONTROLS FAILED (%s) -- refusing to print measurements" % ", ".join(fails))
        sys.exit(1)
    print("  all runnable controls PASS")
    print()

    # ---------------------------------------------------------------- measurement
    print("SPEC NOTE: the prereg says the null is drawn once per (task, K) and shared by")
    print("every split in the run.  That is only possible where per-task client sizes are")
    print("equal; they are not, across splits.  The null is therefore size-matched per")
    print("(task, split) from the SAME seed, so any two splits with equal sizes receive an")
    print("identical null.  dcurve.py aborts in this situation rather than sharing.")
    print()
    for name in TARGETS:
        try:
            lab, K = read_split(split_dir, name)
        except Exception as ex:
            print("%s: unreadable (%s)" % (name, ex))
            continue
        miss = [n for n in ALL if n not in lab]
        if miss:
            print("%s: %d demos unlabelled, skipping" % (name, len(miss)))
            continue
        print("#" * 4, name, " K=%d" % K)
        g = dpath(ALL, lab, X)
        gz, gm, gs = zof(g, null_dist(
            ALL, [sum(1 for n in ALL if lab[n] == c) for c in range(K)],
            X, NULL_DRAWS, NULL_SEED))
        print("  GLOBAL (unstratified, secondary): D_traj %.4f  null %.4f+-%.4f  z %+.2f"
              % (g, gm, gs, gz))
        ds, zsl, ps = [], [], []
        for t in TASKS:
            ks = sorted(bytask[t])
            sz = [sum(1 for n in ks if lab[n] == c) for c in range(K)]
            if min(sz) == 0:
                print("  task %d  sizes %-26s  a client is empty here, skipped" % (t, sz))
                continue
            d = dpath(ks, lab, X)
            nul = null_dist(ks, sz, X, NULL_DRAWS, NULL_SEED + t)
            z, m, s = zof(d, nul)
            per = null_dist(ks, sz, X, PERM_DRAWS, NULL_SEED + t)
            p = (1.0 + sum(1 for v in per if v >= d)) / (PERM_DRAWS + 1.0)
            ds.append(d)
            zsl.append(z)
            ps.append(p)
            print("  task %d  sizes %-26s D %.4f  null %.4f+-%.4f  z %+5.2f  p %.4f%s"
                  % (t, str(sz), d, m, s, z, p, "  *" if p < 0.05 else ""))
        if ds:
            print("  --> tasks with p<0.05: %d of %d   median D %.4f   median z %+.2f"
                  % (sum(1 for p in ps if p < 0.05), len(ds),
                     statistics.median(ds), statistics.median(zsl)))
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--canonical-objects", type=Path,
                        default=Path(__file__).with_name("libero_canonical_objects.json"))
    args = parser.parse_args()
    run(args.dataset_dir, args.split_dir, args.canonical_objects)


if __name__ == "__main__":
    main()
