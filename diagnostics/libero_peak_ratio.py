#!/usr/bin/env python
"""Final peak-S export: statistic, its own reference, validity accounting, figure coordinates.

One pass that produces everything the main-figure change needs, so no further recompute is
required. It can optionally check a historical reference log and also records validity.

THE DISPLAYED STATISTIC
    S_h = mean(d_cross) / mean(d_same)      per phase bin h
    T   = max_h S_h                          the peak, which is what the main figure shows
Every permutation is re-matched, re-scanned and reduced by the SAME maximum, so the reference
belongs to T and not to any other functional. The max-z band is never reused.

VALIDITY, on the main statistic's OWN criterion. The registered main diagnostic applies no
minimum-usable threshold; a bin is simply computable or not, and `S_of` returns nan when it is
not. So this reports:
    obs_bad      observed bins that are not computable, out of 5
    perm_bad     permutation rows with at least one non-computable bin, out of 1000
    reportable   observed value reportable / permutation test reportable, judged separately
The MIN_USABLE=20 rule belongs to the supplementary re-matching diagnostic and is NOT imported
here; per-bin usable counts are reported as descriptive support instead.

S = 1 means the two mean distances are equal. It is NOT a significance threshold for T, and is
recorded that way in the output so the figure caption cannot drift.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

import h5py
import numpy as np

from action_chunks import chunk_actions
from dbeh_fast import build, S_of

H_CHUNK, M, K_NN, PERMS, SEED = 16, 20, 6, 1000, 20260816
CELLS = ["libero10_sev0p00_PLD_K2", "libero10_sev0p25_PLD_K2", "libero10_sev0p50_PLD_K2",
         "libero10_sev0p75_PLD_K2", "libero10_sev1p00_PLD_K2",
         "libero10_pathPLD_K4_s000", "libero10_pathPLD_K4_s025", "libero10_pathPLD_K4_s050",
         "libero10_pathPLD_K4_s075", "libero10_closelift_S_K4",
         "libero10_pathPLD_K8_s000", "libero10_pathPLD_K8_s025", "libero10_pathPLD_K8_s050",
         "libero10_pathPLD_K8_s075", "libero10_closelift_S_K8"]
SVAL = {"libero10_sev0p00_PLD_K2": ("K2", 0.0), "libero10_sev0p25_PLD_K2": ("K2", 0.25),
        "libero10_sev0p50_PLD_K2": ("K2", 0.5), "libero10_sev0p75_PLD_K2": ("K2", 0.75),
        "libero10_sev1p00_PLD_K2": ("K2", 1.0),
        "libero10_pathPLD_K4_s000": ("K4", 0.0), "libero10_pathPLD_K4_s025": ("K4", 0.25),
        "libero10_pathPLD_K4_s050": ("K4", 0.5), "libero10_pathPLD_K4_s075": ("K4", 0.75),
        "libero10_closelift_S_K4": ("K4", 1.0),
        "libero10_pathPLD_K8_s000": ("K8", 0.0), "libero10_pathPLD_K8_s025": ("K8", 0.25),
        "libero10_pathPLD_K8_s050": ("K8", 0.5), "libero10_pathPLD_K8_s075": ("K8", 0.75),
        "libero10_closelift_S_K8": ("K8", 1.0)}

TL = re.compile(r"^ {2}(\S.*?)\s+sizes=(\[[^\]]*\])\s+dims=(\d+)\s+peak D=\s*(-?[\d.]+)\s+"
                r"max-z=\s*(-?[\d.]+)\s+p=([\d.]+)\s+usable=(\d+)/(\d+)\s*$")


def load_reference_log(path):
    record, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#### "):
            cur = line[5:].strip()
            continue
        m = TL.match(line.rstrip("\n"))
        if m and cur:
            record[(cur, m.group(1))] = m.groups()
    if len(record) != 150:
        raise ValueError(f"Expected 150 reference records, found {len(record)}")
    return record


def run(dataset_dir, split_dir, output, reference_log=None):
    if output.exists():
        raise FileExistsError(output)
    files = sorted(dataset_dir.glob("*.hdf5"))
    if len(files) != 10:
        raise ValueError(f"Expected 10 LIBERO task files, found {len(files)}")
    BASE = {ti: p.name[:46] for ti, p in enumerate(files)}
    record = load_reference_log(reference_log) if reference_log else None
    rows, reg_bad = [], 0
    for cell in CELLS:
        with (split_dir / (cell + ".json")).open(encoding="utf-8") as handle:
            j = json.load(handle)
        cl = j.get("clients", j)
        ks = sorted([k for k in cl if k.startswith("client_") and isinstance(cl[k], list)],
                    key=lambda k: int(k.split("_")[1]))
        assign = {}
        for gi, k in enumerate(ks):
            for e in cl[k]:
                assign[str(e)] = gi
        for ti in range(10):
            fh = h5py.File(files[ti], "r")
            d = fh["data"]
            raw = {}
            for nm in sorted(d.keys(), key=lambda k: int(k.split("_")[1])):
                if "%d|%s" % (ti, nm) not in assign:
                    continue
                g = d[nm]
                raw[nm] = (np.asarray(g["states"][:], dtype=float),
                           np.asarray(g["actions"][:], dtype=float), assign["%d|%s" % (ti, nm)])
            fh.close()
            Sl = list(raw.values())
            early = np.array([v[0][:max(3, len(v[0]) // 5)].std(0) for v in Sl]).mean(0)
            across = np.array([v[0][0] for v in Sl]).std(0)
            sit = (early < np.median(early)) & (across > np.median(across))
            if sit.sum() < 3:
                sit = across > np.percentile(across, 60)
            O, A_, demo, bins, groups = [], [], [], [], {}
            for i, (nm, (st, a, gi)) in enumerate(sorted(raw.items())):
                T = min(len(st), len(a))
                if T < M:
                    continue
                idx = np.linspace(0, T - 1, M).astype(int)
                O.append(st[idx][:, sit]); A_.append(chunk_actions(a, idx, H_CHUNK))
                demo.append(np.full(M, i)); groups.setdefault(gi, []).append(i)
                bins.append(np.minimum((idx / max(T - 1, 1) * 5).astype(int), 4))
            O = np.concatenate(O); A_ = np.concatenate(A_)
            demo = np.concatenate(demo); bins = np.concatenate(bins)
            keep = O.std(0) > 1e-8
            O = (O[:, keep] - O[:, keep].mean(0)) / (O[:, keep].std(0) + 1e-8)
            A_ = (A_ - A_.mean(0)) / (A_.std(0) + 1e-8)
            gl = [groups[k] for k in sorted(groups)]
            Bn = int(bins.max()) + 1
            pre = [(bins == b,) + build(O[bins == b], A_[bins == b], demo[bins == b])
                   for b in range(Bn)]
            d2g = {dd: gi for gi, g in enumerate(gl) for dd in g}
            lab0 = np.array([d2g[dd] for dd in demo])
            obs_S, used = [], []
            for b in range(Bn):
                m, order, kp, ad = pre[b]
                s, u = S_of(order, kp, ad, lab0[m], K_NN)
                obs_S.append(s); used.append(u)
            obs_S = np.asarray(obs_S)
            obs_bad = int(np.sum(~np.isfinite(obs_S)))

            sizes = [len(g) for g in gl]
            alld = np.concatenate([np.asarray(g) for g in gl])
            rng = np.random.default_rng(SEED)
            null = np.empty((PERMS, Bn))
            for pi in range(PERMS):
                p = rng.permutation(alld)
                mm, cut = {}, 0
                for gi, sz in enumerate(sizes):
                    for dd in p[cut:cut + sz]:
                        mm[dd] = gi
                    cut += sz
                lab = np.array([mm[dd] for dd in demo])
                for b in range(Bn):
                    m, order, kp, ad = pre[b]
                    null[pi, b] = S_of(order, kp, ad, lab[m], K_NN)[0]
            perm_bad = int(np.sum(~np.isfinite(null).all(axis=1)))

            peakS = float(np.nanmax(obs_S))
            pk_null = np.nanmax(null, 1)
            pS = (1 + int(np.sum(pk_null >= peakS))) / (1 + PERMS)
            mu, sd = np.nanmean(null, 0), np.nanstd(null, 0)
            z = (obs_S - mu) / np.where(sd > 0, sd, np.nan)
            maxz = float(np.nanmax(z))
            zn = (null - mu) / np.where(sd > 0, sd, np.nan)
            pz = (1 + int(np.sum(np.nanmax(zn, 1) >= maxz))) / (1 + PERMS)
            peakD = float(np.nanmax(obs_S / mu))

            ok = None
            if record is not None:
                rec = record[(cell, BASE[ti])]
                ok = (("%6.3f" % peakD, "%6.2f" % maxz, "%.4f" % pz, str(sum(used)))
                      == ("%6.3f" % float(rec[3]), "%6.2f" % float(rec[4]), "%.4f" % float(rec[5]),
                          rec[6]))
                if not ok:
                    reg_bad += 1
                    print("REGRESSION MISMATCH %s t%d" % (cell, ti))
            K, sv = SVAL[cell]
            rows.append(dict(cell=cell, K=K, s=sv, task=ti, peakS=peakS, pS=pS,
                             lo=float(np.percentile(pk_null, 5)),
                             hi=float(np.percentile(pk_null, 95)),
                             used=sum(used), per_bin_used=used, obs_bad=obs_bad,
                             perm_bad=perm_bad, maxz=maxz, pz=pz, peakD=peakD, reg_ok=ok))
        print("  setting done: %s" % cell)
        sys.stdout.flush()

    if record is not None:
        print("\nlog regression: %d of %d instances reproduce the record at its stored precision"
              % (len(rows) - reg_bad, len(rows)))
        if reg_bad:
            print("ABORT: regression failed on %d instances." % reg_bad)
            sys.exit(1)

    n_obs_ok = sum(1 for r in rows if r["obs_bad"] == 0)
    n_perm_ok = sum(1 for r in rows if r["perm_bad"] == 0)
    n_both = sum(1 for r in rows if r["obs_bad"] == 0 and r["perm_bad"] == 0)
    print("\nVALIDITY over all %d task-level instances (15 (K,s) settings x 10 tasks):" % len(rows))
    print("  observed value reportable (all 5 bins computable) : %d" % n_obs_ok)
    print("  permutation test reportable (0 failing rows)      : %d" % n_perm_ok)
    print("  both                                              : %d" % n_both)
    for r in rows:
        if r["obs_bad"] or r["perm_bad"]:
            print("    %-30s t%-2d obs_bad %d  perm_bad %d  usable %d"
                  % (r["cell"], r["task"], r["obs_bad"], r["perm_bad"], r["used"]))

    print("\nsupport, min per-bin usable across instances: %d ; max: %d"
          % (min(min(r["per_bin_used"]) for r in rows), max(max(r["per_bin_used"]) for r in rows)))
    with output.open("x", encoding="utf-8") as handle:
        payload = [{key: (None if isinstance(value, float) and not math.isfinite(value)
                          else value) for key, value in row.items()} for row in rows]
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("\nMEDIAN OVER TASKS per setting -- descriptive summary only, no common band")
    print("%-4s %6s %10s %8s %10s" % ("K", "s", "median T", "n tasks", "all 10?"))
    for cell in CELLS:
        sub = [r for r in rows if r["cell"] == cell and r["obs_bad"] == 0]
        K, sv = SVAL[cell]
        print("%-4s %6.2f %10.4f %8d %10s"
              % (K, sv, float(np.median([r["peakS"] for r in sub])), len(sub),
                 "yes" if len(sub) == 10 else "NO -- do not call it a ten-task median"))
    print("\nS = 1 means the two mean distances are equal. It is NOT a significance threshold for T.")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path)
    args = parser.parse_args()
    run(args.dataset_dir, args.split_dir, args.output, args.reference_log)


if __name__ == "__main__":
    main()
