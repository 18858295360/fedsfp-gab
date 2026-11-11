"""Matched-observation action disagreement with enough permutations to quote a
p-value.

The first pass used 40 shuffles, whose smallest resolvable p is 1/41 ~ 0.024, so
"p = 0.000" claimed more than the evidence allowed.  Raising the count naively is
slow, but the observation geometry does not change between shuffles -- only the
labels do.  So the neighbour ordering and the action distances along it are
computed once per bin, and each shuffle is then a vectorised pass over that
fixed structure:

  same[i, j]  = does the j-th nearest neighbour of i share i's client
  r_index     = position of the k-th same-client neighbour (its radius)
  cross       = first k neighbours before r_index with a different client

p is reported as (1 + #{shuffle max-z >= observed}) / (1 + n_shuffles), the
standard permutation estimate, which never returns exactly zero.
"""
import numpy as np

NEI = 400          # neighbours retained per point; far beyond any k used


def build(O, A, demo_of):
    n = len(O)
    d2 = ((O[:, None, :] - O[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    d2 = np.where(demo_of[:, None] == demo_of[None, :], np.inf, d2)
    m = min(NEI, n)
    order = np.argsort(d2, 1)[:, :m]
    keep = np.isfinite(np.take_along_axis(d2, order, 1))
    adist = np.linalg.norm(A[order] - A[:, None, :], axis=2)
    return order, keep, adist


def S_of(order, keep, adist, lab, k):
    n, m = order.shape
    same = (lab[order] == lab[:, None]) & keep
    cross = (~(lab[order] == lab[:, None])) & keep
    cs_same = np.cumsum(same, 1)
    ok_i = cs_same[:, -1] >= k
    # radius = position of the k-th same-client neighbour
    rpos = np.argmax(cs_same >= k, 1)
    within = np.arange(m)[None, :] <= rpos[:, None]
    cs_cross = np.cumsum(cross & within, 1)
    ok_j = cs_cross[:, -1] >= k
    use = ok_i & ok_j
    if not use.any():
        return np.nan, 0
    d_same = np.where(same & (np.cumsum(same, 1) <= k), adist, 0.0).sum(1) / k
    d_cross = np.where(cross & within & (cs_cross <= k), adist, 0.0).sum(1) / k
    return float(d_cross[use].mean() / d_same[use].mean()), int(use.sum())


def phase_scan_fast(O, A, demo_of, bins, groups, k=6, perms=1000, seed=20260816):
    B = int(bins.max()) + 1
    pre = []
    for b in range(B):
        m = bins == b
        pre.append((m, *build(O[m], A[m], demo_of[m])))
    demo2g = {d: gi for gi, g in enumerate(groups) for d in g}
    lab_all = np.array([demo2g[d] for d in demo_of])

    obs_S, used = [], []
    for b in range(B):
        m, order, keep, adist = pre[b]
        s, u = S_of(order, keep, adist, lab_all[m], k)
        obs_S.append(s); used.append(u)

    sizes = [len(g) for g in groups]
    alld = np.concatenate([np.asarray(g) for g in groups])
    rng = np.random.default_rng(seed)
    null = np.empty((perms, B))
    for p_i in range(perms):
        p = rng.permutation(alld)
        mm, cut = {}, 0
        for gi, s in enumerate(sizes):
            for d in p[cut:cut + s]:
                mm[d] = gi
            cut += s
        lab = np.array([mm[d] for d in demo_of])
        for b in range(B):
            m, order, keep, adist = pre[b]
            null[p_i, b] = S_of(order, keep, adist, lab[m], k)[0]

    mu, sd = np.nanmean(null, 0), np.nanstd(null, 0)
    z = (np.asarray(obs_S) - mu) / np.where(sd > 0, sd, np.nan)
    zn = (null - mu) / np.where(sd > 0, sd, np.nan)
    maxz_null = np.nanmax(zn, 1)
    obs_max = float(np.nanmax(z))
    hits = int((maxz_null >= obs_max).sum())
    return dict(D=(np.asarray(obs_S) / mu).tolist(), z=z.tolist(), used=used,
                max_z=obs_max, perms=perms, hits=hits,
                p=(1 + hits) / (1 + perms),
                p_floor=1.0 / (1 + perms))


def show(label, r):
    print("  === %s ===" % label)
    for b in range(len(r["D"])):
        print("    %3d-%3d%%  D_cond=%6.3f  z=%7.2f  usable=%d"
              % (20 * b, 20 * (b + 1), r["D"][b], r["z"][b], r["used"][b]))
    print("    max-z=%.2f over %d permutations: %d exceeded, p=%.4f (floor %.4f)"
          % (r["max_z"], r["perms"], r["hits"], r["p"], r["p_floor"]))
