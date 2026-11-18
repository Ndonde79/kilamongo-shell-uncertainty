# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Simulation checks for each shell and the union run (node-level files rev/sim_<name>_f2.npz from rev_sim.py):
target versus realised mean (original units), normal-score mean and variance (target weighted and unweighted, realised),
and normal-score variogram reproduction across the zone and in its plane, with the range over realisations.
Writes rev/sim_checks.json."""
import json, os, sys
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree
V = json.load(open('vario_model_zone.json')); CAP = V['cap_gpt']
az, dp = np.radians(V['azimuth']), np.radians(V['dip'])
u_dd = np.array([np.sin(az) * np.cos(dp), np.cos(az) * np.cos(dp), -np.sin(dp)]); u_st = np.array([np.sin(az - np.pi / 2), np.cos(az - np.pi / 2), 0.0])
u_n = np.cross(u_dd, u_st); u_n /= np.linalg.norm(u_n); Rm = np.c_[u_dd, u_st, u_n]
def model(h):
    g = V['nugget'] * (h > 0)
    for s in ('struct1', 'struct2'):
        r = np.minimum(h / V[s]['ranges'][0], 1); g = g + V[s]['cc'] * (1.5 * r - 0.5 * r ** 3)
    return g
LAGS = {'across': (np.array([1.25, 2.5, 5.0, 10.0, 20.0, 30.0]), 0.6, lambda d, L: np.abs(d[:, 2]) >= 0.9 * L),
        'in_plane': (np.array([2.5, 5.0, 10.0, 20.0, 30.0, 40.0]), 1.2, lambda d, L: np.abs(d[:, 2]) <= 0.26 * L)}
rng = np.random.default_rng(11)
def pairs(P, lags, tol, sel, nsamp=3000, cap=150000):
    tree = cKDTree(P); idx = rng.choice(len(P), min(len(P), nsamp), replace=False); out = []
    for h in lags:
        I, J = [], []
        for a, js in zip(idx, tree.query_ball_point(P[idx], h + tol)):
            js = np.asarray(js)
            if len(js) == 0: continue
            d = P[js] - P[a]; L = np.linalg.norm(d, axis=1); ok = (np.abs(L - h) <= tol) & sel(d, L)
            I.append(np.full(ok.sum(), a)); J.append(js[ok])
        I, J = np.concatenate(I), np.concatenate(J)
        if len(I) > cap: k = rng.choice(len(I), cap, replace=False); I, J = I[k], J[k]
        out.append((I, J))
    return out
res = {}
for name in sys.argv[1:]:
    f = f'rev/sim_{name}_f2.npz'
    if not os.path.exists(f): print('missing', f); continue
    S = np.load(f); sy = S['sy'].astype(np.float64); X, zc, w = S['X'], S['zc'], S['w']
    o = np.argsort(zc); zs = zc[o]; ps = np.cumsum(w[o]) - w[o] / 2; ys = norm.ppf(np.clip(ps, 1e-6, 1 - 1e-6)); y = np.interp(zc, zs, ys)
    sim = np.interp(norm.cdf(sy), np.r_[0.0, ps, 1.0], np.r_[0.0, zs, CAP])
    q = lambda a: [float(np.percentile(a, 5)), float(np.median(a)), float(np.percentile(a, 95))]
    r = dict(nodes=int(len(sy)), realisations=int(sy.shape[1]),
             target_mean_gpt=float((w * zc).sum()), realised_mean_gpt_P5_P50_P95=q(sim.mean(0)),
             data_ns_weighted_mean=float((w * y).sum()), data_ns_weighted_var=float((w * (y - (w * y).sum()) ** 2).sum()),
             data_ns_unweighted_mean=float(y.mean()), data_ns_unweighted_var=float(y.var()),
             realised_ns_mean_P5_P50_P95=q(sy.mean(0)), realised_ns_var_P5_P50_P95=q(sy.var(0)))
    P = (S['node_xyz'].astype(float) - S['node_xyz'].mean(0)) @ Rm
    rs = np.linspace(0, sy.shape[1] - 1, 10).astype(int)
    for key, (lags, tol, sel) in LAGS.items():
        pr = pairs(P, lags, tol, sel); g = np.array([((sy[I][:, rs] - sy[J][:, rs]) ** 2).mean(0) / 2 for I, J in pr])
        r[f'vario_{key}'] = dict(lags=lags.tolist(), model=model(lags).tolist(), P5=np.percentile(g, 5, axis=1).tolist(), P50=np.median(g, axis=1).tolist(), P95=np.percentile(g, 95, axis=1).tolist(), pairs=[int(len(I)) for I, J in pr])
    res[name] = r; print(name, {k: v for k, v in r.items() if not k.startswith('vario')}, flush=True)
json.dump(res, open('rev/sim_checks_' + '_'.join(sys.argv[1:]) + '.json', 'w'), indent=1)
