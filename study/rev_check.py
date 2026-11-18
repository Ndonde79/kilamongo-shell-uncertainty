# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Checks on a node-level revision run: agreement with the production run, histogram and variogram reproduction
(normal scores, across-zone and in-plane), swaths of E-type against declustered composites."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.spatial import cKDTree
SH = sys.argv[1] if len(sys.argv) > 1 else 'signed_distance'
S = np.load(f'rev/sim_{SH}_f2.npz'); B = np.load('gt_blocks_v2.npz')
V = json.load(open('vario_model_zone.json'))
out = {}
a, b = S['bg'], B[f'{SH}_bg']
out['same_blocks'] = bool(np.array_equal(S['bid'], B[f'{SH}_bid']))
vol = S['bvol'].astype(float)
f = lambda g: np.array([(vol * (g[:, r] >= 0.5) * g[:, r]).sum() / (vol * (g[:, r] >= 0.5)).sum() for r in range(g.shape[1])])
t = lambda g: (vol[:, None] * (g >= 0.5)).sum(0)
out['tonnes_ratio_cloud_vs_production_P50'] = float(np.median(t(a)) / np.median(t(b))); out['grade_ratio_P50'] = float(np.median(f(a)) / np.median(f(b)))
xyz = S['node_xyz'].astype(float); sy = S['sy'].astype(float)
az, dp = np.radians(V['azimuth']), np.radians(V['dip'])
u_dd = np.array([np.sin(az) * np.cos(dp), np.cos(az) * np.cos(dp), -np.sin(dp)])
u_st = np.array([np.sin(az - np.pi / 2), np.cos(az - np.pi / 2), 0.0]); u_n = np.cross(u_dd, u_st); u_n /= np.linalg.norm(u_n)
R = np.c_[u_dd, u_st, u_n]; loc = (xyz - xyz.mean(0)) @ R
rng = np.random.default_rng(1)
def vario(vals, P, axis_sel, lags, tol):
    tree = cKDTree(P); res = []
    idx = rng.choice(len(P), min(len(P), 6000), replace=False)
    for h in lags:
        pr = tree.query_ball_point(P[idx], h + tol)
        g, n = 0.0, 0
        for ii, js in zip(idx, pr):
            js = np.array(js); 
            if len(js) == 0: continue
            d = P[js] - P[ii]; L = np.linalg.norm(d, axis=1)
            ok = (np.abs(L - h) <= tol) & axis_sel(d, L)
            if ok.any(): g += ((vals[js[ok]] - vals[ii]) ** 2).sum() / 2; n += ok.sum()
        res.append(g / max(n, 1))
    return np.array(res)
across = lambda d, L: np.abs(d[:, 2]) >= 0.9 * L
inplane = lambda d, L: np.abs(d[:, 2]) <= 0.26 * L
def model(h):
    g = V['nugget'] * (h > 0)
    for s in ('struct1', 'struct2'):
        r = np.minimum(h / V[s]['ranges'][0], 1); g = g + V[s]['cc'] * (1.5 * r - 0.5 * r ** 3)
    return g
lags_n = np.array([1.25, 2.5, 3.75, 5, 7.5, 10, 15, 20, 30]); lags_p = np.array([2.5, 5, 7.5, 10, 15, 20, 30, 40])
vr = []
for r in range(0, sy.shape[1], 10):
    vr.append(dict(r=r, across=vario(sy[:, r], loc, across, lags_n, 0.6).tolist(), inplane=vario(sy[:, r], loc, inplane, lags_p, 1.2).tolist(), var=float(sy[:, r].var()), mean=float(sy[:, r].mean())))
out['vario'] = dict(lags_across=lags_n.tolist(), lags_inplane=lags_p.tolist(), model_across=model(lags_n).tolist(), model_inplane=model(lags_p).tolist(), real=vr)
X, zc, w = S['X'], S['zc'], S['w']
o = np.argsort(zc); cdf = np.cumsum(w[o]); qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
tq = [float(zc[o][np.searchsorted(cdf, q)]) for q in qs]
CAP = V['cap_gpt']; zs = zc[o]; ps = cdf - w[o] / 2
sim = np.interp(norm.cdf(sy), np.r_[0.0, ps, 1.0], np.r_[0.0, zs, CAP])
out['hist'] = dict(q=qs, target=tq, sim_P50_of_realisations=[float(np.median(np.quantile(sim, q, axis=0))) for q in qs], target_mean=float((w * zc).sum()), sim_mean=float(sim.mean()), sim_mean_range=[float(sim.mean(0).min()), float(sim.mean(0).max())])
et = sim.mean(1)
sw = {}
for k, ax in ((0, 'E'), (1, 'N'), (2, 'Z')):
    edges = np.arange(np.floor(xyz[:, k].min() / 25) * 25, xyz[:, k].max() + 25, 25 if k < 2 else 10)
    nb = np.digitize(xyz[:, k], edges); dbn = np.digitize(X[:, k], edges)
    rows = []
    for i in np.unique(nb):
        m = dbn == i
        if m.sum() < 10: continue
        rows.append([float(edges[i - 1]), float(et[nb == i].mean()), float((w[m] * zc[m]).sum() / w[m].sum()), int(m.sum())])
    sw[ax] = rows
out['swath'] = sw
json.dump(out, open(f'rev/check_{SH}.json', 'w'), indent=1)
print({k: v for k, v in out.items() if k in ('same_blocks', 'tonnes_ratio_cloud_vs_production_P50', 'grade_ratio_P50')}); print(out['hist'])
for v in vr[:2]: print(np.round(v['across'], 3), np.round(v['inplane'], 3), round(v['var'], 3))
print(np.round(out['vario']['model_across'], 3), np.round(out['vario']['model_inplane'], 3))
