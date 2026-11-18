# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Post-processing of the revision runs.
  python rev_post.py union <file>   common grade model reported under the four shell masks
  python rev_post.py disc           block discretisation (signed-distance shell, 1/8/27 nodes per block)"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.spatial import cKDTree
SH = ['reference', 'indicator', 'signed_distance', 'contacts']; OZ = 31.1035; COG = 0.5
Z = np.load('grid.npz'); NB, BS, LO = Z['NB'], Z['BS'], Z['LO']
DEN = pd.read_csv('comp_density.csv'); DT = cKDTree(DEN[['X', 'Y', 'Z']].to_numpy())
def rho(bid):
    i, j, k = np.unravel_index(bid, tuple(NB)); return DEN.rho.to_numpy()[DT.query(LO + BS * (np.c_[i, j, k] + 0.5))[1]]
def tm(bid, bvol, bg):
    t = bvol * rho(bid); sel = bg >= COG
    return (t[:, None] * sel).sum(0), (t[:, None] * bg * sel).sum(0)
if sys.argv[1] == 'union':
    U = np.load(sys.argv[2]); X, zc, w, sy = U['X'], U['zc'], U['w'], U['sy'].astype(float)
    o = np.argsort(zc); zs = zc[o]; ps = np.cumsum(w[o]) - w[o] / 2
    sim = np.interp(norm.cdf(sy), np.r_[0.0, ps, 1.0], np.r_[0.0, zs, 30.0])
    nb = U['bid'][U['node_block']]; nv = float(U['node_vol'])
    res, rows = {}, []
    for s in SH:
        m = U[f'in_{s}']; ub, inv, cnt = np.unique(nb[m], return_inverse=True, return_counts=True)
        bg = np.stack([np.bincount(inv, weights=sim[m, r], minlength=len(ub)) / cnt for r in range(sim.shape[1])], 1)
        T, M = tm(ub, cnt * nv, bg); res[s] = dict(T=T, M=M)
        q = lambda a: np.percentile(a, [10, 50, 90])
        rows.append(dict(shell=s, volume_Mm3=cnt.sum() * nv / 1e6, T_P10=q(T)[0] / 1e6, T_P50=q(T)[1] / 1e6, T_P90=q(T)[2] / 1e6, G_P50=np.median(M / T),
                         koz_P10=q(M)[0] / OZ / 1e3, koz_P50=q(M)[1] / OZ / 1e3, koz_P90=q(M)[2] / OZ / 1e3))
    tab = pd.DataFrame(rows); print(tab.round(3).to_string())
    dec = {}
    for qn in ('T', 'M'):
        Xq = np.stack([res[s][qn] for s in SH]); b = Xq.mean(1).var(); wv = Xq.var(1, ddof=1).mean(); dec[qn] = dict(between=b, within=wv, share=b / (b + wv))
        Xd = np.stack([res[s][qn] for s in ('indicator', 'signed_distance', 'contacts')]); b = Xd.mean(1).var(); wv = Xd.var(1, ddof=1).mean(); dec[qn + '_deeprbf'] = dict(share=b / (b + wv))
        dec[qn + '_rel_spread'] = float(Xq.mean(1).std() / Xq.mean())
    print(json.dumps(dec, indent=1, default=float))
    tab.to_csv('rev/common_grade_model_by_shell.csv', index=False, float_format='%.4f'); json.dump(dec, open('rev/common_grade_model_decomp.json', 'w'), indent=1, default=float)
else:
    out = {}
    for f in (1, 2, 3):
        S = np.load(f'rev/sim_signed_distance_f{f}.npz'); bid, bvol, bg = S['bid'], S['bvol'].astype(float), S['bg'].astype(float)
        T, M = tm(bid, bvol, bg)
        i, j, k = np.unravel_index(bid, tuple(NB)); p = (16, 16, 4)
        pid = np.ravel_multi_index((i // p[0], j // p[1], k // p[2]), tuple(-(-NB // np.array(p)))); up, inv = np.unique(pid, return_inverse=True)
        t = bvol * rho(bid); met = np.stack([np.bincount(inv, weights=(t * bg[:, r] * (bg[:, r] >= COG)), minlength=len(up)) for r in range(bg.shape[1])], 1)
        ton = np.bincount(inv, weights=t, minlength=len(up)); full = ton >= np.median(ton)
        lo, hi = np.percentile(met, [5, 95], axis=1); h = (hi - lo) / (2 * np.maximum(met.mean(1), 1e-9))
        out[f] = dict(nodes_per_block=f ** 3, blocks=int(len(bid)), volume_Mm3=float(bvol.sum() / 1e6), block_var=float(bg.var(0).mean()), block_mean=float(bg.mean()),
                      frac_above=float((bg >= COG).mean()), T_P50_Mt=float(np.median(T) / 1e6), G_P50=float(np.median(M / T)), koz_P50=float(np.median(M) / OZ / 1e3), h80=float(np.median(h[full])))
    print(json.dumps(out, indent=1)); json.dump(out, open('rev/discretisation.json', 'w'), indent=1)
