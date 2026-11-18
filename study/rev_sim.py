# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Revision runs with the gt_sim_zone2 settings (zone variogram, envelope, cap 30, declustered normal scores, seed 20261118).
  python rev_sim.py <shell|union> <f> [nreal]
f = nodes per block edge (2 = baseline 2x2x2, 1 = one node per block, 3 = 3x3x3).
'union' simulates once inside the union of the four shells, conditioned on all composites in the union, so the
four shells can then be reported from one common grade model. Writes rev/sim_<shell>_f<f>.npz."""
import json, sys, time
import numpy as np, pandas as pd
from scipy.stats import norm
import deeprbf_recovered as n
SHELLS = ['reference', 'indicator', 'signed_distance', 'contacts']
WHO, F = sys.argv[1], int(sys.argv[2]); NREAL = int(sys.argv[3]) if len(sys.argv) > 3 else 50
Z = np.load('grid.npz'); LO, BS, ND, NB, ns0 = Z['LO'], Z['BS'], Z['ND'], Z['NB'], Z['ns']
NN0 = NB * ND; EM = np.load('env_masks.npz')
comp = pd.read_csv('composites_shells.csv'); V = json.load(open('vario_model_zone.json')); CAP = V['cap_gpt']
vg = n.GSLIBVariogram(V['nugget'], [n.GSLIBStructure('spherical', V[s]['cc'], *V[s]['ranges'], V['azimuth'], V['dip'], V['rake']) for s in ('struct1', 'struct2')])
search = n.SearchEllipsoid(*V.get('search', [80.0, 80.0, 40.0]), V['azimuth'], V['dip'], V['rake'])
def bits(a): return np.unpackbits(a)[:NN0.prod()].astype(bool)
mem0 = {s: bits(Z[f'mem_{s}']) & bits(EM[f'env_{s}']) for s in SHELLS}
inside0 = np.zeros(NN0.prod(), bool)
for s in (SHELLS if WHO == 'union' else [WHO]): inside0 |= mem0[s]
ns = BS / F; NN = NB * F
fi = np.indices(tuple(NN)).reshape(3, -1).T
cen = LO + ns * (fi + 0.5)
i0 = np.clip(np.floor((cen - LO) / ns0).astype(int), 0, NN0 - 1); id0 = np.ravel_multi_index(i0.T, tuple(NN0))
keep = inside0[id0]; fi, cen, id0 = fi[keep], cen[keep], id0[keep]
bidx = np.ravel_multi_index((fi // F).T, tuple(NB)); ub, binv, bcnt = np.unique(bidx, return_inverse=True, return_counts=True)
cols = [f'in_{s}' for s in SHELLS]
c = comp[comp[cols].any(axis=1)] if WHO == 'union' else comp[comp[f'in_{WHO}']]
c = c.copy(); c['value'] = np.minimum(c.value, CAP)
c = c.groupby([c.X.round(2), c.Y.round(2), c.Z.round(2)], as_index=False).agg(X=('X', 'mean'), Y=('Y', 'mean'), Z=('Z', 'mean'), value=('value', 'mean'))
X = c[['X', 'Y', 'Z']].to_numpy(); zc = c.value.to_numpy()
def decluster_w(P, s):
    w = np.zeros(len(P))
    for o in range(5):
        key = np.floor((P + o / 5 * s) / np.array([s, s, s / 2])).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); w += 1.0 / cnt[inv.ravel()]
    return w / w.sum()
w = decluster_w(X, V['decluster_cell_m']); o = np.argsort(zc); zs, ws = zc[o], w[o]; ps = np.cumsum(ws) - ws / 2
ys = norm.ppf(np.clip(ps, 1e-6, 1 - 1e-6)); y = np.interp(zc, zs, ys)
t0 = time.time(); print(WHO, F, 'nodes', len(cen), 'blocks', len(ub), 'data', len(zc), flush=True)
sy = np.asarray(n.sgs_gslib(X.tolist(), y.tolist(), cen.tolist(), vg, search, n_realizations=NREAL, max_data=24, max_nodes=16, min_data=0,
                            kriging_type='simple', normal_score=False, multigrid_levels=3, seed=20261118), dtype=np.float32)
sim = np.interp(norm.cdf(sy), np.r_[0.0, ps, 1.0], np.r_[0.0, zs, CAP]).astype(np.float32)
bg = np.stack([np.bincount(binv, weights=sim[:, r], minlength=len(ub)) / bcnt for r in range(NREAL)], 1).astype(np.float32)
bvol = (bcnt * ns.prod()).astype(np.float32)
extra = {}
if WHO == 'union':
    for s in SHELLS: extra[f'in_{s}'] = mem0[s][id0]
np.savez_compressed(f'rev/sim_{WHO}_f{F}.npz', bid=ub.astype(np.int32), bvol=bvol, bg=bg, node_xyz=cen.astype(np.float32), node_block=binv.astype(np.int32),
                    sy=sy if (WHO != 'union' and F == 2) or WHO == 'union' else np.zeros(1), node_vol=np.float32(ns.prod()), X=X, zc=zc, w=w, **extra)
print('done', round(time.time() - t0, 1), 's', flush=True)
