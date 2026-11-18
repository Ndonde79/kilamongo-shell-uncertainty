# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Kilamongo South grade-tonnage under geometric and grade uncertainty (step 3).
For each of four ore shells: SGS of capped Au (declustered normal scores of the composites inside the shell, one common
variogram) on a 2.5 x 2.5 x 1.25 m node grid, averaged to 5 x 5 x 2.5 m blocks; ordinary kriging of the same blocks for
comparison; grade-tonnage curves per realisation. Run on the workstation:  python gt_sim.py [n_realisations] [--quick]
Inputs (same folder): grid.npz, composites_shells.csv, vario_model.json. Output: gt_results.npz, gt_summary.json (and gt_blocks.npz with --blocks: per-block realisations for classification)."""
import json, sys, time, faulthandler
faulthandler.enable()
print("start", flush=True)
import numpy as np, pandas as pd
from scipy.stats import norm
import deeprbf_recovered as n

QUICK = '--quick' in sys.argv
NREAL = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 50
SHELLS = ['reference', 'indicator', 'signed_distance', 'contacts']
CUTS = np.array([0.0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
Z = np.load('grid.npz'); LO, BS, ND, NB, ns = Z['LO'], Z['BS'], Z['ND'], Z['NB'], Z['ns']
NN = NB * ND
comp = pd.read_csv('composites_shells.csv')
ENV = '--envelope' in sys.argv                                # reporting envelope: below the collar (bench) surface and within DMAX of a sample
DMAX = 50.0                                                   # masks precomputed by envelope.py -> env_masks.npz
if ENV: EM = np.load('env_masks.npz')
VFILE = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--vario=')), 'vario_model.json')
V = json.load(open(VFILE))
CAP = V['cap_gpt']
vg = n.GSLIBVariogram(V['nugget'], [n.GSLIBStructure('spherical', V[s]['cc'], *V[s]['ranges'], V['azimuth'], V['dip'], V['rake']) for s in ('struct1', 'struct2')])
SR = V.get('search', [80.0, 40.0, 40.0])
search = n.SearchEllipsoid(*SR, V['azimuth'], V['dip'], V['rake'])

def decluster_w(P, s):
    w = np.zeros(len(P))
    for o in range(5):
        key = np.floor((P + o / 5 * s) / np.array([s, s, s / 2])).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); w += 1.0 / cnt[inv.ravel()]
    return w / w.sum()

def table(z, w):
    o = np.argsort(z); zs, ws = z[o], w[o]; p = np.cumsum(ws) - ws / 2
    return zs, norm.ppf(np.clip(p, 1e-6, 1 - 1e-6)), p

def back(y, zs, ps, zmin=0.0, zmax=CAP):
    p = norm.cdf(y)
    return np.interp(p, np.r_[0.0, ps, 1.0], np.r_[zmin, zs, zmax])

def node_xyz(idx):
    i, j, k = np.unravel_index(idx, tuple(NN)); return np.c_[LO[0] + ns[0] * (i + 0.5), LO[1] + ns[1] * (j + 0.5), LO[2] + ns[2] * (k + 0.5)], (i // ND[0], j // ND[1], k // ND[2])

def gt(block_grade, block_vol, density=1.0):
    t = np.array([(block_vol * (block_grade >= c)).sum() for c in CUTS]) * density
    m = np.array([(block_vol * block_grade * (block_grade >= c)).sum() for c in CUTS]) * density
    return t, m

res, summ, blk = {}, {}, {}
BLOCKS = '--blocks' in sys.argv
SNAP = '--snap' in sys.argv                                   # GSLIB-style: assign data to their nearest simulation node
TAG = ('_snap' if SNAP else '') + next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--tag=')), '')
t_all = time.time()
for sh in SHELLS:
    t0 = time.time(); print('shell', sh, flush=True)
    mem = np.unpackbits(Z[f'mem_{sh}'])[:NN.prod()].astype(bool)
    idx = np.flatnonzero(mem)
    if QUICK: idx = idx[::37]
    if ENV:
        nin = len(idx); em = np.unpackbits(EM[f'env_{sh}'])[:NN.prod()].astype(bool); idx = idx[em[idx]]; env_kept = len(idx) / nin
    else:
        env_kept = 1.0
    G, (bi, bj, bk) = node_xyz(idx)
    bid = np.ravel_multi_index((bi, bj, bk), tuple(NB)); ub, binv, bcnt = np.unique(bid, return_inverse=True, return_counts=True)
    bvol = bcnt * ns.prod()                                   # inside volume of each block (m3)
    c = comp[comp[f'in_{sh}']].copy(); c['value'] = np.minimum(c.value, CAP)
    c = c.groupby([c.X.round(2), c.Y.round(2), c.Z.round(2)], as_index=False).agg(X=('X', 'mean'), Y=('Y', 'mean'), Z=('Z', 'mean'), value=('value', 'mean'))  # merge duplicate samples
    X = c[['X', 'Y', 'Z']].to_numpy(); zc = c.value.to_numpy()
    w = decluster_w(X, V['decluster_cell_m']); zs, ys, ps = table(zc, w); y = np.interp(zc, zs, ys)
    if SNAP:                                                  # data falling on an inside node fix that node; nodes averaged in normal-score space
        nid = np.ravel_multi_index(np.clip(np.floor((X - LO) / ns).astype(int), 0, NN - 1).T, tuple(NN))
        pos = np.searchsorted(idx, nid); hit = (pos < len(idx)) & (idx[np.minimum(pos, len(idx) - 1)] == nid)
        un, inv2 = np.unique(pos[hit], return_inverse=True)
        yn = np.bincount(inv2, weights=y[hit]) / np.bincount(inv2)
        free = np.ones(len(idx), bool); free[un] = False
        Xd = np.vstack([G[un], X[~hit]]); yd = np.r_[yn, y[~hit]]
        simf = n.sgs_gslib(Xd.tolist(), yd.tolist(), G[free].tolist(), vg, search, n_realizations=NREAL, max_data=24, max_nodes=16, min_data=0,
                           kriging_type='simple', normal_score=False, multigrid_levels=3, seed=20261118)
        sy = np.empty((len(idx), NREAL)); sy[free] = np.asarray(simf); sy[un] = yn[:, None]
        sim = back(sy, zs, ps)
    else:
        sim = n.sgs_gslib(X.tolist(), y.tolist(), G.tolist(), vg, search, n_realizations=NREAL, max_data=24, max_nodes=16, min_data=0,
                          kriging_type='simple', normal_score=False, multigrid_levels=3, seed=20261118)
        sim = back(np.asarray(sim), zs, ps)
    bgrade = np.stack([np.bincount(binv, weights=sim[:, r], minlength=len(ub)) / bcnt for r in range(NREAL)], 1)
    ok = n.kriging_gslib(X.tolist(), zc.tolist(), G.tolist(), vg, search, max_data=24, min_data=3, kriging_type='ordinary')
    ke = np.asarray(ok['estimate']); bad = ~np.isfinite(ke) | (ke < 0) | (ke > CAP); ke = np.where(bad, np.clip(np.nan_to_num(ke, nan=(w * zc).sum()), 0, CAP), ke)
    kgrade = np.bincount(binv, weights=ke, minlength=len(ub)) / bcnt
    T, M = np.zeros((NREAL, len(CUTS))), np.zeros((NREAL, len(CUTS)))
    for r in range(NREAL): T[r], M[r] = gt(bgrade[:, r], bvol)
    kT, kM = gt(kgrade, bvol)
    res[f'{sh}_T'], res[f'{sh}_M'], res[f'{sh}_kT'], res[f'{sh}_kM'] = T, M, kT, kM
    if BLOCKS: blk[f'{sh}_bid'], blk[f'{sh}_bvol'], blk[f'{sh}_bg'], blk[f'{sh}_kg'] = ub.astype(np.int32), bvol.astype(np.float32), bgrade.astype(np.float32), kgrade.astype(np.float32)
    summ[sh] = dict(nodes=int(len(idx)), envelope_kept=round(env_kept, 4), blocks=int(len(ub)), volume_m3=float(bvol.sum()), composites=int(len(zc)),
                    declustered_mean=float((w * zc).sum()), ok_nan=int((~np.isfinite(np.asarray(ok['estimate']))).sum()), ok_clipped=int(bad.sum()), sim_mean=float(sim.mean()), ok_mean=float(ke.mean()), seconds=round(time.time() - t0, 1))
    print(sh, summ[sh], flush=True)
np.savez(f'gt_results{TAG}.npz', CUTS=CUTS, **res)
if BLOCKS: np.savez_compressed(f'gt_blocks{TAG}.npz', NB=NB, BS=BS, LO=LO, **blk)
summ['_meta'] = dict(realisations=NREAL, envelope=ENV, envelope_dmax=DMAX, snap=SNAP, quick=QUICK, cap=CAP, variogram={k: V[k] for k in ('nugget', 'struct1', 'struct2', 'azimuth', 'dip', 'rake')},
                     search=SR, vario_file=VFILE, block=BS.tolist(), nodes_per_block=ND.tolist(), native=n.__version__, total_minutes=round((time.time() - t_all) / 60, 1),
                     note='tonnes and metal are per unit density: multiply by density in t/m3')
json.dump(summ, open(f'gt_summary{TAG}.json', 'w'), indent=1)
