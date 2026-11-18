# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Step 2 (revised): one normal-score variogram for all shells, oriented on the gently ENE-dipping mineralised zone.
The shells are tabular bodies dipping ~10-30 deg to the NE-E (boundary-normal and body-axis analysis, orient.py), and the
inclined holes (-60 toward 260) are drilled perpendicular to a ~30 deg ENE-dipping zone. Major axis = down-dip
(azimuth 075, dip 20), semi-major = strike (165, horizontal), minor = normal to the zone (close to down-hole in the
vertical holes). Usage: python vario4.py  -> vario_model.json (the previous model is kept as vario_model_steep.json)."""
import json, os, shutil
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from vario import decluster, ns_table, forward, sph
CAP, AZ, DIP = 30.0, 75.0, 20.0
a, b = np.radians(AZ), np.radians(DIP)
MAJ = np.array([np.sin(a) * np.cos(b), np.cos(a) * np.cos(b), -np.sin(b)])
SEM = np.array([np.sin(a + np.pi / 2), np.cos(a + np.pi / 2), 0.0])
NRM = np.cross(MAJ, SEM); NRM /= np.linalg.norm(NRM)
comp = pd.read_csv('composites_shells.csv')
U = comp.in_reference | comp.in_indicator | comp.in_signed_distance | comp.in_contacts
c = comp[U].reset_index(drop=True)
P = c[['X', 'Y', 'Z']].to_numpy(); zr = c.value.to_numpy(); z = np.minimum(zr, CAP)
dc = decluster(P); s_best, w = min(dc, key=lambda t: (t[1] * z).sum())
zs, ys = ns_table(z, w); y = forward(z, zs, ys)
VARY = float(y.var())                                       # experimental variograms are standardised by the data variance
t = cKDTree(P); pr = t.query_pairs(82, output_type='ndarray'); d = P[pr[:, 1]] - P[pr[:, 0]]; h = np.linalg.norm(d, axis=1)
cn, cm, cs = np.abs(d @ NRM) / h, np.abs(d @ MAJ) / h, np.abs(d @ SEM) / h
inplane = cn <= np.sin(np.radians(15))
def ev(mask, lags, tol):
    g, k = [], []
    for L in lags:
        s = mask & (np.abs(h - L) <= tol); k.append(int(s.sum())); g.append(0.5 * np.mean((y[pr[s, 1]] - y[pr[s, 0]]) ** 2) / VARY if s.sum() > 30 else np.nan)
    return np.array(g), np.array(k)
lN = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]); gN, nN = ev(cn >= np.cos(np.radians(25)), lN, 0.5)
lP = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60, 80]); gP, nP = ev(inplane, lP, 2.5)
gMa, nMa = ev(inplane & (cm >= np.cos(np.radians(25))), lP, 2.5); gSe, nSe = ev(inplane & (cs >= np.cos(np.radians(25))), lP, 2.5)
def mod(p, l, ax):   # p = c0, c1, a1 normal, a2 normal, k = in-plane / normal range ratio (>= 1: the zone is more continuous along than across)
    k = p[4] if ax == 'P' else 1.0; return p[0] + p[1] * sph(l, p[2] * k) + (1 - p[0] - p[1]) * sph(l, p[3] * k)
def res(p):   # normal: down-hole pairs; in-plane: down-dip pairs (strike is not resolved by the 20-40 m hole spacing and is set equal)
    mN, mP = np.isfinite(gN), np.isfinite(gMa) & (nMa >= 3000)
    return np.concatenate([(mod(p, lN[mN], 'N') - gN[mN]) * np.sqrt(nN[mN] / nN.max()), (mod(p, lP[mP], 'P') - gMa[mP]) * np.sqrt(nMa[mP] / nMa[mP].max())])
f = least_squares(res, [0.3, 0.3, 3, 20, 1.5], bounds=([0.05, 0, 0.5, 2, 1.0], [0.9, 0.95, 15, 60, 6.0]))
p = f.x
if os.path.exists('vario_model.json') and not os.path.exists('vario_model_steep.json'): shutil.copy('vario_model.json', 'vario_model_steep.json')
model = dict(cap_gpt=CAP, ns_data_variance=round(VARY, 3), decluster_cell_m=s_best, nugget=round(p[0], 3), azimuth=AZ, dip=DIP, rake=0.0,
             orientation_note='major down-dip 075/20, semi-major strike 165/00, minor normal to the gently ENE-dipping zone',
             struct1=dict(cc=round(p[1], 3), ranges=[round(p[2] * p[4], 1), round(p[2] * p[4], 1), round(p[2], 1)]),
             struct2=dict(cc=round(1 - p[0] - p[1], 3), ranges=[round(p[3] * p[4], 1), round(p[3] * p[4], 1), round(p[3], 1)]), anisotropy_ratio=round(p[4], 2),
             experimental=dict(normal=dict(lags=lN.tolist(), gamma=np.round(gN, 3).tolist(), pairs=nN.tolist(), model=np.round(mod(p, lN, 'N'), 3).tolist()),
                               in_plane=dict(lags=lP.tolist(), gamma=np.round(gP, 3).tolist(), pairs=nP.tolist(), model=np.round(mod(p, lP, 'P'), 3).tolist()),
                               down_dip=dict(lags=lP.tolist(), gamma=np.round(gMa, 3).tolist(), pairs=nMa.tolist(), model=np.round(mod(p, lP, 'P'), 3).tolist()),
                               strike=dict(lags=lP.tolist(), gamma=np.round(gSe, 3).tolist(), pairs=nSe.tolist())),
             data=dict(n=int(len(z)), n_capped=int((zr > CAP).sum()), max_gpt=float(zr.max()), raw_mean=float(zr.mean()), mean_capped=float(z.mean()),
                       declustered_mean_capped=float((w * z).sum()), metal_removed_by_cap_pct=round(100 * (1 - z.sum() / zr.sum()), 1)))
json.dump(model, open('vario_model.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in model.items() if k not in ('experimental', 'data')}))
for k, e in model['experimental'].items(): print(k, e['lags'], e['gamma'], e.get('model'), e['pairs'])
