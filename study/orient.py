# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Orientation of the Kilamongo South mineralised zone from the data.
1. Shell shape: principal axes of the inside nodes and area-weighted dips/dip directions of the shell contacts.
2. Drilling: hole dips and azimuths (survey.csv).
3. Grade continuity: standardised indicator variograms (0.3 and 1.0 g/t) by direction.
4. Reporting envelope: share of each shell above the collar (bench) surface, outside the collar hull, or > 50 m from samples.
Usage: python orient.py   (needs grid.npz, composites_shells.csv, collar.csv, survey.csv)"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
Z = np.load('grid.npz'); LO, ND, NB, ns = Z['LO'], Z['ND'], Z['NB'], Z['ns']; NN = NB * ND
SH = ['reference', 'indicator', 'signed_distance', 'contacts']
comp = pd.read_csv('composites_shells.csv'); col = pd.read_csv('../data/collar.csv'); sv = pd.read_csv('../data/survey.csv')
print('Drilling:'); print(sv.assign(DIP=sv.DIP.round(0), BRG=sv.BRG.round(0)).groupby(['DIP', 'BRG']).size().sort_values(ascending=False).head(6).to_string())
STREE = cKDTree(comp[['X', 'Y', 'Z']].to_numpy())
for sh in SH:
    m = np.unpackbits(Z['mem_' + sh])[:NN.prod()].reshape(tuple(NN)).astype(float)
    f = gaussian_filter(m, sigma=(1.2, 1.2, 2.4)); g = np.stack(np.gradient(f, *ns), -1)
    e = (f > 0.35) & (f < 0.65); G = g[e]; w = np.linalg.norm(G, axis=1); nrm = G / w[:, None]; nrm *= np.where(nrm[:, 2:3] < 0, -1, 1)
    dip = np.degrees(np.arccos(np.clip(nrm[:, 2], -1, 1))); dd = np.degrees(np.arctan2(nrm[:, 0], nrm[:, 1])) % 360
    h, _ = np.histogram(dip, bins=[0, 15, 30, 45, 60, 75, 90], weights=w)
    k = dip < 30; hd, _ = np.histogram(dd[k], bins=np.arange(0, 361, 45), weights=w[k])
    P = np.argwhere(m > 0) * ns; ev, V = np.linalg.eigh(np.cov((P - P.mean(0)).T)); v = V[:, 0] * np.sign(V[2, 0])
    print(f'\n{sh}: contact dip % (0-15,15-30,...,75-90) {(100*h/h.sum()).round(0)}; gentle contacts dip-direction % N..NW {(100*hd/hd.sum()).round(0)}')
    print(f'  thinnest body axis plunge {np.degrees(np.arcsin(abs(v[2]))):.0f} toward {np.degrees(np.arctan2(v[0], v[1])) % 360:.0f}; axis sd (m) {np.sqrt(ev).round(1)}')
    idx = np.flatnonzero(m.ravel() > 0)[::7]; i, j, kk = np.unravel_index(idx, tuple(NN))
    X = np.c_[LO[0] + ns[0] * (i + .5), LO[1] + ns[1] * (j + .5), LO[2] + ns[2] * (kk + .5)]
    topo = griddata(col[['XCOLLAR', 'YCOLLAR']].to_numpy(), col.ZCOLLAR.to_numpy(), X[:, :2], method='linear'); d, _ = STREE.query(X)
    print(f'  above collar surface {np.nanmean(X[:, 2] > topo):.2f}; outside collar hull {np.isnan(topo).mean():.2f}; >50 m from samples {np.mean(d > 50):.2f}')
z = np.minimum(comp.value.to_numpy(), 30); P = comp[['X', 'Y', 'Z']].to_numpy()
pr = cKDTree(P).query_pairs(32, output_type='ndarray'); d = P[pr[:, 1]] - P[pr[:, 0]]; h = np.linalg.norm(d, axis=1); u = d / h[:, None]
def gam(I, az, dp, lag, tol=22.5, lt=4):
    a, b = np.radians(az), np.radians(dp); v = np.array([np.sin(a) * np.cos(b), np.cos(a) * np.cos(b), -np.sin(b)])
    k = (np.abs(u @ v) >= np.cos(np.radians(tol))) & (np.abs(h - lag) <= lt)
    return 0.5 * ((I[pr[k, 0]] - I[pr[k, 1]]) ** 2).mean() / I.var()
for cut in (0.3, 1.0):
    I = (z >= cut).astype(float)
    print(f'\nIndicator {cut} g/t, standardised gamma: down-hole 2/8 m {gam(I,0,90,2,15,1):.2f}/{gam(I,0,90,8,15,3):.2f};',
          'horizontal 12 m by azimuth 0..150:', [round(gam(I, a, 0, 12), 2) for a in range(0, 180, 30)],
          f'; ENE 20-deg dipping 12 m {gam(I,75,20,12):.2f}')
