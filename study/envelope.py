# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Reporting envelope for each shell: nodes below the collar (bench) surface, interpolated linearly between collars
(nearest collar outside their hull), and within 50 m of a sample. Writes env_masks.npz (packbits, same layout as grid.npz)."""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
DMAX = 50.0
Z = np.load('grid.npz'); LO, ND, NB, ns = Z['LO'], Z['ND'], Z['NB'], Z['ns']; NN = NB * ND
comp = pd.read_csv('composites_shells.csv'); col = pd.read_csv('../data/collar.csv')
T = cKDTree(comp[['X', 'Y', 'Z']].values); CXY, CZ = col[['XCOLLAR', 'YCOLLAR']].values, col.ZCOLLAR.values
out = {}
for sh in ['reference', 'indicator', 'signed_distance', 'contacts']:
    mem = np.unpackbits(Z['mem_' + sh])[:NN.prod()].astype(bool); idx = np.flatnonzero(mem)
    i, j, k = np.unravel_index(idx, tuple(NN)); G = np.c_[LO[0] + ns[0] * (i + .5), LO[1] + ns[1] * (j + .5), LO[2] + ns[2] * (k + .5)]
    topo = griddata(CXY, CZ, G[:, :2], method='linear'); far = np.isnan(topo); topo[far] = griddata(CXY, CZ, G[far, :2], method='nearest')
    d, _ = T.query(G); keep = (G[:, 2] < topo) & (d <= DMAX)
    m = np.zeros(NN.prod(), bool); m[idx[keep]] = True; out['env_' + sh] = np.packbits(m)
    print(sh, 'kept', round(keep.mean(), 4), 'above collar surface', round((G[:, 2] >= topo).mean(), 4), 'beyond 50 m', round((d > DMAX).mean(), 4))
np.savez_compressed('env_masks.npz', **out)
