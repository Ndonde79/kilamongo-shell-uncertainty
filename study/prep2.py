# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Kilamongo grade-tonnage study (dataset v2: exploration + grade-control export), step 1: simulation grid and shell
membership. DeepRBF shells: workflow fields on the 2 m grid (inside = field >= iso; masks from build_masks.py).
Reference shell: point-in-solid test against the commercial-software wireframe (unchanged; built on the earlier data).
Usage: python prep2.py <v2 folder with out/*_composites.csv> <shell_masks_2m.npz> <reference.dtm>"""
import sys, time, json
import numpy as np, pandas as pd
from deeprbf.workflow.mesh import read_wireframe, inside
K, MASKS, REF = sys.argv[1], sys.argv[2], sys.argv[3]
LO, HI = np.array([621590.0, 9801940.0, 1288.0]), np.array([622090.0, 9802390.0, 1446.6])
BS = np.array([5.0, 5.0, 2.5]); ND = np.array([2, 2, 2]); NB = np.floor((HI - LO) / BS).astype(int); ns = BS / ND
ax = [LO[i] + ns[i] / 2 + ns[i] * np.arange(NB[i] * ND[i]) for i in range(3)]
G = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
comp = pd.read_csv(f'{K}/out/Kilamongo_Au_0.3_indicator_composites.csv'); comp = comp[comp.sampled > 0].reset_index(drop=True)
Z = np.load(MASKS)
def from_mask(name, P):
    shp = tuple(Z[name + '_shape']); m = np.unpackbits(Z[name])[:np.prod(shp)].reshape(shp).astype(bool)
    idx = np.rint((P - LO) / 2.0).astype(int); ok = np.all((idx >= 0) & (idx < np.array(shp)), axis=1)
    out = np.zeros(len(P), bool); out[ok] = m[idx[ok, 0], idx[ok, 1], idx[ok, 2]]; return out
mem, cin, info = {}, {}, {}
P = comp[['X', 'Y', 'Z']].to_numpy(); T = read_wireframe(REF)
for name in ['reference', 'indicator', 'signed_distance', 'contacts']:
    t = time.time()
    if name == 'reference': m = inside(T, G); cin[name] = inside(T, P)
    else: m = from_mask(name, G); cin[name] = from_mask(name, P)
    mem[name] = m
    info[name] = dict(node_volume_m3=float(m.sum() * ns.prod()), composites_inside=int(cin[name].sum()), mean_au_inside=float(comp.value[cin[name]].mean()), seconds=round(time.time() - t, 1))
    print(name, info[name], flush=True)
np.savez_compressed('grid.npz', LO=LO, BS=BS, ND=ND, NB=NB, ns=ns, **{f'mem_{k}': np.packbits(v) for k, v in mem.items()})
for k, v in cin.items(): comp[f'in_{k}'] = v
comp.to_csv('composites_shells.csv', index=False); json.dump(info, open('shells.json', 'w'), indent=1)
