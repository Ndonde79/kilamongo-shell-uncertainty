# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Revision figures: location and drilling plan with shell footprints; directional variograms with pair counts.
Also writes rev/thickness.json (vertical and approximate true thickness of each shell from the node masks)."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
SH = ['reference', 'indicator', 'signed_distance', 'contacts']
LAB = {'reference': 'Reference', 'indicator': 'Indicator', 'signed_distance': 'Signed distance', 'contacts': 'Contacts'}
COL = {'reference': '#2a78d6', 'indicator': '#eb6834', 'signed_distance': '#1baf7a', 'contacts': '#eda100'}
Z = np.load('grid.npz'); EM = np.load('env_masks.npz'); LO, ND, NB, ns = Z['LO'], Z['ND'], Z['NB'], Z['ns']; NN = NB * ND
col = pd.read_csv('../data/collar.csv'); geo = pd.read_csv('../data/geology.csv'); ht = geo.groupby('hole_id').hole_type.first()
asy = pd.read_csv('../data/assay.csv'); assayed = set(asy.iloc[:, 0])
gc = set(pd.read_csv('../data/gc_holes.csv').hole_id)
col['type'] = np.where(col.BHID.isin(gc), 'GC', col.BHID.map(ht).fillna('EXP')); col['assayed'] = col.BHID.isin(assayed)
fig, ax = plt.subplots(figsize=(6.6, 6.0))
thick = {}
for sh in SH:
    m = (np.unpackbits(Z[f'mem_{sh}'])[:NN.prod()].astype(bool) & np.unpackbits(EM[f'env_{sh}'])[:NN.prod()].astype(bool)).reshape(tuple(NN))
    foot = m.any(2).T.astype(float)
    xs = LO[0] + ns[0] * (np.arange(NN[0]) + 0.5); ys = LO[1] + ns[1] * (np.arange(NN[1]) + 0.5)
    ax.contour(xs, ys, foot, levels=[0.5], colors=COL[sh], linewidths=1.3)
    vt = m.sum(2) * ns[2]; vt = vt[vt > 0]
    thick[sh] = dict(vertical_median=float(np.median(vt)), vertical_P90=float(np.percentile(vt, 90)), true_median=float(np.median(vt) * np.cos(np.radians(20))), true_P90=float(np.percentile(vt, 90) * np.cos(np.radians(20))))
json.dump(thick, open('rev/thickness.json', 'w'), indent=1)
MK = {'RC': ('o', '#444441'), 'RAB': ('D', '#8a8984'), 'DD': ('s', '#7a3fb0'), 'EXP': ('.', '#5f5e5a'), 'GC': ('x', '#c0392b')}
NAME = {'RC': 'RC', 'RAB': 'RAB', 'DD': 'Diamond', 'EXP': 'Exploration, type not logged', 'GC': 'Grade control'}
for t, (mk, c) in MK.items():
    s = col[(col.type == t) & col.assayed]
    if len(s): ax.scatter(s.XCOLLAR, s.YCOLLAR, marker=mk, s={'.': 14, 'D': 8}.get(mk, 12), c=c, lw=0.8, label=f'{NAME[t]} ({len(s)})')
s = col[~col.assayed]; ax.scatter(s.XCOLLAR, s.YCOLLAR, marker='^', s=18, facecolors='none', edgecolors='k', lw=0.7, label=f'No assays ({len(s)})')
c0 = np.array([621834.0, 9802150.0]); u = np.array([np.sin(np.radians(75)), np.cos(np.radians(75))]); v = np.array([-u[1], u[0]])
p0 = c0 + 10 * v; a, b = p0 - 260 * u, p0 + 260 * u
ax.plot([a[0], b[0]], [a[1], b[1]], 'k--', lw=1); ax.text(b[0], b[1], ' Fig. 6 section', fontsize=8, va='center')
ax.set_aspect('equal'); ax.set_xlabel('Easting (m, UTM 36S)'); ax.set_ylabel('Northing (m)')
ax.ticklabel_format(useOffset=False, style='plain'); ax.tick_params(axis='x', labelrotation=30)
h = [Line2D([0], [0], color=COL[sh], lw=1.5, label=f'{LAB[sh]} shell') for sh in SH]
ax.legend(handles=h + ax.get_legend_handles_labels()[0], loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
ax.annotate('N', xy=(0.04, 0.95), xycoords='axes fraction', ha='center', fontsize=11, fontweight='bold')
ax.annotate('', xy=(0.04, 0.93), xytext=(0.04, 0.83), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.2))
fig.savefig('rev/fig_location_plan.png', bbox_inches='tight'); plt.close(fig)
V = json.load(open('vario_model_zone.json')); E = V['experimental']
def model(h):
    g = V['nugget'] * (h > 0)
    for s in ('struct1', 'struct2'):
        a = V[s]['ranges'][0]; r = np.minimum(h / a, 1); g = g + V[s]['cc'] * (1.5 * r - 0.5 * r ** 3)
    return g
fig, axs = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
spec = [('normal', 'Across the zone (normal)'), ('down_dip', 'Down dip (075/20)'), ('strike', 'Along strike (165/00)')]
for ax, (k, t) in zip(axs, spec):
    e = E[k]; h = np.array(e['lags'], float); g = np.array(e['gamma']); n = np.array(e['pairs'])
    ax.scatter(h, g, s=np.clip(n / 300, 6, 60), c='#2a78d6', zorder=3, label='Experimental (size ∝ pairs)')
    for hi, gi, ni in zip(h, g, n):
        if k == 'normal' and hi not in (1, 4, 8, 12, 20, 30): continue
        if k != 'normal' and hi in (25, 50): continue
        ax.annotate(f'{ni}', (hi, gi), textcoords='offset points', xytext=(0, 6), fontsize=8, ha='center', color='#444441')
    hh = np.linspace(0, max(h.max(), 80), 300); ax.plot(hh, model(hh), color='#c0392b', lw=1.5, label='Isotropic model')
    ax.axhline(1, color='#8a8984', lw=0.7, ls=':'); ax.set_title(t); ax.set_xlabel('Lag (m)'); ax.set_ylim(0, 2.0); ax.set_xlim(0, hh.max())
axs[0].set_ylabel('Standardised variogram (normal scores)')
axs[0].legend(loc='lower right', fontsize=7, frameon=False)
fig.text(0.5, -0.04, 'Model: nugget 0.19 + Sph 0.16 (2.0 m) + Sph 0.65 (19.9 m), isotropic; experimental values divided by the unweighted variance of the pooled normal scores (0.89). Numbers are pair counts.', ha='center', fontsize=7.5)
fig.tight_layout(); fig.savefig('rev/fig_variogram.png', bbox_inches='tight'); plt.close(fig)
print(json.dumps(thick, indent=1))
