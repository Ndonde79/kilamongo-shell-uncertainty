# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Revised precision figure: evaluated medians (solid, markers) and P10-P90 bands for 10 m panels, fitted power-law
extrapolation (dashed) with its 90% band, and the larger full-height supports (open markers)."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'savefig.dpi': 300})
SH = ['reference', 'indicator', 'signed_distance', 'contacts']
LAB = {'reference': 'Reference', 'indicator': 'Indicator', 'signed_distance': 'Signed distance', 'contacts': 'Contacts'}
COL = {'reference': '#2a78d6', 'indicator': '#eb6834', 'signed_distance': '#1baf7a', 'contacts': '#eda100'}
MK = {'reference': 'o', 'indicator': 's', 'signed_distance': '^', 'contacts': 'D'}
V = pd.read_csv('rev/panel_precision.csv'); F = json.load(open('rev/rev_stats.json'))['fits']
B = np.load('gt_blocks_v2.npz')
fig, ax = plt.subplots(figsize=(6.4, 4.6))
rng = np.random.default_rng(3)
for sh in SH:
    v = V[(V.shell == sh) & (V.rule == 'ge_median') & (V.height_m == 10)]
    ax.plot(v.panel_kt_median, 100 * v.hw50, color=COL[sh], marker=MK[sh], ms=4, lw=1.8)
    f = F[f'{sh}_ge_median']; x0 = v.panel_kt_median.max(); xs = np.geomspace(x0, max(f['T15_P95'], 1500), 60)
    ax.plot(xs, 100 * np.exp(f['intercept']) * xs ** f['slope'], color=COL[sh], lw=1.2, ls='--')
    big = V[(V.shell == sh) & (V.rule == 'ge_median') & (V.height_m > 10)]
    ax.scatter(big.panel_kt_median, 100 * big.hw50, marker=MK[sh], s=36, facecolors='white', edgecolors=COL[sh], lw=1.3, zorder=4)
    ax.errorbar([f['T15_kt']], [15], xerr=[[f['T15_kt'] - f['T15_P5']], [f['T15_P95'] - f['T15_kt']]], fmt='none', ecolor=COL[sh], elinewidth=2.2, capsize=0, alpha=0.6)
ax.axhline(15, color='#444441', ls=':', lw=1); ax.text(1.6, 16, 'h = 0.15', fontsize=8, color='#444441')
F2 = json.load(open('rev/panel_fits_two_sided.json'))
for sh in SH: ax.scatter([F2[f'{sh}_dmax']['T_kt']], [15], marker='D', s=40, facecolors=COL[sh], edgecolors='k', lw=0.6, zorder=5)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1.5, 10000); ax.set_ylim(3, 120)
ax.set_xlabel('Shell tonnage in panel (kt)'); ax.set_ylabel('90% half-width h of metal above 0.5 g/t (%)')
ax.set_yticks([5, 10, 15, 20, 30, 50, 100]); ax.set_yticklabels(['5', '10', '15', '20', '30', '50', '100'])
h = [Line2D([0], [0], color=COL[s], marker=MK[s], ms=4, lw=1.8, label=LAB[s]) for s in SH]
h += [Line2D([0], [0], color='#5f5e5a', lw=1.8, label='10 m panels, evaluated median'),
      Line2D([0], [0], color='#5f5e5a', lw=1.2, ls='--', label='Power-law extrapolation'),
      Line2D([0], [0], color='#5f5e5a', lw=2.2, alpha=0.6, label='$T_h$ (h = 0.15), 90% range of fit'),
      Line2D([0], [0], color='#5f5e5a', marker='D', mec='k', lw=0, label='$T_{\\pm}$ (two-sided criterion), extrapolated'),
      Line2D([0], [0], color='#5f5e5a', marker='o', mfc='white', lw=0, label='Full-height quarters and whole shell, evaluated')]
ax.legend(handles=h, loc='lower left', fontsize=7, frameon=False, ncol=1)
fig.tight_layout(); fig.savefig('rev/fig_uncertainty_vs_tonnage_rev.png'); print('ok')
