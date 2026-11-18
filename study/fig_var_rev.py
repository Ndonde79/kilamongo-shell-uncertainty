# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Variance-share figure with the revised labels (between shells / within shells)."""
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
d = pd.read_csv('out_v2/variance_decomposition.csv')
fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
for ax, q in zip(axs, ('tonnes', 'grade', 'metal')):
    v = d[d.quantity == q]; x = range(len(v)); lab = [f'{c:g}' for c in v.cutoff]
    ax.bar(x, v.share_geometry, color='#2a78d6', width=0.8); ax.bar(x, v.share_grade, bottom=v.share_geometry, color='#eb6834', width=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(lab); ax.set_title(q.capitalize()); ax.set_xlabel('Cut-off (g/t Au)'); ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color='#e4e3df', lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('Share of total variance')
fig.legend(handles=[Patch(color='#2a78d6', label='Between shells (shell-conditioned means)'), Patch(color='#eb6834', label='Within shells (grade simulation)')], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig('rev/fig_variance_decomposition_rev.png', bbox_inches='tight')
