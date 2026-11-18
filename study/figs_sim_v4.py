# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Figures of the conditional simulations: (4) a vertical section along the dip direction through all four shells,
showing two realisations, the E-type mean and the probability of exceeding 0.5 g/t Au, with the composites in the
section; (5) 3-D views of the E-type block grades above 0.5 g/t with the collar (ground) surface.
Usage: python figs_sim.py <folder with gt_blocks_v2.npz, composites_shells.csv, collar.csv>"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap
from scipy.interpolate import griddata
D = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = os.path.join(D, 'out_v2'); os.makedirs(OUT, exist_ok=True)
B = np.load(os.path.join(D, 'gt_blocks_v2.npz')); NB, BS, LO = B['NB'], B['BS'], B['LO']
comp = pd.read_csv(os.path.join(D, 'composites_shells.csv')); col = pd.read_csv(os.path.join(D, 'collar.csv'))
SH = ['reference', 'indicator', 'signed_distance', 'contacts']
LAB = {'reference': 'Reference shell', 'indicator': 'Indicator RBF', 'signed_distance': 'Signed-distance RBF', 'contacts': 'Contacts RBF'}
COG, AZ, HALF = 0.5, 75.0, 5.0
# discrete grade classes (g/t Au)
SEC_B = [0, 0.3, 0.5, 1.0, 2.0, 5.0, 1000]; SEC_L = ['< 0.3', '0.3–0.5', '0.5–1', '1–2', '2–5', '≥ 5']
SEC_CM = ListedColormap(plt.get_cmap('YlOrRd')(np.linspace(0.08, 1.0, len(SEC_L)))); SEC_N = BoundaryNorm(SEC_B, SEC_CM.N)
E_B = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1000]; E_L = ['0.5–1', '1–1.5', '1.5–2', '2–3', '3–5', '≥ 5']
E_CM = ListedColormap(plt.get_cmap('YlOrRd')(np.linspace(0.15, 1.0, len(E_L)))); E_N = BoundaryNorm(E_B, E_CM.N)
P_B = [0, 0.25, 0.5, 0.75, 0.9, 1.0001]; P_L = ['< 0.25', '0.25–0.5', '0.5–0.75', '0.75–0.9', '≥ 0.9']
P_CM = ListedColormap(plt.get_cmap('Blues')(np.linspace(0.15, 1.0, len(P_L)))); P_N = BoundaryNorm(P_B, P_CM.N)
from matplotlib.patches import Patch
def legend_patches(cm, labels): return [Patch(facecolor=cm(i), edgecolor='none', label=l) for i, l in enumerate(labels)]                     # section along the dip direction (075), +/-5 m thick
u = np.array([np.sin(np.radians(AZ)), np.cos(np.radians(AZ))]); v = np.array([u[1], -u[0]])
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
def centres(sh):
    i, j, k = np.unravel_index(B[f'{sh}_bid'], tuple(NB)); return LO + BS * (np.c_[i, j, k] + 0.5)
# section line through the centre of the union of shells, placed where the shells are widest along dip
allc = np.vstack([centres(sh) for sh in SH]); ctr = allc[:, :2].mean(0)
cands = np.arange(-120, 121, 10.0); cnt = [(np.abs((allc[:, :2] - ctr) @ v - c) <= HALF).sum() for c in cands]
V0 = cands[int(np.argmax(cnt))]
s_c = (comp[['X', 'Y']].to_numpy() - ctr) @ u; v_c = (comp[['X', 'Y']].to_numpy() - ctr) @ v; insec = np.abs(v_c - V0) <= HALF
xs = np.linspace(-300, 300, 300); P = ctr + np.outer(xs, u) + V0 * v
topo = griddata(col[['XCOLLAR', 'YCOLLAR']].to_numpy(), col.ZCOLLAR.to_numpy(), P, method='linear')
cols = [('Realisation 1', 'r0'), ('Realisation 2', 'r1'), ('E-type mean', 'e'), (f'P(Au ≥ {COG} g/t)', 'p')]
f, axs = plt.subplots(4, 4, figsize=(7.2, 7.0), sharex=True, sharey=True, constrained_layout=True)
gn, pn = LogNorm(0.05, 10), plt.Normalize(0, 1)
for r, sh in enumerate(SH):
    c = centres(sh); s = (c[:, :2] - ctr) @ u; vv = (c[:, :2] - ctr) @ v; m = np.abs(vv - V0) <= HALF
    bg = B[f'{sh}_bg'][m].astype(float)
    vals = {'r0': bg[:, 0], 'r1': bg[:, 1], 'e': bg.mean(1), 'p': (bg >= COG).mean(1)}
    for k, (title, key) in enumerate(cols):
        ax = axs[r, k]
        im = ax.scatter(s[m], c[m, 2], c=vals[key], s=3.2, marker='s', linewidths=0,
                        cmap=SEC_CM if key != 'p' else P_CM, norm=SEC_N if key != 'p' else P_N, rasterized=True)
        ax.plot(xs, topo, color='#5f5e5a', lw=0.7)
        ax.scatter(s_c[insec], comp.Z[insec], s=0.3, c='#2c2c2a', linewidths=0, rasterized=True)
        if r == 0: ax.set_title(title, fontsize=8)
        if k == 0: ax.set_ylabel(f'{LAB[sh]}\nElevation (m)', fontsize=7)
        if r == 3: ax.set_xlabel('Distance along 075° (m)', fontsize=7)
        ax.grid(False)
        if r == 3 and k in (0, 3):
            pass
    if r == 0: ims = im
f.legend(handles=legend_patches(SEC_CM, SEC_L) + [Patch(facecolor='none', edgecolor='none', label='   ')] + legend_patches(P_CM, P_L),
         labels=SEC_L + ['   '] + P_L, loc='outside lower center', ncol=12, frameon=False, fontsize=7,
         title='Block grade (g/t Au), columns 1–3            Probability of ≥ 0.5 g/t, column 4', title_fontsize=7, handlelength=1.2, columnspacing=0.8)
for ax in axs.ravel(): ax.set_xlim(-260, 260); ax.set_ylim(1300, 1450)
f.savefig(os.path.join(OUT, 'fig_sim_section.png')); plt.close(f)
print('section offset across strike (m):', V0, 'centre', ctr.round(0))
# 3-D views of E-type >= cut-off with the collar surface
from mpl_toolkits.mplot3d import Axes3D  # noqa
gx, gy = np.meshgrid(np.linspace(LO[0], LO[0] + NB[0] * BS[0], 60), np.linspace(LO[1], LO[1] + NB[1] * BS[1], 60))
gz = griddata(col[['XCOLLAR', 'YCOLLAR']].to_numpy(), col.ZCOLLAR.to_numpy(), np.c_[gx.ravel(), gy.ravel()], method='linear').reshape(gx.shape)
f = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
for n, sh in enumerate(SH):
    ax = f.add_subplot(2, 2, n + 1, projection='3d'); c = centres(sh); e = B[f'{sh}_bg'].astype(float).mean(1); k = e >= COG
    sc = ax.scatter(c[k, 0] - LO[0], c[k, 1] - LO[1], c[k, 2], c=e[k], cmap=E_CM, norm=E_N, s=1.0, marker='s', linewidths=0, depthshade=False, rasterized=True)
    ax.plot_wireframe(gx - LO[0], gy - LO[1], gz, color='#8a8984', linewidth=0.3, alpha=0.5, rstride=4, cstride=4)
    ax.set_title(LAB[sh], fontsize=9); ax.view_init(elev=28, azim=-120)
    ax.set_xlabel('E (m)', fontsize=7, labelpad=-2); ax.set_ylabel('N (m)', fontsize=7, labelpad=-2); ax.set_zlabel('Elev. (m)', fontsize=7, labelpad=-2)
    ax.tick_params(labelsize=6, pad=-1); ax.set_xlim(0, 500); ax.set_ylim(0, 450); ax.set_zlim(1300, 1450); ax.set_box_aspect((500, 450, 150 * 1.5))
f.legend(handles=legend_patches(E_CM, E_L) + [matplotlib.lines.Line2D([], [], color='#8a8984', lw=0.8, label='Ground surface (mesh)')], loc='outside lower center', ncol=7, frameon=False, fontsize=8,
         title='E-type block grade (g/t Au); local E/N from the south-west corner of the model', title_fontsize=8, handlelength=1.4)
f.savefig(os.path.join(OUT, 'fig_sim_3d.png')); plt.close(f)
