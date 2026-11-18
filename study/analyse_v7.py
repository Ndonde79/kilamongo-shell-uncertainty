# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Kilamongo South grade-tonnage under geometric and grade uncertainty (step 4: analysis).
Reads gt_results.npz, gt_blocks.npz, gt_summary.json from gt_sim.py. Writes tables (CSV/JSON) and figures (PNG) to ./out.
Density 2.75 t/m3. Classification from simulated uncertainty: a block is Measured if the metal above the reporting
cut-off in its quarterly-production panel is known within +/-15 % at 90 % confidence, Indicated if the same holds for its
annual panel, otherwise Inferred. Panel sizes are assumptions for Competent Person review.
Usage: python analyse.py [folder]"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else '.'
TAG = sys.argv[2] if len(sys.argv) > 2 else ''            # '' or '_snap'
OUT = os.path.join(D, 'out' + TAG); os.makedirs(OUT, exist_ok=True)
RHO = 2.75                                  # fallback only; block density comes from comp_density.csv (nearest logged SG)
COG = 0.5                                   # reporting cut-off, g/t
TOL, CONF = 0.15, 0.90
PANEL_Q = (8, 8, 4)                         # blocks: 40 x 40 x 10 m  = 16 000 m3 ~ 44 kt  (quarter)
PANEL_A = (16, 16, 4)                       # blocks: 80 x 80 x 10 m  = 64 000 m3 ~ 176 kt (year)
SH = ['reference', 'indicator', 'signed_distance', 'contacts']
LAB = {'reference': 'Reference shell', 'indicator': 'Indicator RBF', 'signed_distance': 'Signed-distance RBF', 'contacts': 'Contacts RBF'}
COL = {'reference': '#2a78d6', 'indicator': '#eb6834', 'signed_distance': '#1baf7a', 'contacts': '#eda100'}
MK = {'reference': 'o', 'indicator': 's', 'signed_distance': '^', 'contacts': 'D'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8984', 'savefig.dpi': 300})

R = np.load(os.path.join(D, f'gt_results{TAG}.npz')); B = np.load(os.path.join(D, f'gt_blocks{TAG}.npz'))
S = json.load(open(os.path.join(D, f'gt_summary{TAG}.json')))
CUTS = R['CUTS']; NB = B['NB']
from scipy.spatial import cKDTree
DEN = pd.read_csv(os.path.join(D, 'comp_density.csv')); DT = cKDTree(DEN[['X', 'Y', 'Z']].to_numpy())
LOB, BSB = B['LO'], B['BS']
RHOB, GT = {}, {}
for sh in SH:                                                # block density = SG of the nearest logged interval; tonnes and metal per realisation
    bid = B[f'{sh}_bid']; i, j, k = np.unravel_index(bid, tuple(NB)); xyz = LOB + BSB * (np.c_[i, j, k] + 0.5)
    rb = DEN.rho.to_numpy()[DT.query(xyz)[1]]; RHOB[sh] = rb
    tb = B[f'{sh}_bvol'].astype(float) * rb; bg = B[f'{sh}_bg'].astype(float); kg = B[f'{sh}_kg'].astype(float)
    T = np.stack([(tb[:, None] * (bg >= c)).sum(0) for c in CUTS], 1); M = np.stack([(tb[:, None] * bg * (bg >= c)).sum(0) for c in CUTS], 1)
    GT[sh] = (T, M, np.array([(tb * (kg >= c)).sum() for c in CUTS]), np.array([(tb * kg * (kg >= c)).sum() for c in CUTS]))

# ---------- 1. grade-tonnage per shell: P10/P50/P90 of tonnes, grade, metal; OK ----------
rows = []
for sh in SH:
    T, M = GT[sh][0], GT[sh][1]                               # t, g
    G = np.divide(M, T, out=np.zeros_like(M), where=T > 0)
    kT, kM = GT[sh][2], GT[sh][3]
    for j, c in enumerate(CUTS):
        q = lambda a: np.percentile(a[:, j], [10, 50, 90])
        t, g, m = q(T), q(G), q(M / 31.1035)
        rows.append(dict(shell=sh, cutoff=c, T_P10=t[0] / 1e6, T_P50=t[1] / 1e6, T_P90=t[2] / 1e6, T_mean=T[:, j].mean() / 1e6,
                         G_P10=g[0], G_P50=g[1], G_P90=g[2], G_mean=G[:, j].mean(),
                         koz_P10=m[0] / 1e3, koz_P50=m[1] / 1e3, koz_P90=m[2] / 1e3, koz_mean=M[:, j].mean() / 31.1035e3,
                         T_OK=kT[j] / 1e6, G_OK=kM[j] / kT[j] if kT[j] > 0 else 0, koz_OK=kM[j] / 31.1035e3))
gtab = pd.DataFrame(rows); gtab.to_csv(os.path.join(OUT, 'grade_tonnage.csv'), index=False, float_format='%.4f')

# ---------- 2. decomposition: geometry (between shells) vs grade (within shell), law of total variance ----------
dec = []
for j, c in enumerate(CUTS):
    for name, f in (('tonnes', lambda sh: GT[sh][0][:, j]),
                    ('metal', lambda sh: GT[sh][1][:, j]),
                    ('grade', lambda sh: np.divide(GT[sh][1][:, j], GT[sh][0][:, j], out=np.zeros(GT[sh][0].shape[0]), where=GT[sh][0][:, j] > 0))):
        X = np.stack([f(sh) for sh in SH])                     # shells x realisations, shells equally likely
        within = X.var(axis=1, ddof=1).mean(); between = X.mean(axis=1).var(ddof=0)
        tot = within + between; mu = X.mean()
        dec.append(dict(cutoff=c, quantity=name, mean=mu, cv_total=np.sqrt(tot) / mu, cv_geometry=np.sqrt(between) / mu,
                        cv_grade=np.sqrt(within) / mu, share_geometry=between / tot, share_grade=within / tot,
                        ref_vs_ok=None))
dtab = pd.DataFrame(dec); dtab.to_csv(os.path.join(OUT, 'variance_decomposition.csv'), index=False, float_format='%.4f')

# ---------- 3. classification from simulated uncertainty ----------
def panel_ok(bid, bvol, bg, panel, rb):
    i, j, k = np.unravel_index(bid, tuple(NB))
    pid = np.ravel_multi_index((i // panel[0], j // panel[1], k // panel[2]), tuple(-(-NB // np.array(panel))))
    up, inv = np.unique(pid, return_inverse=True)
    sel = (bg >= COG) * (bvol * rb)[:, None]                    # tonnes above cut-off per block, per realisation
    metal = np.stack([np.bincount(inv, weights=(sel * bg)[:, r], minlength=len(up)) for r in range(bg.shape[1])], 1)
    lo, hi = np.percentile(metal, [50 * (1 - CONF), 100 - 50 * (1 - CONF)], axis=1)
    mu = metal.mean(1); half = np.divide(hi - lo, 2 * mu, out=np.full(len(mu), np.inf), where=mu > 0)
    return (half <= TOL)[inv], half[inv]

cls_rows, cls_maps = [], {}
for sh in SH:
    bid, bvol, bg, kg = B[f'{sh}_bid'], B[f'{sh}_bvol'].astype(float), B[f'{sh}_bg'].astype(float), B[f'{sh}_kg'].astype(float)
    mq, hq = panel_ok(bid, bvol, bg, PANEL_Q, RHOB[sh]); ma, ha = panel_ok(bid, bvol, bg, PANEL_A, RHOB[sh])
    cat = np.where(mq, 1, np.where(ma, 2, 3))                  # 1 Measured, 2 Indicated, 3 Inferred
    cls_maps[sh] = (bid, cat)
    e = bg.mean(1)                                             # E-type block grade
    for cval, cname in ((1, 'Measured'), (2, 'Indicated'), (3, 'Inferred')):
        s = (cat == cval) & (e >= COG)
        t = (bvol[s] * RHOB[sh][s]).sum(); m = (bvol[s] * RHOB[sh][s] * e[s]).sum()
        cls_rows.append(dict(shell=sh, category=cname, Mt=t / 1e6, grade=m / t if t else 0, koz=m / 31.1035e3,
                             share_volume=bvol[cat == cval].sum() / bvol.sum()))
# ---------- 3b. uncertainty against production volume (classification volume) ----------
SIZES = [20, 30, 40, 60, 80, 120, 160, 240]
PROD_KT = 176.0                                            # assumed annual production, kt                     # panel side (m), 10 m high (4 blocks)
vrows = []
for sh in SH:
    bid, bvol, bg = B[f'{sh}_bid'], B[f'{sh}_bvol'].astype(float), B[f'{sh}_bg'].astype(float)
    i, j, k = np.unravel_index(bid, tuple(NB))
    for L in SIZES:
        p = (L // 5, L // 5, 4)
        pid = np.ravel_multi_index((i // p[0], j // p[1], k // p[2]), tuple(-(-NB // np.array(p))))
        up, inv = np.unique(pid, return_inverse=True)
        sel = (bg >= COG) * (bvol * RHOB[sh])[:, None]
        metal = np.stack([np.bincount(inv, weights=(sel * bg)[:, r], minlength=len(up)) for r in range(bg.shape[1])], 1)
        ton = np.bincount(inv, weights=bvol * RHOB[sh], minlength=len(up))
        lo, hi = np.percentile(metal, [50 * (1 - CONF), 100 - 50 * (1 - CONF)], axis=1); mu = metal.mean(1)
        hw = np.divide(hi - lo, 2 * mu, out=np.full(len(mu), np.inf), where=mu > 0)
        full = ton >= np.median(ton)                             # the better-filled half of the panels
        vrows.append(dict(shell=sh, side_m=L, panel_kt=float(np.median(ton[full])) / 1e3, n_panels=int(full.sum()),
                          hw_P10=np.percentile(hw[full], 10), hw_P50=np.percentile(hw[full], 50), hw_P90=np.percentile(hw[full], 90),
                          frac_within_tol=float((hw[full] <= TOL).mean())))
vtab = pd.DataFrame(vrows); vtab.to_csv(os.path.join(OUT, 'uncertainty_vs_volume.csv'), index=False, float_format='%.4f')
need = {}
for sh in SH:
    v = vtab[vtab.shell == sh]; x, y = np.log(v.panel_kt.values), np.log(v.hw_P50.values)
    b, a = np.polyfit(x, y, 1); need[sh] = dict(slope=b, kt_for_tol=float(np.exp((np.log(TOL) - a) / b)))
ctab = pd.DataFrame(cls_rows); ctab.to_csv(os.path.join(OUT, 'classification.csv'), index=False, float_format='%.4f')

# ---------- figures ----------
def fig_gt():
    f, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for sh in SH:
        g = gtab[gtab.shell == sh]
        for a, k, sc in ((ax[0], 'T', 1), (ax[1], 'G', 1), (ax[2], 'koz', 1)):
            a.fill_between(g.cutoff, g[f'{k}_P10'], g[f'{k}_P90'], color=COL[sh], alpha=0.18, lw=0)
            a.plot(g.cutoff, g[f'{k}_P50'], color=COL[sh], lw=2, marker=MK[sh], ms=4, label=LAB[sh])
            a.plot(g.cutoff, g[f'{k}_OK'], color=COL[sh], lw=1, ls=':')
    ax[0].set_ylabel('Tonnes above cut-off (Mt)'); ax[1].set_ylabel('Grade above cut-off (g/t Au)'); ax[2].set_ylabel('Metal above cut-off (koz Au)')
    for a in ax: a.set_xlabel('Cut-off (g/t Au)')
    ax[0].plot([], [], color='#8a8984', lw=1, ls=':', label='Ordinary kriging (dotted)')
    h, l = ax[0].get_legend_handles_labels(); f.legend(h, l, frameon=False, fontsize=7, loc='outside lower center', ncol=5)
    f.savefig(os.path.join(OUT, 'fig_grade_tonnage.png')); plt.close(f)

def fig_dec():
    f, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True, sharey=True)
    for a, q in zip(ax, ('tonnes', 'grade', 'metal')):
        d = dtab[dtab.quantity == q]; x = np.arange(len(d))
        a.bar(x, d.share_geometry, color='#2a78d6', width=0.8, label='Geometry (shell choice)')
        a.bar(x, d.share_grade, bottom=d.share_geometry, color='#eb6834', width=0.8, label='Grade (SGS)')
        a.set_xticks(x, [f'{c:g}' for c in d.cutoff], fontsize=7); a.set_title(q.capitalize(), fontsize=9); a.set_xlabel('Cut-off (g/t Au)')
        a.grid(axis='x', visible=False)
    ax[0].set_ylabel('Share of total variance')
    h, l = ax[0].get_legend_handles_labels(); f.legend(h, l, frameon=False, fontsize=7, loc='outside lower center', ncol=2)
    f.savefig(os.path.join(OUT, 'fig_variance_decomposition.png')); plt.close(f)

def fig_cls():
    f, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
    cc = {'Measured': '#2a78d6', 'Indicated': '#1baf7a', 'Inferred': '#eda100'}
    x = np.arange(len(SH)); bot = np.zeros(len(SH))
    for cname in ('Measured', 'Indicated', 'Inferred'):
        v = np.array([ctab[(ctab.shell == sh) & (ctab.category == cname)].Mt.iloc[0] for sh in SH])
        ax.bar(x, v, bottom=bot, color=cc[cname], width=0.7, label=cname, edgecolor='white', linewidth=1); bot += v
    ax.set_xticks(x, [LAB[s].replace(' RBF', '\nRBF') for s in SH], fontsize=7); ax.set_ylabel(f'Mt above {COG} g/t Au')
    ax.grid(axis='x', visible=False); ax.legend(frameon=False, fontsize=7)
    f.savefig(os.path.join(OUT, 'fig_classification.png')); plt.close(f)

def fig_map():
    f, ax = plt.subplots(1, 4, figsize=(7.2, 2.3), constrained_layout=True, sharex=True, sharey=True)
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(['#2a78d6', '#1baf7a', '#eda100'])
    for a, sh in zip(ax, SH):
        bid, cat = cls_maps[sh]; i, j, k = np.unravel_index(bid, tuple(NB))
        img = np.full((NB[1], NB[0]), np.nan)
        best = np.full((NB[1], NB[0]), 9)
        np.minimum.at(best, (j, i), cat); img[best < 9] = best[best < 9]   # best category in each column
        a.imshow(img, origin='lower', cmap=cm, vmin=1, vmax=3, interpolation='nearest'); a.set_title(LAB[sh], fontsize=8)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    f.savefig(os.path.join(OUT, 'fig_classification_plan.png')); plt.close(f)

def fig_vol():
    f, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    for sh in SH:
        v = vtab[vtab.shell == sh]
        ax.fill_between(v.panel_kt, 100 * v.hw_P10, 100 * v.hw_P90, color=COL[sh], alpha=0.15, lw=0)
        ax.plot(v.panel_kt, 100 * v.hw_P50, color=COL[sh], lw=2, marker=MK[sh], ms=4, label=LAB[sh])
    ax.axhline(100 * TOL, color='#8a8984', lw=1, ls='--'); ax.text(9, 100 * TOL * 1.04, f'±{100*TOL:.0f}% at {100*CONF:.0f}%', fontsize=7, color='#5f5e5a')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Shell tonnage in panel (kt)'); ax.set_ylabel('90% half-width of metal (%)')
    ax.set_yticks([10, 15, 20, 30, 50, 70], ['10', '15', '20', '30', '50', '70']); ax.legend(frameon=False, fontsize=7, loc='lower left')
    f.savefig(os.path.join(OUT, 'fig_uncertainty_vs_volume.png')); plt.close(f)

for fn in (fig_gt, fig_dec, fig_vol): fn()
summary = dict(density='nearest logged SG (2.45/2.65/2.80)', mean_block_density={sh: round(float((B[f'{sh}_bvol'] * RHOB[sh]).sum() / B[f'{sh}_bvol'].sum()), 3) for sh in SH}, reporting_cutoff=COG, tolerance=TOL, confidence=CONF, panel_quarter_m=[5 * PANEL_Q[0], 5 * PANEL_Q[1], 2.5 * PANEL_Q[2]],
               panel_year_m=[5 * PANEL_A[0], 5 * PANEL_A[1], 2.5 * PANEL_A[2]], realisations=int(R['reference_T'].shape[0]),
               at_cutoff=gtab[gtab.cutoff == COG].round(4).to_dict('records'),
               decomposition_at_cutoff=dtab[dtab.cutoff == COG].round(4).drop(columns='ref_vs_ok').to_dict('records'),
               classification=ctab.round(4).to_dict('records'), tonnage_for_tolerance_kt=need)
json.dump(summary, open(os.path.join(OUT, 'analysis_summary.json'), 'w'), indent=1)
print(gtab[gtab.cutoff == COG][['shell', 'T_P10', 'T_P50', 'T_P90', 'G_P50', 'koz_P10', 'koz_P50', 'koz_P90', 'T_OK', 'G_OK', 'koz_OK']].round(3).to_string(index=False))
print(dtab[dtab.cutoff == COG][['quantity', 'cv_total', 'cv_geometry', 'cv_grade', 'share_geometry']].round(3).to_string(index=False))
print(ctab.round(3).to_string(index=False))
print(vtab.round(3).to_string(index=False)); print(json.dumps(need, indent=1))
