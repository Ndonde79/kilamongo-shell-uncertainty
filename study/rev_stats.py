# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Revision statistics from the existing 4 x 50 realisations: unrounded variance components, all 200 outputs,
scenario-set and weight sensitivity, panel precision with pass fractions, asymmetric deviations, fit uncertainty,
percentile stability. Writes rev/*.csv and rev/rev_stats.json."""
import json, itertools
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
rng = np.random.default_rng(20261118)
SH = ['reference', 'indicator', 'signed_distance', 'contacts']
B = np.load('gt_blocks_v2.npz'); NB, LOB, BSB = B['NB'], B['LO'], B['BS']
DEN = pd.read_csv('comp_density.csv'); DT = cKDTree(DEN[['X', 'Y', 'Z']].to_numpy())
COG, TOL, CONF, OZ = 0.5, 0.15, 0.90, 31.1035
TB, BG = {}, {}
for sh in SH:
    i, j, k = np.unravel_index(B[f'{sh}_bid'], tuple(NB)); xyz = LOB + BSB * (np.c_[i, j, k] + 0.5)
    TB[sh] = B[f'{sh}_bvol'].astype(float) * DEN.rho.to_numpy()[DT.query(xyz)[1]]; BG[sh] = B[f'{sh}_bg'].astype(float)
Q = {}
rows = []
for sh in SH:
    sel = BG[sh] >= COG; T = (TB[sh][:, None] * sel).sum(0); M = (TB[sh][:, None] * BG[sh] * sel).sum(0)
    Q[sh] = dict(tonnes=T, metal=M, grade=M / T)
    for r in range(len(T)): rows.append(dict(shell=sh, realisation=r + 1, tonnes_t=T[r], metal_oz=M[r] / OZ, grade_gpt=M[r] / T[r]))
pd.DataFrame(rows).to_csv('rev/all_200_outputs_0.5gpt.csv', index=False, float_format='%.6g')
def decomp(shells, w, q):
    X = np.stack([Q[s][q] for s in shells]); w = np.asarray(w, float); w = w / w.sum()
    m = X.mean(1); v = X.var(1, ddof=1); mu = (w * m).sum()
    between = (w * (m - mu) ** 2).sum(); within = (w * v).sum()
    return between, within, between / (between + within)
out = {}
comp = []
for q in ('tonnes', 'metal', 'grade'):
    b, wv, s = decomp(SH, [1] * 4, q)
    unit = {'tonnes': 't^2', 'metal': 'g^2', 'grade': '(g/t)^2'}[q]
    comp.append(dict(quantity=q, between_shell=b, within_shell=wv, total=b + wv, share_between=s, unit=unit))
pd.DataFrame(comp).to_csv('rev/variance_components_0.5gpt.csv', index=False, float_format='%.6g'); out['components'] = comp
sens = []
for kset in [4, 3, 2]:
    for subset in itertools.combinations(SH, kset):
        for q in ('tonnes', 'metal'):
            sens.append(dict(set='+'.join(subset), n=kset, quantity=q, share_between=decomp(subset, [1] * kset, q)[2]))
for q in ('tonnes', 'metal'):
    for sh in SH:
        for wt in (0.1, 0.4, 0.7):
            w = [wt if s == sh else (1 - wt) / 3 for s in SH]
            sens.append(dict(set=f'weight {sh}={wt}', n=4, quantity=q, share_between=decomp(SH, w, q)[2]))
    d = np.array([decomp(SH, rng.dirichlet(np.ones(4)), q)[2] for _ in range(5000)])
    sens.append(dict(set='Dirichlet(1,1,1,1) weights: min', n=4, quantity=q, share_between=d.min()))
    sens.append(dict(set='Dirichlet(1,1,1,1) weights: P5', n=4, quantity=q, share_between=np.percentile(d, 5)))
    sens.append(dict(set='Dirichlet(1,1,1,1) weights: median', n=4, quantity=q, share_between=np.median(d)))
st = pd.DataFrame(sens); st.to_csv('rev/scenario_weight_sensitivity.csv', index=False, float_format='%.4f')
gq = {}
for sh in SH:
    for q in ('tonnes', 'metal', 'grade'):
        x = Q[sh][q]; p = np.percentile(x, [5, 10, 50, 90, 95])
        gq[f'{sh}_{q}'] = dict(P5=p[0], P10=p[1], P50=p[2], P90=p[3], P95=p[4], lo10=p[1] / p[2] - 1, hi90=p[3] / p[2] - 1, lo5=p[0] / p[2] - 1, hi95=p[4] / p[2] - 1)
out['global_quantiles'] = gq
SIZES = [20, 30, 40, 60, 80, 120, 160, 240, 320, 480]
vr, pan80 = [], {}
for sh in SH:
    i, j, k = np.unravel_index(B[f'{sh}_bid'], tuple(NB)); bg = BG[sh]; tb = TB[sh]; sel = (bg >= COG) * tb[:, None]
    for L in SIZES:
        H = 4 if L < 320 else 64
        p = (L // 5, L // 5, H); pid = np.ravel_multi_index((i // p[0], j // p[1], k // p[2]), tuple(-(-NB // np.array(p))))
        up, inv = np.unique(pid, return_inverse=True)
        metal = np.stack([np.bincount(inv, weights=(sel * bg)[:, r], minlength=len(up)) for r in range(bg.shape[1])], 1)
        ore = np.stack([np.bincount(inv, weights=sel[:, r], minlength=len(up)) for r in range(bg.shape[1])], 1)
        ton = np.bincount(inv, weights=tb, minlength=len(up))
        lo, md, hi = np.percentile(metal, [5, 50, 95], axis=1)
        ok = md > 0
        hw = np.where(ok, (hi - lo) / (2 * np.where(ok, metal.mean(1), 1)), np.inf)
        up_dev = np.where(ok, hi / np.where(ok, md, 1) - 1, np.inf); lo_dev = np.where(ok, 1 - lo / np.where(ok, md, 1), np.inf)
        both = (up_dev <= TOL) & (lo_dev <= TOL)
        for rule, f in (('all', ton > 0), ('ge_P25', ton >= np.percentile(ton, 25)), ('ge_median', ton >= np.median(ton)), ('ge_P75', ton >= np.percentile(ton, 75))):
            if f.sum() == 0: continue
            vr.append(dict(shell=sh, side_m=L, height_m=2.5 * H, rule=rule, n_panels=int(f.sum()), panel_kt_median=np.median(ton[f]) / 1e3,
                           panel_kt_max=ton[f].max() / 1e3, ore_kt_median=np.median(np.median(ore, 1)[f]) / 1e3,
                           hw50=np.median(hw[f]), up50=np.median(up_dev[f]), lo50=np.median(lo_dev[f]),
                           frac_hw_pass=float((hw[f] <= TOL).mean()), frac_both_pass=float(both[f].mean()),
                           tonnage_share_both_pass=float(ton[f][both[f]].sum() / ton[f].sum())))
        if L == 80:
            f = ton >= np.median(ton); m80 = metal[f]
            bs = []
            for _ in range(500):
                rr = rng.integers(0, m80.shape[1], m80.shape[1]); mb = m80[:, rr]
                l, h = np.percentile(mb, [5, 95], axis=1); bs.append(np.median((h - l) / (2 * np.maximum(mb.mean(1), 1e-9))))
            pan80[sh] = dict(hw50=float(np.median(hw[f])), boot_P5=float(np.percentile(bs, 5)), boot_P95=float(np.percentile(bs, 95)))
V = pd.DataFrame(vr); V.to_csv('rev/panel_precision.csv', index=False, float_format='%.4f')
out['panel80_percentile_stability'] = pan80
fits = {}
for sh in SH:
    for rule in ('ge_P25', 'ge_median', 'ge_P75'):
        v = V[(V.shell == sh) & (V.rule == rule) & (V.height_m == 10)]
        x, y = np.log(v.panel_kt_median.values), np.log(v.hw50.values)
        (b, a), cov = np.polyfit(x, y, 1, cov=True)
        T = np.exp((np.log(TOL) - a) / b)
        draws = rng.multivariate_normal([b, a], cov, 4000); Td = np.exp((np.log(TOL) - draws[:, 1]) / draws[:, 0])
        Td = Td[(draws[:, 0] < 0)]
        fits[f'{sh}_{rule}'] = dict(slope=b, intercept=a, slope_se=float(np.sqrt(cov[0, 0])), T15_kt=T, T15_P5=float(np.percentile(Td, 5)), T15_P95=float(np.percentile(Td, 95)),
                                    fit_range_kt=[float(v.panel_kt_median.min()), float(v.panel_kt_median.max())], hw_at_max=float(v.hw50.values[-1]),
                                    extrapolated=bool(T > v.panel_kt_median.max()))
out['fits'] = fits
json.dump(out, open('rev/rev_stats.json', 'w'), indent=1, default=float)
