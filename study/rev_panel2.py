# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""Panel precision, second revision: pass counts for the two-sided criterion (d+ <= 0.15 and d- <= 0.15) and power-law
fits of both h and max(d+, d-) against panel tonnage (10 m panels at or above the median panel tonnage)."""
import json
import numpy as np, pandas as pd
exec(open('rev_stats.py').read().split('Q = {}')[0])
rng = np.random.default_rng(7)
SIZES = [20, 30, 40, 60, 80, 120, 160, 240]; rows = []; fits = {}
for sh in SH:
    i, j, k = np.unravel_index(B[f'{sh}_bid'], tuple(NB)); bg = BG[sh]; tb = TB[sh]; sel = (bg >= COG) * tb[:, None]
    agg = []
    for L in SIZES:
        p = (L // 5, L // 5, 4); pid = np.ravel_multi_index((i // p[0], j // p[1], k // p[2]), tuple(-(-NB // np.array(p))))
        up, inv = np.unique(pid, return_inverse=True)
        metal = np.stack([np.bincount(inv, weights=(sel * bg)[:, r], minlength=len(up)) for r in range(bg.shape[1])], 1)
        ton = np.bincount(inv, weights=tb, minlength=len(up))
        lo, md, hi = np.percentile(metal, [5, 50, 95], axis=1); ok = md > 0
        h = np.where(ok, (hi - lo) / (2 * np.where(ok, metal.mean(1), 1)), np.inf)
        dp = np.where(ok, hi / np.where(ok, md, 1) - 1, np.inf); dm = np.where(ok, 1 - lo / np.where(ok, md, 1), np.inf)
        dmax = np.maximum(dp, dm); both = dmax <= 0.15; f = ton >= np.median(ton)
        rows.append(dict(shell=sh, side_m=L, n_all=int(len(up)), pass_all=int(both.sum()), n_half=int(f.sum()), pass_half=int(both[f].sum()),
                         kt_med=np.median(ton[f]) / 1e3, h50=np.median(h[f]), dmax50=np.median(dmax[f]), dmax_min=float(dmax[f].min())))
        agg.append((np.median(ton[f]) / 1e3, np.median(h[f]), np.median(dmax[f])))
    a = np.array(agg)
    for name, col in (('h', 1), ('dmax', 2)):
        (b, c), cov = np.polyfit(np.log(a[:, 0]), np.log(a[:, col]), 1, cov=True)
        T = float(np.exp((np.log(0.15) - c) / b)); d = rng.multivariate_normal([b, c], cov, 4000); d = d[d[:, 0] < 0]
        Td = np.exp((np.log(0.15) - d[:, 1]) / d[:, 0])
        fits[f'{sh}_{name}'] = dict(slope=float(b), intercept=float(c), T_kt=T, P5=float(np.percentile(Td, 5)), P95=float(np.percentile(Td, 95)), max_fit_kt=float(a[:, 0].max()))
R = pd.DataFrame(rows); R.to_csv('rev/panel_pass_counts.csv', index=False, float_format='%.4f'); json.dump(fits, open('rev/panel_fits_two_sided.json', 'w'), indent=1)
pd.set_option('display.width', 200); print(R.round(3).to_string())
for k, v in fits.items(): print(k, {a: round(b, 3) for a, b in v.items()})
