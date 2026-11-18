# kilamongo-shell-uncertainty

Data, code and results for the paper *Effects of Ore-Shell Interpretation on Simulated Gold Inventory and Production-Panel Uncertainty at Kilamongo South, Tanzania* (R.W. Ndonde, submitted to *Minerals*).

The study builds three implicit ore shells with DeepRBF (indicator, signed-distance and contact formulations) from the same drill data, and keeps a fourth shell, from commercial implicit-modelling software, as a benchmark. Grade is simulated by sequential Gaussian simulation inside each shell and in the union of the shells. The between-shell and within-shell spread of tonnes, grade and metal is compared, and the precision of simulated metal is measured against aggregation tonnage.

## Contents

| Folder | What it holds |
|---|---|
| `data/` | The final Kilamongo South drill-hole data set: 417 collars (408 with gold assays), surveys, gold assays, lithology and weathering logs, class densities, and the list of grade-control holes. |
| `shells/` | DeepRBF shell configurations (indicator, signed distance, contacts) and the benchmark shell wireframe (`Au_Ind_kila_0.3.dtm/.str`). |
| `study/` | Scripts, derived inputs (simulation grid with shell membership, reporting envelope, composites with shell membership, variogram models) and results. |
| `study/out_v2/` | Grade–tonnage, variance decomposition and panel-precision tables from the production run. |
| `study/rev/` | Revision results: all 200 shell–realisation outputs, variance components, scenario and weight sensitivity, panel precision, common grade model, discretisation test, simulation checks and the signed-distance run history. |

Gold grades are in g/t Au; coordinates are UTM zone 36S (m). Density values are averages of measured specific gravity by weathering class (I highly weathered 2.45, M moderately weathered 2.65, F fresh 2.80 t/m³); the individual measurements are not preserved.

## Software

DeepRBF 3.4.2 (https://github.com/Ndonde79/DeepRBF, tag v3.4.2, commit c8e01fa44b; doi:10.5281/zenodo.24246601), Python 3.11, NumPy, SciPy, pandas and Matplotlib.

## Run order

1. `shells/RUN_SHELLS_V2.bat` builds the three DeepRBF shells (`python -m deeprbf.workflow.shell <config>.json`).
2. In `study/`: `build_masks.py` (2 m shell masks), `prep2.py` (simulation grid and shell membership), `orient.py` (zone geometry), `vario4.py` (variogram), `envelope.py` (reporting envelope).
3. `RUN_GT_V2.bat` runs `gt_sim_zone2.py`: 50 SGS realisations and ordinary kriging in each shell (about 30 minutes on a workstation).
4. `analyse_v7.py` and `figs_sim_v4.py`: grade–tonnage, variance decomposition, panel precision and figures. `boot_mean.py`: bootstrap of the declustered means.
5. Revision analyses: `rev_stats.py` (variance components, sensitivity, panel precision), `rev_sim.py` (common grade model with `union 2 50`; discretisation with `signed_distance 1|2|3`), `rev_post.py`, `rev_check.py`, `rev_check2.py` (node-level checks for each shell and the union run; run `rev_sim.py <shell> 2 50` first), `rev_panel2.py` (two-sided panel criterion), `figs_rev.py`, `fig_vol_rev.py`, `fig_var_rev.py`.

`gt_blocks_v2.npz` holds the block-level realisations of the production run, so the analyses in steps 4 and 5 can be repeated without rerunning the simulation. The node-level realisations of the revision runs are not stored; `rev_sim.py` regenerates them with the same seed.

## Licence and citation

Code: BSD 3-Clause (see `LICENSE`). Data and results: CC BY 4.0. Please cite the paper and this archive (see `CITATION.cff`).
