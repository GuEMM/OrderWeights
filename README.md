# What measures of high-order interdependence count

Code to reproduce the figures and tables of

> G. Menesse and H. J. Jensen, *What measures of high-order interdependence count: order weights, local geometry and sampling bias* (2026).

Every labeling-symmetric Shannon measure of interdependence (total correlation, dual total correlation, O- and S-information, TSE complexity, co-information, …) is a weighted sum of the co-informations of all subsets of variables, with one weight per subset size. These *order weights* fix what a measure counts, how it responds to weak interdependence, and how finite samples bias its estimates. This repository contains the code behind every computed figure and table of the paper.

## Contents

```
orderweights/        Python package with the shared functions
  measures.py        coefficients c_k of the measures and their order weights lambda_m
  bias.py            exact and simulated plug-in bias, discrete variables
  gaussian.py        exact and simulated plug-in bias, Gaussian variables
  slice.py           the pairwise-triplet slice of the simplex for N = 3
  paths.py           linear-mixing and maximum-entropy paths from the uniform distribution (COPY, two copies, common field)
  fourier.py         Fourier coefficients of Boolean distributions
  dyadic.py          dyadic and triadic distributions of James & Crutchfield
  cache.py           storage of the slow simulations in results/
notebooks/
  reproduce_figures_and_tables.ipynb   one section per figure and table
results/             stored Monte Carlo simulations (read by the notebook)
figures/             figures written by the notebook (PNG and PDF)
tables/              tables written by the notebook (LaTeX and CSV)
tests/               quick checks of the closed forms used in the paper
```

## Figures and tables

| Paper | Notebook section | Output |
|---|---|---|
| Fig. 2, order weights | 1 | `figures/Specturm_some_meas` |
| Fig. 3b,c, slice for q = 2 | 3 | `figures/Full_kite` (b), `figures/Zoom_in` (c) |
| Fig. 4, mixing vs maximum entropy (independent regimes, two copies, COPY) | 4 | `figures/fig_mixing_maxent` |
| Fig. 5, plug-in bias, discrete (a,b) and Gaussian (c,d) | 5, 6 | `figures/bias_main` |
| Methods, Fourier components of Boolean rules | 2 | `figures/fig_fourier_rules` |
| SI Fig., plug-in bias, all measures | 5 | `figures/bias_violins_SI` |
| SI Fig., fewer samples than states | 5 | `figures/bias_saturation_SI` |
| SI Fig., Gaussian bias, all measures | 6 | `figures/gaussian_bias_SI` |
| SI Table S1, Fourier characters | 2 | `tables/Tab_1_characters` |
| SI Tables S-beta, S-bias | 5 | `tables/tab_S-beta`, `tables/tab_S-bias` |
| SI Tables S-marginals, S-autocorr | 5 | `tables/tab_S-marginals`, `tables/tab_S-autocorr` |
| SI Table S-gauss and correlated check | 6 | `tables/tab_S-gauss` |
| SI Tables S-dt, S-dt-family | 7 | `tables/tab_S-dt`, `tables/tab_S-dt-family` |

Fig. 1 and Table 1 contain no computed content. Panel (a) of Fig. 3 is a schematic, so the notebook produces the computed panels only.

The notebook also checks, in code, the closed forms and the numbers quoted in the text: the order weights of Eq. (spectra), the two forms of the exact bias, the variations along the paths, the sign reversal of the bias below about $q^{N/2}$ samples, and others.

## How to run

Requires Python 3.10 or later.

```bash
git clone <this repository>
cd order-weights
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
jupyter lab notebooks/reproduce_figures_and_tables.ipynb
```

Or with conda: `conda env create -f environment.yml && conda activate order-weights`.

With `RECOMPUTE = False` (the default, first code cell) the notebook reads the stored simulations in `results/` and runs in about a minute. With `RECOMPUTE = True` it runs every simulation again with the seeds that produced the published numbers; this takes about 5 to 10 minutes on a laptop, mostly for SI Tables S-marginals and S-autocorr. Deleting a file in `results/` recomputes only that simulation.

The simulations use NumPy's `default_rng` with fixed seeds, and their algorithms consume the random stream in the same order as the runs reported in the paper. With NumPy 2.x they reproduce the published tables to the last digit shown.

The package can also be used on its own:

```python
from orderweights import measures as M, bias as B
lam = M.order_weights(M.c_measure("Omega", 6))      # [0, 0, -1, -2, -3, -4]
B.exact_bias(M.c_measure("Omega", 6), N=6, q=2, T=64)
```

To run the quick checks: `python -m pytest tests` (or `python tests/test_closed_forms.py`). They include the closed forms of the design section: the bias bound and its attainment by $\Omega^*$, the cheapest measure (the O-information, $2N+1$ entropies) and the ratio of Eq. (omega-ratio).

## Citation

If you use this code, please cite the paper above.
