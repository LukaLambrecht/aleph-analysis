# Hemisphere-hemisphere correlation uncertainty study

Data/MC study of the tagging correlation between the two jets ("hemispheres") of an
event, used as an input to assessing a systematic uncertainty on the correlation
parameters (rho) of the Rb double-tag method in `rbanalysis/analysis/rbcalc.py`.

Method follows:
- D. Buskulic et al. (ALEPH Collab.), Phys. Lett. B 401 (1997) 150, Section 4.2.2.
- D. Buskulic et al. (ALEPH Collab.), Phys. Lett. B 401 (1997) 163, Section 6.3.

## Method summary

The double-tag method assumes the two jets of an event tag independently. The
correlation parameter rho quantifies the (flavour- and tag-pair-dependent) deviation
from that assumption. `rbcalc.py` already computes one aggregate rho value per
(tag pair, truth flavour), with its MC-statistical uncertainty (via bootstrap).

This is a *different*, complementary uncertainty: whether simulation correctly
predicts *how* rho arises, by checking whether it can be traced to specific physical
sources (gluon radiation reducing both jet momenta together, detector acceptance
depending on the shared event topology, etc). This is done by making rho differential
in a variable sensitive to each source, and comparing data to MC.

For a variable `v` and tag `T`, two curves are built:
- `eps_own(v)`: probability that a jet is tagged `T`, as a function of *its own* `v`.
- `eps_oppo(v)`: probability that the *opposite* jet is tagged `T`, as a function of
  *this* jet's `v`. If jets tagged independently, this would be flat (no dependence
  on the other jet's v); structure in it is the correlation signal.

For a tag pair (I, J), the differential correlation contribution is:
```
rho(v) contribution = f(v) * [eps_I_own(v)*eps_J_oppo(v) + eps_J_own(v)*eps_I_oppo(v)]
                       ---------------------------------------------------------------
                                        2 * mean(eps_I) * mean(eps_J)
```
summed over bins, minus 1 (generalizing Eq. (4) of the 1997 lifetime-mass paper to a
pair of possibly different tags, per Section 6.3 of the mutually-exclusive-tags paper).

**Sanity check**: with the finest possible binning (one hemisphere per bin), this
expression is mathematically identical to the aggregate, non-differential correlation
already computed in `rbcalc.py`'s `calculate_efficiencies`. This was verified
numerically (agreement to ~0.002 using bin edges at the unique data values, on a
small test sample) — rerun it by binning the variable with edges at every unique
value and comparing the resulting `rho` to `calculate_efficiencies(...)['correlations']`
for the same events.

## What's implemented

- `correlation_uncertainty.py`: for every (tag pair, truth flavour, variable)
  combination — **all 45 correlations**, not just the ones historically found to
  have large impact on Rb — computes:
  - the MC-truth curves and aggregate rho (straightforward, truth flavour known).
  - a data-driven estimate, using a loose ("soft") NN-score cut on *both* jets
    (`tools/thresholds_soft.json`) to isolate a flavour-enriched data sample,
    corrected for residual background using two independent methods (following
    Section 4.2.2 of Phys. Lett. B 401 (1997) 150): an MC-based **purity
    reweighting** (bin-by-bin, as a function of the same variable), and a
    **bin-by-bin background subtraction** normalized to data using the same
    overall data/MC scale-factor convention as `calibration/calibration.py`'s
    tag-and-probe method (`norm_factor = N_data / N_sim`).
- `plot_efficiencies.py`: diagnostic plots from the JSON output — the
  own/opposite efficiency curves per tag (drawn as `stairs`, i.e. step functions over
  the actual bin edges, with a semi-transparent band showing a rough binomial
  statistical uncertainty; most informative check, also produced as a
  `curves_mc_<variable>.png` simulation-only version for a simpler, step-by-step
  first look before comparing to data), and a bar-chart summary of the aggregate rho
  per tag pair, MC vs. data (also with a `rho_summary_mc_<variable>.png`
  simulation-only version).
- Also from `correlation_uncertainty.py` directly (needs the raw events, not just the
  JSON, so these live there rather than in the post-hoc plotting script):
  - `distribution_<variable>.png`: simulation-only distributions of each variable
    (after the same baseline selection as the rest of the analysis), with the
    resolved analysis bin edges overlaid — used to judge the binning (see below).
  - `controlplot_<variable>.png`: standard stacked-simulation-vs-data plot, one panel
    per soft tag (light/c/b-enriched) side by side, with simulation scaled by the
    exact data/MC normalization factor the background-subtraction method uses —
    lets you check, per soft tag, whether the resulting sample has adequate
    statistics and purity (i.e. whether the target flavour actually dominates
    the stack).
- Bin edges are **quantile-based** (equal population per bin), resolved at runtime
  from the pooled (all truth flavours) simulated sample, rather than fixed
  equal-width bins — several of these variables (jet momentum, strongly peaked near
  the kinematic edge; y3, steeply falling) are poorly suited to equal-width binning.
- Four variables, chosen to trace the classic correlation sources: jet momentum,
  y3 (`Event_dmerge2 / 91.2^2`, the 2->3 jet transition), and `cos(theta)`/`phi` of
  the jet direction as a stand-in for the thrust axis (there is no explicit thrust
  axis in these ntuples, but this is a good approximation since events are required
  to have exactly 2, necessarily back-to-back, jets).
- Reuses `rbanalysis.analysis.rbcalc.load_and_tag_events` (event loading + tag
  assignment, factored out of `rbcalc.py` without changing its own output) and
  `calibration.calibration.build_tag_selection_expression` (for the soft-tag cuts).

## Deliberately not yet implemented (per current scope)

- No combination of the per-variable, per-correlation results into a single
  systematic uncertainty number yet (no impact-weighting, no linear/quadrature
  combination). The individual differential results are the object of interest for
  now; combination is expected to happen downstream (see below).
- No "impact" weighting: the fit that extracts Rb from these efficiencies/
  correlations lives outside this repository, so "impact" (dRb/drho) isn't computed
  here. The natural path is to feed the resulting uncertainties into that fit as
  constrained nuisance parameters (in the same `corr_<tag1>_<tag2>_<flavour>` naming
  already used in `rbcalc.py`'s `params.json`/`systematics_mcstats.json`) and let it
  report impacts automatically, rather than hand-computing impact tables.
- Purity weights are computed per-hemisphere but then averaged into one weight per
  *event* before histogramming (simplification); could be refined to weight each
  hemisphere assignment independently if this turns out to matter.
- Soft-tag thresholds (`tools/thresholds_soft.json`) are initial reasonable guesses
  (`Jets_score_isB > 0.5`, `Jets_score_isC > 0.4`, `Jets_score_isUDG > 0.5`, applied
  to both jets), not yet tuned for optimal purity/statistics trade-off.

## Usage

```
cd rbanalysis/correlation-uncertainties
python correlation_uncertainty.py \
  -s <sim files/globs> -d <data files/globs> \
  -t ../analysis/thresholds/thresholds_20260515.json \
  --objectselection ../../analysis/selections/selection_jets.json \
  --eventselection ../../analysis/selections/selection.json \
  --external_variables <path to NN score pkl directory> \
  --year 1994 --xsections ../../analysis/cross-sections/cross_sections.json \
  --merge ../analysis/merging/merging.json --split ../analysis/merging/splitting.json \
  -o output

python plot_efficiencies.py -i output/correlation_curves.json -o output
```

To submit the same command as a condor job instead of running it locally, just add
`--runmode condor`. The script re-issues the exact same command line it was called
with (only switching `--runmode` back to `local`) and submits that as a single condor
job (job flavour `workday`), sourcing the repo's `setup.sh` inside the job before
running.
```
