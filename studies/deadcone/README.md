# Dead cone study

Search for the dead cone effect in ALEPH b-quark jets at the Z pole.
The dead cone is the suppression of gluon radiation from a heavy quark within an angle θ₀ ≈ m_Q/E around its direction.
Two complementary approaches are used:

1. **Momentum space**, as in Kluth, Ochs, Pérez-Ramos, [PRD 107 (2023) 094039](https://arxiv.org/abs/2303.13343):
   - Compare the charged-particle spectrum in ξ_p = ln(1/x_p), with x_p = 2p/√s, between b and uds jets.
   - The dead cone suppresses high-momentum (small-ξ_p) particles in heavy-quark jets.
   - Heavy-hadron decay products are partly removed by rejecting displaced tracks.
   - Reference: OPAL, [Eur. Phys. J. C7 (1999) 369](https://arxiv.org/abs/hep-ex/9807004), Table 5.
2. **Iterative declustering**, as in ALICE, [Nature 605 (2022) 440](https://arxiv.org/abs/2106.05713):
   - The charged constituents of each jet are reclustered with the Cambridge/Aachen algorithm and then declustered.
   - At each step the declustering follows the branch containing the heavy-hadron proxy (the sum of the displaced tracks), or else the harder branch.
   - Each step is a splitting of a "radiator" with energy E_rad = E₁ + E₂ (charged energy only).
   - The quantity studied is the ratio R(θ) = b/uds of the splitting-angle distributions in bins of E_rad.
   - For a dead cone, R should drop below 1 for θ ≲ m_b/E_rad. Curves for different E_rad should coincide as a function of θ/θ₀ = θ·E_rad/m_b.

Results and their interpretation are in [NOTES.md](NOTES.md).

## Analysis in short

- **Events:** hadronic events (`recoEventType == 16`) with ≥ 2 jets and y₂₃ = `Event_dmerge2`/s < 0.13 (`selections/selection.json`).
  Both jets must pass the quality selection: p > 10 GeV and |cos θ| < 0.65 (`selections/selection_jets.json`).
  The jet selection is applied before the event selection, so events where a jet fails are rejected.
- **Particles:** charged jet constituents.
  A *displaced* track is charged with Sip2dSig > 3 and |JetDistVal| < 0.07 (setting `displaced`).
- **Jet tags**, applied to the other jet of the two-jet event (hemisphere-opposite tagging):
  - b-tag: ≥ 3 displaced tracks;
  - light tag: no displaced tracks;
  - the expressions are settings (`btag`, `lighttag`), so they can be replaced by classifier scores.
- **Flavour spectra:**
  - In simulation, the true event flavour gives the uds, c and b spectra.
  - In data, the spectra of jets opposite to a tagged jet are corrected bin by bin to pure flavours with simulation:
    `D_b(data) = D_btag(data) · D_b(sim) / D_btag(sim)`, and similarly for uds with the light tag.
- **Normalization:**
  - ξ_p spectra: per jet. They are shown per event (×2) for the comparison with OPAL.
  - Declustering spectra: per radiator, i.e. per jet with at least one splitting in the E_rad bin.
- **Not used:** the V0/K⁰_S/Λ collections.
- **Level of the results:** all are at detector level; there is no unfolding.

## Structure

- `run.py`: makes the histograms (and default data vs. simulation plots) with `make_histograms` and `plot_hists_default` from `analysis/plot.py`.
  - It takes the same arguments as `analysis/plot.py` (selections, merging, splitting, `--external_variables`, ...).
  - Extra arguments:
    - `--derived`: groups of derived variables to compute;
    - `--settings`: settings file for the derived variables;
    - `--maxfiles_sim`, `--maxfiles_data`: limits for quick tests;
    - `--noplots`.
  - Output: `<outputdir>/hists.pkl`, with the settings and variables used.
- `plot_results.py`: flavour spectra, ratios and comparisons to references, from one or more `hists.pkl`.
  The first input is the reference, used for all plots. The others are only compared to it in `lund_ratios.png`.
  The script header describes each plot.
- `lib/derived.py`: derived variables, organized in groups:
  - `angle`: angle of each constituent to the jet axis;
  - `xi`: x_p and ξ_p;
  - `tags`: displaced tracks and jet tags;
  - `lund`: the declustering.
- `lib/lund.py`: vectorized Cambridge/Aachen reclustering and iterative declustering (numpy only).
- `lib/variables.py`: histogram variable definitions, including the E_rad bins and splitting cuts.
  Run it to regenerate `configs/variables_*.json`.
- `configs/`: settings for the derived variables and the variable files:
  - `settings_default.json`: the reference settings;
  - `settings_nomerge.json`: no heavy-hadron proxy;
  - `settings_tight.json`: displaced tracks with Sip2dSig > 5.
- `selections/`: event and object selections.
- `references/opal1999_xi.json`: OPAL ξ_p spectra for uds and b events (hadron level, per event).
- `jobs/`: Slurm scripts for the full-statistics runs and plots.
- `output/` (not in git): histograms, plots and logs.

## Reproducing the results

The environment is the conda environment `aleph_analysis_env`; the job scripts activate it.
All commands are run from `studies/deadcone`.

```bash
cd studies/deadcone
python lib/variables.py      # (re)generate configs/variables_{xi,lund,lund_variations}.json
mkdir -p output/logs         # Slurm log directory (must exist before submitting)

# histograms on all simulation (70 files) and data (189 files), about 20 minutes each
j1=$(sbatch --parsable jobs/run_full.sh main configs/settings_default.json \
       configs/variables_xi.json configs/variables_lund.json configs/variables_lund_variations.json configs/variables_angle.json)
j2=$(sbatch --parsable jobs/run_full.sh nomerge configs/settings_nomerge.json \
       configs/variables_lund.json configs/variables_lund_variations.json)
j3=$(sbatch --parsable jobs/run_full.sh tight configs/settings_tight.json \
       configs/variables_xi.json configs/variables_lund.json configs/variables_lund_variations.json)

# flavour comparison plots, once all histograms are done
sbatch --dependency=afterok:$j1:$j2:$j3 jobs/plot_full.sh
```

`jobs/run_full.sh <name> ...` writes `output/<name>/hists.pkl` and default data vs. simulation plots in `output/<name>/plots/`.
`jobs/plot_full.sh` writes:
- `output/results/`: the reference, with `nomerge` overlaid in `lund_ratios.png`;
- `output/results_displaced/`: the `tight` settings, with the reference overlaid.

Quick test on a few files, without Slurm:
```bash
D=/HEP/data/share/aleph/ntuples-withksloose/eventlevel
python run.py -s $D/mc -d $D/data --maxfiles_sim 2 --maxfiles_data 1 \
  -v configs/variables_xi.json configs/variables_lund.json --derived xi tags lund \
  --objectselection selections/selection_jets.json --eventselection selections/selection.json \
  --merge ../../analysis/merging/merging.json --split ../../analysis/merging/splitting.json \
  -o output/test --noplots
python plot_results.py -i output/test/hists.pkl -o output/test/results
```

## Classifier scores as jet tags

Pass `--external_variables <dir>` to `run.py` (add it in `jobs/run_full.sh`).
Then set `btag` and `lighttag` in a settings file to expressions on the external branches,
e.g. `"btag": "score_recojet_isB > 0.9"`.
