#!/bin/bash
# Full-statistics histogramming for the dead cone study as a condor job (lxplus).
# Baseline xi_p analysis only: reference settings, xi_p variables (no declustering),
# including the spectra of the tracks from B decays (truth flag) for the simulation template subtraction
# (the ntuples must have the truth branches, see bdecay_truth in the settings).
# Usage (from studies/deadcone): bash jobs/submit_condor.sh <sample list json> <output tag>
# Output: output/<output tag>/main/hists.pkl.
# Make the flavour comparison plots once the job is done with jobs/plot_condor.sh <output tag>.

samples=$1; tag=$2
python3 run.py --samples $samples -o output/$tag/main --settings configs/settings_default.json \
  -v configs/variables_xi.json configs/variables_xi_bdecay.json --derived xi tags bdecay \
  --objectselection selections/selection_jets.json --eventselection selections/selection.json \
  --merge ../../analysis/merging/merging.json --split ../../analysis/merging/splitting.json \
  --files_per_batch 5 --normalizesim --dolog --runmode condor
