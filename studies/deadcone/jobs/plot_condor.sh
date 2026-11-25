#!/bin/bash
# Flavour comparison plots (xi_p spectra) from the histograms of jobs/submit_condor.sh.
# Usage (from studies/deadcone): bash jobs/plot_condor.sh <output tag>
# Output: output/<output tag>/results.
tag=$1
python plot_results.py -i output/$tag/main/hists.pkl -o output/$tag/results
