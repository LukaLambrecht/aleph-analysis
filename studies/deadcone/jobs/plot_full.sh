#!/bin/bash
# Flavour comparison plots from the full-statistics histograms (see jobs/run_full.sh):
#   output/results:           reference (main), compared to no merging of the displaced tracks;
#   output/results_displaced: tighter displaced-track definition (tight), compared to the reference.
# Usage: sbatch --dependency=afterok:<jobids> jobs/plot_full.sh
#SBATCH --job-name=deadcone-plot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=output/logs/%x_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate aleph_analysis_env
python plot_results.py -i output/main/hists.pkl output/nomerge/hists.pkl \
  -l "proxy merged" "no merging" -o output/results
python plot_results.py -i output/tight/hists.pkl output/main/hists.pkl \
  -l "Sip2dSig > 5" "Sip2dSig > 3" -o output/results_displaced
