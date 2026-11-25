#!/bin/bash
# Full-statistics histogramming for the dead cone study (all simulation and data files).
# Usage: sbatch jobs/run_full.sh <name> <settings json> <variable jsons ...>
# Output: output/<name>/hists.pkl and default data vs. simulation plots in output/<name>/plots.
#SBATCH --job-name=deadcone
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=output/logs/%x_%j.out

name=$1; settings=$2; shift 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aleph_analysis_env
echo "host: $(hostname), start: $(date)"
D=/HEP/data/share/aleph/ntuples-withksloose/eventlevel
python -u run.py -s $D/mc -d $D/data -v "$@" -o output/$name \
  --derived angle xi tags lund --settings $settings \
  --objectselection selections/selection_jets.json --eventselection selections/selection.json \
  --merge ../../analysis/merging/merging.json --split ../../analysis/merging/splitting.json \
  --files_per_batch 5 --normalizesim --dolog
echo "end: $(date)"
