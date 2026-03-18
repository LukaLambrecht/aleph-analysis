# Script to keep track of default commands to run

# Full model (for b and c)
python calculate_purities.py \
-s /eos/user/l/llambrec/aleph-data/ntuples-withnewks/eventlevel/mc/output_qqb_*.root \
-v variables/variables_scores.json \
-o output_20260305_withnewks_standardized \
--objectselection ../analysis/selections/selection_jets.json \
--eventselection ../analysis/selections/selection.json \
--external_variables /eos/user/l/llambrec/aleph-data/model_output_scores/output_scores_model_20260305_withnewks_standardized/ \
--merge ../analysis/merging/merging.json \
--split ../analysis/merging/splitting.json

# Model with masked dedx (for s)
python calculate_purities.py \
-s /eos/user/l/llambrec/aleph-data/ntuples-withnewks/eventlevel/mc/output_qqb_*.root \
-v variables/variables_scores.json \
-o output_20260305_withnewks_withdedx_masked_standardized \
--objectselection ../analysis/selections/selection_jets.json \
--eventselection ../analysis/selections/selection.json \
--external_variables /eos/user/l/llambrec/aleph-data/model_output_scores/output_scores_model_20260305_withnewks_withdedx_masked_standardized/ \
--merge ../analysis/merging/merging.json \
--split ../analysis/merging/splitting.json

# Print tables
python print_purities.py output_20260305_withnewks_standardized/purity_baseline_score_is?.csv
python print_purities.py output_20260305_withnewks_withdedx_masked_standardized/purity_baseline_score_is?.csv
