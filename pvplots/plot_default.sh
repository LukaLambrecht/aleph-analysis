# Script to keep track of default plotting command

python plot.py \
-s /eos/user/l/llambrec/aleph-data/ntuples-withksloose/eventlevel/mc/output_qqb_*.root \
-v variables/variables_pv_residuals.json \
-o output_test \
--objectselection ../analysis/selections/selection_jets.json \
--eventselection ../analysis/selections/selection.json \
--merge ../analysis/merging/merging.json \
--split ../analysis/merging/splitting.json \
--shapes \
--dolog
