# Keep track of commands run for final plots

### Version 03/03/2026

# Compare b and c tagging for following models:
# - Latest model (03/03) with (loose) V0 and dEdx
# - Vs. ALEPH reference for R_b measurement.
# In the Rb-synchronized phase space.
#python evaluate_compare_Rb.py -i ../models/output_20260303_withksloose_retest_rb/output_rerun_test.root

# Compare b and c tagging for following models:
# - Latest model (03/03) with (loose) V0 and dEdx
# - Vs. ALEPH reference for A_FB measurement.
# In the A_FB-synchronized phase space.
#python evaluate_compare_AFB.py -i ../models/output_20260303_withksloose_retrain_afb/output_rerun_train.root

# Compare strange tagging for following models:
# - Latest model (03/03) with (loose) V0 and dEdx (but masked in order to have good sim to data agreement).
# - Earlier iteration (26/02) with dEdx (masked as above) but no V0.
# - Earlier iteration (26/02) trained without dEdx or particle type variables.
# All of this in the conventional training and testing selection
# (does not correspond 100% with R_b or A_FB selection, but that's fine
# because only a relative comparison is made between these models, which are consistent)
python evaluate_strange.py -i ../models/output_20260228_withks_withdedx_masked/output.root ../models/output_20260226_withstrange_withdedx_masked/output.root ../models/output_20260226_withstrange_nodedx_noptype/output.root

### Version 05/03/2026

# Compare b and c tagging for following models:
# - Latest model (05/03) with (tight) V0 and (masked) dEdx
# - Vs. ALEPH reference for R_b measurement.
# In the Rb-synchronized phase space.
python evaluate_compare_Rb.py -i ../models/output_20260305_withnewks_standardized_retest_rb/output_rerun_test.root

# Compare b and c tagging for following models:
# - Latest model (05/03) with (tight) V0 and (masked) dEdx
# - Vs. ALEPH reference for A_FB measurement.
# In the A_FB-synchronized phase space.
python evaluate_compare_AFB.py -i ../models/output_20260305_withnewks_standardized_retrain_afb/output_rerun_train.root

#python evaluate_strange.py -i ../models/output_20260305_withnewks_withdedx_masked_standardized/output.root ../models/output_20260305_noks_withdedx_masked_standardized/output.root ../models/output_20260305_noks_nodedx_noptype_standardized/output.root # contains error in noks_withdedx, to retrain. 
