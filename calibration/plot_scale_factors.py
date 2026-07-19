# Post-hoc plotting of the scale factor curves produced by calibration.py
# (i.e. the content of output.json), without needing to rerun the full calibration.

import os
import sys
import json
import argparse

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.plottools import default_flavour_labels
from analysis.plot import get_default_style_dicts
from calibration import make_scale_factor_plot

# map each tag region to the corresponding truth-flavour subprocess
# (same convention as in calibration.py)
process_of_interest = {
    'b_jets': 'bb',
    'c_jets': 'cc',
    's_jets': 'ss',
    'ud_jets': 'uudd',
}

# axis titles per probe score variable
# (same convention as the "variables" defined in calibration.py)
axtitledict = {
    'b_score': 'Probe jet classifier b score',
    'c_score': 'Probe jet classifier c score',
    's_score': 'Probe jet classifier s score',
    'ud_score': 'Probe jet classifier udg score',
}


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--calibration', required=True,
      help='Path to an output.json file produced by calibration.py')
    parser.add_argument('-o', '--outputfile', required=True,
      help='Path of the figure file to write (e.g. output_test/scale_factors_smoothed.png).')
    parser.add_argument('--smooth_window', default=None, type=int,
      help='If provided, smooth the scale factor plot with a double-sided sliding window'
           + ' of this half-width (in number of bins on either side), with each point'
           + ' weighted by its statistical uncertainty. Requires "errors" to be present'
           + ' in the calibration file (i.e. produced by an updated calibration.py).')
    args = parser.parse_args()

    # read the scale factor curves
    with open(args.calibration, 'r') as f:
        output_struct = json.load(f)

    # build annotation labels for each tag region
    flavour_labels = default_flavour_labels()
    region_labeldict = {
        region_name: flavour_labels[process_of_interest[region_name]] + '-enriched region'
        for region_name in output_struct.keys() if region_name in process_of_interest
    }

    # map each tag region to the color used for the corresponding process in the stacked plots
    process_colordict, _, _, _, _ = get_default_style_dicts(list(process_of_interest.values()))
    region_colordict = {
        region_name: process_colordict[process_of_interest[region_name]]
        for region_name in output_struct.keys() if region_name in process_of_interest
    }

    # make the plot
    make_scale_factor_plot(output_struct, args.outputfile,
        region_labeldict=region_labeldict, region_colordict=region_colordict, axtitledict=axtitledict,
        smooth_window=args.smooth_window)
