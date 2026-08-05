import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

thisdir = os.path.abspath(os.path.dirname(__file__))
fittoolsdir = os.path.abspath(os.path.join(thisdir, '../fittools'))
# add the fittools root (not e.g. fittools/fit) to sys.path: fittools/fit/fit.py
# itself does dotted imports ('from fit.model import ...', 'from plotting.yieldplot
# import ...'), which only resolve if fittools/fit and fittools/plotting are
# importable as *packages* named 'fit'/'plotting' -- i.e. only if fittools itself
# (their parent) is what's on sys.path. This mirrors fittools/run/run_fit.py's own
# working setup. Note: if this script starts needing repo-root modules too (tools/,
# analysis/) later, watch out for 'plotting' -- the aleph-analysis repo root also
# has its own top-level 'plotting' package (used by rbcalc.py), so putting both
# roots on sys.path at once would risk a name collision between the two.
sys.path.insert(0, fittoolsdir)

from fit.fit import fitrb
from plotting.yieldplot import plot_yields


if __name__=='__main__':

    # set parameter file
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', required=True,
                        help='Path to json file with model parameters.')
    parser.add_argument('-d', '--data', default=None,
                        help='Path to json file with observed events per region'
                            +' (default: None, in which case asimov data is constructed).')
    parser.add_argument('-s', '--systematics', default=None, nargs='+',
                        help='Path to json file(s) with systematics'
                            +' (default: None, i.e. do not run any systematic variations).')
    parser.add_argument('--correlated_source_params', default=False, action='store_true',
                        help='If set, parameters affected by the same systematic source'
                            +' are shifted together (fully positively correlated) instead of'
                            +' independently, i.e. one up/down refit per source instead of'
                            +' one up/down refit per parameter (default: False, independent).')
    args = parser.parse_args()

    # load parameters
    with open(args.parameters, 'r') as f:
        params = json.load(f)

    # load observed
    obs = None
    if args.data is not None:
        with open(args.data, 'r') as f:
            obs = json.load(f)

    # read systematics
    systematics = None
    if args.systematics is not None:
        systematics = {}
        for sysfile in args.systematics:
            # define the group name (hard-coded for now based on file name, maybe extend later)
            group_name = os.path.basename(sysfile).replace('.json', '').replace('systematics_', '')
            with open(sysfile, 'r') as f:
                systematics[group_name] = json.load(f)

    # run fit
    fitrb(params, obs=obs, systematics=systematics,
          correlated_source_params=args.correlated_source_params,
          verbose=True, outputdir=thisdir)
