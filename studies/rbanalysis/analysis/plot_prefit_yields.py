# Make a prefit yield comparison plot (data vs. prefit expectation, optionally
# vs. the directly simulated yield) from the json files produced by rbcalc.py,
# reusing the model/plotting code of the fittools submodule.

import os
import sys
import json
import argparse

thisdir = os.path.abspath(os.path.dirname(__file__))
fittoolsdir = os.path.abspath(os.path.join(thisdir, '../fittools'))
# import fittools' 'fit' and 'plotting' submodules directly by their own
# subdirectory (rather than a dotted 'fittools.fit'/'fittools.plotting' import),
# to avoid any collision with this repo's own top-level 'plotting' package
# (used e.g. by rbcalc.py) and any future top-level 'fit' package (as planned
# for the eventual fit implementation).
sys.path.insert(0, os.path.join(fittoolsdir, 'fit'))
sys.path.insert(0, os.path.join(fittoolsdir, 'plotting'))

from model import get_expected_events
from yieldplot import plot_yields

# display order for the categories along the plot's x-axis
CATEGORY_ORDER = [
    'Q', 'S', 'L', 'X', 'C',
    'QQ', 'SS', 'LL', 'XX', 'CC',
    'QS', 'QL', 'QX', 'QC',
    'SL', 'SX', 'SC',
    'LX', 'LC',
    'CX',
]


def load_params(paramsfile):
    '''
    Load a params.json file (as written by rbcalc.py: a list of
    {"name":..., "initial_value":...} dicts) into a flat {name: value} dict,
    as expected by fittools' get_expected_events.
    '''
    with open(paramsfile, 'r') as f:
        params = json.load(f)
    return {p['name']: p['initial_value'] for p in params}


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--yields_data', required=True,
      help='Path to a yields_data.json file produced by rbcalc.py')
    parser.add_argument('-p', '--params', required=True,
      help='Path to a params.json file produced by rbcalc.py')
    parser.add_argument('-s', '--yields_sim', default=None,
      help='Optional path to a yields_sim.json file produced by rbcalc.py'
           ' (directly simulated yields, plotted alongside the prefit model'
           ' prediction as a cross-check).')
    parser.add_argument('-o', '--outputfile', default='prefit_yields.png')
    args = parser.parse_args()

    # observed data yields
    with open(args.yields_data, 'r') as f:
        data = json.load(f)

    # prefit expected yields, from the same (initial-value) parameters rbcalc.py wrote
    prefit_values = load_params(args.params)
    prefit = get_expected_events(prefit_values)

    # plot_yields requires data/prefit/sim to share the exact same key order;
    # get_expected_events always returns keys in a fixed (Q,S,L,C,X, then pair)
    # order, so reorder the other inputs to match rather than relying on the
    # json files happening to have been written in that same order
    data = {key: data[key] for key in prefit.keys()}

    sim = None
    if args.yields_sim is not None:
        with open(args.yields_sim, 'r') as f:
            sim = json.load(f)
        sim = {key: sim[key] for key in prefit.keys()}

    fig, axs = plot_yields(data=data, prefit=prefit, sim=sim, category_order=CATEGORY_ORDER)

    outputdir = os.path.dirname(os.path.abspath(args.outputfile))
    if not os.path.exists(outputdir): os.makedirs(outputdir)
    fig.savefig(args.outputfile, bbox_inches='tight')
    print(f'Wrote {args.outputfile}.')
