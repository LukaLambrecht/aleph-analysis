# Test script for applying calibrations
# (mabye later merge into main analysis scripts)

import os
import sys
import json
import uproot
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.variabletools import read_variables
from tools.variabletools import HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist, read_sampledict, find_files
from tools.lumitools import get_lumidict, get_sqrtsdict
from tools.plottools import merge_events, merge_sampledict
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.eventselection import get_selection_mask
from analysis.objectselection import load_objectselection
from analysis.plot import make_events, make_histograms, plot_hists_default


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-d', '--data', default=None, nargs='+')
    parser.add_argument('-c', '--calibration', default=None)
    parser.add_argument('-b', '--calibration_branch', default=None)
    parser.add_argument('-v', '--variables', required=True, nargs='+')
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--objectselection', default=None)
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--select_processes', default=[], nargs='+')
    parser.add_argument('--regions', default=None)
    parser.add_argument('--recalculate_regions', default=False, action='store_true')
    parser.add_argument('--external_variables', default=None)
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--sqrts', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--normalizesim', default=False, action='store_true')
    parser.add_argument('--shapes', default=False, action='store_true')
    parser.add_argument('--dolog', default=False, action='store_true')
    args = parser.parse_args()

    # set weight variations to include in the uncertainty band
    # (hard-coded for now, maybe extend later)
    weight_variations = {}

    # parse arguments
    if args.data is not None and len(args.data)==0: args.data = None

    # read regions
    regions = None
    if args.regions is not None:
        regions = load_eventselection(args.regions)
        if not args.recalculate_regions:
            # if the regions are not to be recalculated,
            # use already existing masks (assumed to be present in input files)
            regions = {s: f'mask-{s}' for s in regions.keys()}
        # also add a region with no additional selection applied
        regions['baseline'] = None
        print('Found following regions:')
        print(list(regions.keys()))

    # read extra object selection to apply
    objectselection = None
    if args.objectselection is not None:
        objectselection = load_objectselection(args.objectselection)
        print('Found following extra object selection to apply:')
        print(objectselection[0])
        print('(to the following branches):')
        print(objectselection[1])

    # read extra selection to apply
    event_selection_name = None
    eventselection = None
    select_processes = None
    if args.eventselection is not None:
        eventselection = load_eventselection(args.eventselection, nexpect=1)
        print('Found following extra event selection to apply:')
        print(eventselection)
        if len(args.select_processes)>0:
            select_processes = args.select_processes
            print('(selection will be applied only to the following processes:'
                    + f' {select_processes})')
        event_selection_name = list(eventselection.keys())[0]
        eventselection = eventselection[event_selection_name]

    # read cross-sections
    xsections = None
    if args.xsections is not None:
        with open(args.xsections, 'r') as f:
            xsections = json.load(f)
        print('Found following cross-sections:')
        print(json.dumps(xsections, indent=2))

    # read merging instructions
    mergedict = None
    if args.merge is not None:
        with open(args.merge, 'r') as f:
            mergedict = json.load(f)
        print('Found following instructions for merging samples:')
        print(json.dumps(mergedict, indent=2))

    # read splitting instructions
    splitdict = None
    if args.split is not None:
        with open(args.split, 'r') as f:
            splitdict = json.load(f)
        print('Found following instructions for splitting samples:')
        print(json.dumps(splitdict, indent=2))

    # find samples for simulation
    sampledirs_sim = []
    print('Finding sample files for simulation...')
    for sampledir in args.sim:
        # first check if a file 'files.json' is present (i.e. after merging years)
        ffile = os.path.join(sampledir, 'files.json')
        if os.path.exists(ffile): sampledirs_sim.append(ffile)
        # else default case: find all .root files in the given directory
        else: sampledirs_sim.append(sampledir)
    sampledict_sim = find_files(sampledirs_sim, verbose=False)
    #print('Found following sample dict for simulation:')
    #print(json.dumps(sampledict_sim, indent=2))
    nsimfiles = sum([len(v) for v in sampledict_sim.values()])
    print(f'Found {nsimfiles} simulation files.')

    # find samples for data
    sampledict_data = None
    if args.data is not None:
        sampledirs_data = []
        print('Finding sample files for data...')
        for sampledir in args.data:
            # first check if a file 'files.json' is present (i.e. after merging years)
            ffile = os.path.join(sampledir, 'files.json')
            if os.path.exists(ffile): sampledirs_data.append(ffile)
            # else default case: find all .root files in the given directory
            else: sampledirs_data.append(sampledir)
        sampledict_data = find_files(sampledirs_data, verbose=False)
        #print('Found following sample dict for data:')
        #print(json.dumps(sampledict_data, indent=2))
        ndatafiles = sum([len(v) for v in sampledict_data.values()])
        print(f'Found {ndatafiles} data files.')

    # do merging
    if mergedict is not None:
        print('Merging samples...')
        sampledict_sim = merge_sampledict(sampledict_sim, mergedict, verbose=True)
        if sampledict_data is not None:
            print('Merging data...')
            sampledict_data = merge_sampledict(sampledict_data, mergedict, verbose=False)
        # printouts for testing
        print('Number of files for (merged) samples:')
        for sampledict in [sampledict_sim, sampledict_data]:
            if sampledict is None: continue
            for key, val in sampledict.items():
                print(f'  - {key}: {len(val)}')

    # read variables
    variables = sum([read_variables(f) for f in args.variables], [])
    variablelist = []
    for variable in variables:
        if isinstance(variable, DoubleHistogramVariable):
            variablelist.append(variable.primary.variable)
            variablelist.append(variable.secondary.variable)
        else:
            variablelist.append(variable.variable)
    variablelist = sum([get_variable_names(v) for v in variablelist], [])
    variablelist = list(set(variablelist))

    # get luminosity and center-of-mass energy from year
    luminosity = args.luminosity
    sqrts = args.sqrts
    if args.year is not None:
        lumi_from_year = get_lumidict()[args.year]
        sqrts_from_year = get_sqrtsdict()[args.year]
        if args.luminosity is None or args.luminosity < 0:
            luminosity = lumi_from_year
        elif luminosity!=lumi_from_year:
            msg = f'WARNING: found inconsistency between provided luminosity ({luminosity})'
            msg += f' and the one corresponding to the provided year ({args.year}: {lumi_from_year}).'
            print(msg)
        if args.sqrts is None or args.sqrts < 0:
            sqrts = sqrts_from_year
        elif sqrts!=sqrts_from_year:
            msg = f'WARNING: found inconsistency between provided sqrt(s) ({sqrts})'
            msg += f' and the one corresponding to the provided year ({args.year}: {sqrts_from_year}).'
            print(msg)
    if luminosity < 0: luminosity = None
    if sqrts < 0: sqrts = None

    # define variables to read
    branches_to_read = []
    # add masks
    if regions is not None:
        if not args.recalculate_regions:
            for mask_name in regions.values():
                if mask_name is None: continue
                branches_to_read.append(mask_name)
        else:
            for selection_string in regions.values():
                if selection_string is None: continue
                branches_to_read += get_variable_names(selection_string)
    # add selection
    if objectselection is not None:
        branches_to_read += get_variable_names(objectselection[0])
    if eventselection is not None:
        branches_to_read += get_variable_names(eventselection)
    # add variables to plot
    branches_to_read += variablelist[:]
    # add variables needed for splitting
    if splitdict is not None:
        for splitkey, this_splitdict in splitdict.items():
            for selection_string in this_splitdict.values():
                branches_to_read += get_variable_names(selection_string)
    # remove potential duplicates
    branches_to_read = list(set(branches_to_read))
    print('Found following branches to read:')
    print(branches_to_read)

    # make events
    dtypedict = {'sim': sampledict_sim, 'data': sampledict_data}
    events_combined = make_events(dtypedict,
                       branches_to_read = branches_to_read,
                       objectselection = objectselection,
                       eventselection = eventselection,
                       select_processes = select_processes,
                       regions = regions,
                       recalculate_regions = args.recalculate_regions,
                       lumi = luminosity,
                       xsections = xsections,
                       external_variables = args.external_variables)

    # set total weights to apply
    weights = {
      'qqb': ['weight']
    }

    # make histograms before calibration
    hists_combined_before = make_histograms(events_combined, variables,
                              regions = regions,
                              recalculate_regions = args.recalculate_regions,
                              splitdict = splitdict,
                              weights = weights)

    # if no calibration requested, stop here
    if args.calibration is None: sys.exit()

    # load calibration
    with open(args.calibration, 'r') as f:
        calibration = json.load(f)
    if args.calibration_branch is None:
        raise Exception('Must provide a calibration branch name.')

    # apply calibration
    for dtype in events_combined.keys():
        if dtype != 'sim': continue
        for process_key in events_combined[dtype].keys():
            this_events = events_combined[dtype][process_key]
            score_name = args.calibration_branch.replace('Jets_', '')
            scores = this_events[args.calibration_branch]
            scores_counts = ak.num(scores)
            scores_flat = ak.flatten(scores)
            geneventtype = this_events['genEventType']
            calibration_weights = np.ones(len(this_events))
            for key, val in calibration.items():
                #if key not in ['light', 'c']: continue # for testing
                # get bins and values
                bins = np.array(val[score_name]['bins'])
                values = np.array(val[score_name]['values'])
                # get per-jet values and aggregate to per-event values
                bin_ids = np.digitize(scores_flat, bins, right=True) - 1
                calibration_values_flat = values[bin_ids]
                calibration_values = ak.unflatten(calibration_values_flat, scores_counts)
                calibration_values_aggregated = np.prod(calibration_values, axis=1).to_numpy()
                #calibration_values_aggregated = calibration_values[:, 0].to_numpy() # alternative for testing
                #calibration_values_aggregated = calibration_values[:, 1].to_numpy() # alternative for testing
                calibration_values_aggregated = np.clip(calibration_values_aggregated, a_min=0.5, a_max=2) # for testing
                # make mask where to apply these
                mapping = {'b': 'bb', 'c': 'cc', 's': 'ss', 'ud': 'uudd'}
                mask_selection = splitdict[process_key][mapping[key]]
                mask = get_selection_mask(this_events, mask_selection)
                calibration_weights = np.where(mask, calibration_values_aggregated, calibration_weights)

                # for debugging: make distributions of the applied weights
                outputdir = os.path.join(args.outputdir, 'weights')
                if not os.path.exists(outputdir): os.makedirs(outputdir)
                bins = np.linspace(0, 3, num=51)
                fig, ax = plt.subplots()
                ax.hist(calibration_values_flat, bins=bins, density=True,
                  histtype='step', color='blue', label='Per-jet reweighting factors')
                ax.hist(calibration_values_aggregated, bins=bins, density=True,
                  histtype='step', color='purple', label='Per-event reweighting factors')
                ax.legend()
                outputfile = f'weights_{process_key}_{key}.png'
                outputfile = os.path.join(outputdir, outputfile)
                fig.savefig(outputfile)

            # add weights to events
            this_events['calibration_weight'] = calibration_weights

    # modify total weights to apply
    weights = {
      'qqb': ['weight', 'calibration_weight']
    }

    # make histograms after calibration
    hists_combined_after = make_histograms(events_combined, variables,
                             regions = regions,
                             recalculate_regions = args.recalculate_regions,
                             splitdict = splitdict,
                             weights = weights)

    # do some parsing after the loop above
    # (now that regions might have been recalculated,
    #  replace their event selection by the corresponding mask)
    if regions is not None:
        regions = {region_name: f'mask-{region_name}' for region_name in regions.keys()}

    # check number of data categories
    # (only one is supported for now)
    datatag = None
    if sampledict_data is not None:
        keys = list(sampledict_data.keys())
        if len(keys)==1: datatag = keys[0]
        else:
            msg = f'Found unexpected number of data categories: {keys}'
            raise Exception(msg)

    # plot aesthetics settings
    extracmstext = 'Archived Data'
    lumiheaderparts = []
    if args.year is not None:
        lumiheaderparts.append(args.year)
    if luminosity is not None:
        lumiheaderparts.append('{:.1f}'.format(luminosity) + ' pb$^{-1}$')
    if sqrts is not None:
        lumiheaderparts.append('{:.1f}'.format(sqrts) + ' GeV')
    lumiheader = ', '.join(lumiheaderparts)

    # plotting loop (histograms before calibration)
    outputdir = os.path.join(args.outputdir, 'before')
    if not os.path.exists(outputdir): os.makedirs(outputdir)
    plot_hists_default(hists_combined_before, variables, outputdir,
      regions=regions, datatag=datatag,
      shapes=args.shapes, normalizesim=args.normalizesim, dolog=args.dolog,
      extracmstext=extracmstext, lumiheader=lumiheader,
      event_selection_name=event_selection_name, select_processes=select_processes)

    # plotting loop (histograms after calibration)
    outputdir = os.path.join(args.outputdir, 'after')
    if not os.path.exists(outputdir): os.makedirs(outputdir)
    plot_hists_default(hists_combined_after, variables, outputdir,
      regions=regions, datatag=datatag,
      shapes=args.shapes, normalizesim=args.normalizesim, dolog=args.dolog,
      extracmstext=extracmstext, lumiheader=lumiheader,
      event_selection_name=event_selection_name, select_processes=select_processes)
