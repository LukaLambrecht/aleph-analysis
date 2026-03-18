# Plot analysis results

import os
import sys
import json
import copy
import pickle
import uproot
import argparse
import numpy as np
import awkward as ak
from fnmatch import fnmatch
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.variabletools import read_variables
from tools.variabletools import HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist, read_sampledict, find_files
from tools.lumitools import get_lumidict
from tools.plottools import merge_sampledict
from tools.processinfo import ProcessInfoCollection, ProcessCollection
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.eventselection import get_selection_mask
from analysis.objectselection import load_objectselection
from analysis.plot import make_histograms, make_events, plot_hists_default
from analysis.external_variables import read_external_variables
from plotting.plot import plot


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-d', '--data', default=None, nargs='+')
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('--objectselection', default=None)
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--select_processes', default=[], nargs='+')
    parser.add_argument('--external_variables', default=None)
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    args = parser.parse_args()

    # set weight variations to include in the uncertainty band
    # (hard-coded for now, maybe extend later)
    weight_variations = {}

    # parse arguments
    if args.data is not None and len(args.data)==0: args.data = None

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

    # get luminosity from year
    luminosity = args.luminosity
    if args.year is not None:
        lumi_from_year = get_lumidict()[args.year]
        if args.luminosity is None or args.luminosity < 0:
            luminosity = lumi_from_year
        elif luminosity!=lumi_from_year:
            msg = f'WARNING: found inconsistency between provided luminosity ({luminosity})'
            msg += f' and the one corresponding to the provided year ({args.year}: {lumi_from_year}).'
            print(msg)
    if luminosity < 0: luminosity = None

    # define variables to read
    branches_to_read = []
    # add selection
    if objectselection is not None:
        branches_to_read += get_variable_names(objectselection[0])
    if eventselection is not None:
        branches_to_read += get_variable_names(eventselection)
    # add variables needed for splitting
    if splitdict is not None:
        for splitkey, this_splitdict in splitdict.items():
            for selection_string in this_splitdict.values():
                branches_to_read += get_variable_names(selection_string)
    # add jet kinematics
    branches_to_read += ['Jets_pt', 'Jets_theta']
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
                       external_variables = args.external_variables)

    # define selections for events with good tags
    print('Applying selection on tag jet...')
    (tag_idx, probe_idx) = (0, 1)
    #(tag_idx, probe_idx) = (1, 0) # to see if it makes a difference
    tag_selections = {
        'b': f'Jets_score_isB[:, {tag_idx}] > 0.9',
        'c': f'Jets_score_isC[:, {tag_idx}] > 0.7',
        's': f'Jets_score_isS[:, {tag_idx}] > 0.5',
        'ud': f'Jets_score_isUDG[:, {tag_idx}] > 0.5',
    }

    # make masks
    tag_masknames = {k: 'mask_' + k for k in tag_selections.keys()}
    for dtype in events_combined.keys():
        for process_key in events_combined[dtype].keys():
            for key, val in tag_selections.items():
                mask = get_selection_mask(
                         events_combined[dtype][process_key], val).to_numpy().astype(bool)
                events_combined[dtype][process_key]['mask_' + key] = mask
                nbefore = len(mask)
                nafter = np.sum(mask.astype(int))
                print(f'  - {process_key}: selected {nafter} out of {nbefore} events for tag {key}.')

    # determine global normalization factor based on number of selected events
    # with a good tag jet
    norm_factors = {}
    for key in tag_selections.keys():
        ndata = np.sum(events_combined['data']['data']['mask_' + key].to_numpy().astype(int))
        nsim = np.sum(events_combined['sim']['qqb']['mask_' + key].to_numpy().astype(int))
        norm_factor = float(ndata) / nsim
        norm_factors[key] = norm_factor
        print(f'Normalization factor for tag {key}: {norm_factor}')

    # make weights corresponding to the normalization factors above
    # (note: assumes the tag regions to be disjoint)
    weights = np.ones(len(events_combined['sim']['qqb']))
    for key in tag_selections.keys():
        mask = events_combined['sim']['qqb']['mask_' + key].to_numpy()
        weights = np.where(mask, norm_factors[key], weights)
    events_combined['sim']['qqb']['weight'] = weights

    # make plots of probe jet
    variables = [
        HistogramVariable.fromdict({
            "name": "score_isB",
            "variable": f"Jets_score_isB[:, {probe_idx}]",
            "bins": np.linspace(0, 1, num=51),
            "axtitle": "Probe jet classifier b score"
        }),
        HistogramVariable.fromdict({
            "name": "score_isC",
            "variable": f"Jets_score_isC[:, {probe_idx}]",
            "bins": np.linspace(0, 1, num=51),
            "axtitle": "Probe jet classifier c score"
        }),
        HistogramVariable.fromdict({
            "name": "score_isS",
            "variable": f"Jets_score_isS[:, {probe_idx}]",
            "bins": np.linspace(0, 1, num=51),
            "axtitle": "Probe jet classifier s score"
        }),
        HistogramVariable.fromdict({
            "name": "score_isUDG",
            "variable": f"Jets_score_isUDG[:, {probe_idx}]",
            "bins": np.linspace(0, 1, num=51),
            "axtitle": "Probe jet classifier udg score"
        })
    ]
    weights = {'qqb': ['weight']}
    hists_combined = make_histograms(events_combined, variables,
                           regions=tag_masknames,
                           splitdict=splitdict,
                           weights=weights)
    plot_hists_default(hists_combined, variables,
          outputdir=args.outputdir,
          regions=tag_selections,
          datatag='data')

    # calculate ratios to store in output
    ratios = {}
    process_of_interest = {
        'b': 'bb',
        'c': 'cc',
        's': 'ss',
        'ud': 'uudd'
    }
    for region_name in tag_selections.keys():
        ratios[region_name] = {}
        poi = process_of_interest.get(region_name, region_name)
        for variable in variables:
            region_variable_key = f'{region_name}_{variable.name}'
            counts_data = hists_combined['data'][region_variable_key]['data']['nominal'][0]
            counts_sim_this = []
            counts_sim_other = []
            for process_key in hists_combined['sim'][region_variable_key].keys():
                counts = hists_combined['sim'][region_variable_key][process_key]['nominal'][0]
                if process_key == poi: counts_sim_this.append(counts)
                else: counts_sim_other.append(counts)
            if len(counts_sim_this)!=1:
                msg = f'Exactly one process of interest was expected, but found {len(counts_sim_this)}'
                raise Exception(msg)
            counts_sim_this = sum(counts_sim_this)
            counts_sim_other = sum(counts_sim_other)
            # original: just do data / total sim
            numerator = np.clip(counts_data, a_min=0, a_max=None)
            counts_sim_this += counts_sim_other
            denominator = np.clip(counts_sim_this, a_min=1, a_max=None)
            # alternative: do (data - other sim) / (this sim)
            #numerator = np.clip(counts_data - counts_sim_other, a_min=0, a_max=None)
            #denominator = np.clip(counts_sim_this, a_min=1, a_max=None)
            # make ratio
            ratio = np.ones(len(counts_data))
            ratio = np.where(counts_sim_this > 0, np.divide(numerator, denominator), 1)
            ratios[region_name][variable.name] = ratio

    # write output file
    output_struct = {}
    for region_name in tag_selections.keys():
        output_struct[region_name] = {}
        for variable in variables:
            values = ratios[region_name][variable.name]
            bins = variable.bins
            output_struct[region_name][variable.name] = {
                "bins": list(bins),
                "values": list(values)
            }
    outputfile = os.path.join(args.outputdir, 'output.json')
    with open(outputfile, 'w') as f:
        json.dump(output_struct, f, indent=2)
