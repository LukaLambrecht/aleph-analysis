# Plot analysis results

import os
import sys
import json
import argparse
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.variabletools import read_variables
from tools.variabletools import HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist, read_sampledict, find_files
from tools.lumitools import get_lumidict
from tools.plottools import merge_sampledict
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.objectselection import load_objectselection
from analysis.plot import make_histograms



if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-v', '--variables', required=True, nargs='+')
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--objectselection', default=None)
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--external_variables', default=None)
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    args = parser.parse_args()

    # parse arguments
    args.data = None # hard-coded here in order not to break code downstream

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
    if args.eventselection is not None:
        eventselection = load_eventselection(args.eventselection, nexpect=1)
        print('Found following extra event selection to apply:')
        print(eventselection)
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

    # make histograms
    dtypedict = {'sim': sampledict_sim, 'data': sampledict_data}
    hists_combined = make_histograms(dtypedict, variables,
                       branches_to_read = branches_to_read,
                       files_per_batch = args.files_per_batch,
                       objectselection = objectselection,
                       eventselection = eventselection,
                       external_variables = args.external_variables,
                       splitdict = splitdict,
                       lumi = luminosity,
                       xsections = xsections)

    # check number of data categories
    # (only one is supported for now)
    datatag = None
    if sampledict_data is not None:
        keys = list(sampledict_data.keys())
        if len(keys)==1: datatag = keys[0]
        else:
            msg = f'Found unexpected number of data categories: {keys}'
            raise Exception(msg)

    # loop over regions/variables
    hists_sim = hists_combined['sim']
    for region_variable_key, hists in hists_sim.items():
        print(f'Now running on {region_variable_key}...')
        categories = list(hists.keys())

        # determine thresholds
        thresholds = variable.bins

        # find total number of events per category
        nevents_per_cat = {}
        for key, hist in hists.items():
            hist = hist['nominal'][0]
            nevents_per_cat[key] = np.sum(hist)

        # loop over thresholds
        table = []
        for threshold_idx in range(len(thresholds)-1):
            threshold_low = thresholds[threshold_idx]
            #threshold_high = thresholds[threshold_idx+1] # use upper threshold, i.e. bin-by-bin
            threshold_high = thresholds[-1] + 1e-6 # use inclusive range
        
            # determine correct bin in histogram
            bins_edges = variable.bins
            bin_edge_idx_low = np.searchsorted(bins_edges, threshold_low)
            # (note: this gives the first bin edge large than or equal to threshold_low,
            #  so should start at bin edge idx_low, i.e. bin idx_low)
            bin_edge_idx_high = np.searchsorted(bins_edges, threshold_high)
            # (note: this gives the first bin edge larger than or equal to threshold_high,
            #  so should go to (including) bin edge idx_high, i.e. bin idx_high-1)

            # loop over histograms and sum appropriate bins to find number of events per category
            nevents = {}
            for key, hist in hists.items():
                hist = hist['nominal'][0]
                hist = hist[bin_edge_idx_low:bin_edge_idx_high]
                this_nevents = np.sum(hist)
                nevents[key] = this_nevents

            # make ratio to calculate purity
            purity = {}
            sum_nevents = sum(nevents.values())
            for key, val in nevents.items():
                this_purity = 0
                if sum_nevents > 0: this_purity = nevents[key] / sum_nevents
                purity[key] = this_purity

            # make ratio to calculate efficiency
            efficiency = {}
            for key, val in nevents.items(): efficiency[key] = nevents[key] / nevents_per_cat[key]
        
            # store info in table
            row = {
                'idx': threshold_idx,
                'threshold_low': threshold_low,
                'threshold_high': threshold_high
            }
            for key, val in purity.items(): row[f'purity_{key}'] = val
            for key, val in efficiency.items(): row[f'efficiency_{key}'] = val
            table.append(row)

            # printouts for testing
            doprint = False
            if doprint:
                print(f'Lower threshold: {threshold_low}')
                print(f'Upper threshold: {threshold_high}')
                print(f'Lower bin: {bin_edge_idx_low} ({bins_edges[bin_edge_idx_low]} - {bins_edges[bin_edge_idx_low+1]})')
                print(f'Upper bin: {bin_edge_idx_high} ({bins_edges[bin_edge_idx_high-1]} - {bins_edges[bin_edge_idx_high]})')
                print(f'Purities:')
                for key, val in purity.items(): print(f' - {key}: {val}')
                print(f'Efficiencies:')
                for key, val in efficiency.items(): print(f' - {key}: {val}')
                print('-----')

        # make dataframe
        table = pd.DataFrame.from_records(table)
        print(table)

        # make output directory
        if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

        # store table
        table.to_csv(os.path.join(args.outputdir, f'purity_{region_variable_key}.csv'))

        # make a figure
        fig, ax = plt.subplots()
        for category in categories:
            values = table[f'purity_{category}'].values
            ax.stairs(values, edges=thresholds, label=category, linewidth=2)

        # plot aesthetics
        ax.grid(which='both', axis='both')
        ax.set_xlabel(f'Score bin ({region_variable_key})', fontsize=12)
        ax.set_ylabel('Purity', fontsize=12)
        ax.legend()

        # save figure
        fig.tight_layout()
        outputfile = os.path.join(args.outputdir, f'purity_{region_variable_key}.png')
        fig.savefig(outputfile)
        plt.close()
