import os
import sys
import json
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.samplelisttools import find_files
from tools.samplelisttools import read_sampledict
from tools.samplelisttools import read_samplelist
from tools.samplelisttools import read_num_entries
from tools.plottools import merge_sampledict
from evaluation.inferencetools import infer_events, add_variables
from analysis.objectselection import load_objectselection
from analysis.objectselection import apply_objectselection
from analysis.eventselection import load_eventselection
from analysis.eventselection import get_variable_names
from analysis.eventselection import get_selection_mask
from evaluation.plot_roc_multi import plot_scores_multi, plot_roc_multi


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples', required=True, nargs='+')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-p', '--preprocess', required=True)
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('--translation', default=None)
    parser.add_argument('--objectselection', default=None)
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--regions', default=None)
    parser.add_argument('--entry_start', default=-1, type=int)
    parser.add_argument('--entry_stop', default=-1, type=int)
    args = parser.parse_args()

    # other settings (hard-coded for now, maybe read as json files later)
    categories = {
        'bb': {
            'selection': 'genEventType==5',
            'score': 'Event_score_isB_prod',
            'color': 'red',
            'label': 'bb'
        },
        'cc': {
            'selection': 'genEventType==4',
            'score': 'Event_score_isC_prod',
            'color': 'blue',
            'label': 'cc'
        },
        'other': {
            'selection': 'genEventType<4',
            'score': 'Event_score_isUDSG_prod',
            'color': 'green',
            'label': 'other'
        }
    }

    # load the preprocess dict
    with open(args.preprocess, 'r') as f:
        prepdict = json.load(f)

    # load the object selection dict and parse selection
    objectselection = None
    if args.objectselection is not None:
        objectselection = load_objectselection(args.objectselection)

    # load the event selection dict and parse selection
    eventselection = None
    if args.eventselection is not None:
        eventselection = load_eventselection(args.eventselection, nexpect=1)
        firstkey = list(eventselection.keys())[0]
        eventselection = eventselection[firstkey]

    # read translation dict
    translation = None
    if args.translation is not None:
        with open(args.translation, 'r') as f:
            translation = json.load(f)

    # load the regions dict and parse selections
    regions = None
    if args.regions is not None:
        regions = load_eventselection(args.regions)

    # define variables to read
    branches_to_read = []
    # add category selection variables
    for cat_settings in categories.values():
        branches_to_read += get_variable_names(cat_settings['selection'])
    # add input variables defined in the preprocess dict
    for key in prepdict['input_names']:
        for varname in prepdict[key]['var_names']:
            if translation is not None: varname = translation.get(varname, varname)
            branches_to_read.append(varname)
    # add variables needed for runtime variable definitions
    newvarnames = add_variables(None, names_only=True)
    for varname in newvarnames['input_names']:
        branches_to_read.append(varname)
    # add branches needed for selection
    if objectselection is not None:
        branches_to_read += get_variable_names(objectselection[0])
    if eventselection is not None:
        eventselection_branches = get_variable_names(eventselection)
        branches_to_read += eventselection_branches
    if regions is not None:
        region_branches = []
        for region_selection in regions.values():
            region_branches += get_variable_names(region_selection)
        branches_to_read += region_branches
    # remove potential duplicates
    branches_to_read = list(set(branches_to_read))
    # filter out runtime variables
    for varname in newvarnames['output_names']:
        if varname in branches_to_read: branches_to_read.remove(varname)
    print('Found following branches to read:')
    print(sorted(branches_to_read))

    # find samples
    print('Finding samples...')
    sampledict = find_files(args.samples)

    # merge into one
    mergedict = {'_': '*'}
    sampledict = merge_sampledict(sampledict, mergedict)
   
    # read all events 
    events = {}
    for key, files in sampledict.items():
        print(f'Reading sample {key}...')
        
        # to avoid out-of-memory issues: split per file and read sequentially
        print(f'Found {len(files)} files for this sample.')
        for fidx, file in enumerate(files):
            print(f'Reading file {file} ({fidx+1}/{len(files)})...')
            this_sampledict = {key: [file]}

            # to avoid even more out-of-memory issues: split in batches
            nevents = read_num_entries(this_sampledict, treename=args.treename, verbose=False)[key][file]
            print(f'Found {nevents} entries in this file.')
            if args.entry_stop is not None and args.entry_stop>0 and args.entry_stop < nevents:
                print(f'Clipping number of entries to read to {args.entry_stop}')
                nevents = args.entry_stop
            batch_size = 1000 # maybe later add as argument
            batch_start_indices = list(range(0, nevents, batch_size))
            print(f'Found {len(batch_start_indices)} batches of size {batch_size} for this file.')
            for bidx, batch_start_index in enumerate(batch_start_indices):
                print(f'Reading batch {bidx+1}/{len(batch_start_indices)}...')

                this_events = read_sampledict(this_sampledict,
                          treename=args.treename,
                          branches=branches_to_read,
                          entry_start=batch_start_index,
                          entry_stop=(batch_start_index+batch_size),
                          verbose=False
                )

                # do object selection
                # note: put before adding new variables for speed,
                # but assumes that the selection does not depend on new variables.
                if objectselection is not None:
                    print('Doing object selection...')
                    this_events[key] = apply_objectselection(this_events[key], objectselection[0], objectselection[1])

                # do pre-selection
                # note: put before adding new variables for speed,
                # but assumes that the selection does not depend on new variables.
                if eventselection is not None:
                    print('Doing pre-selection...')
                    nevents = len(this_events[key])
                    mask = get_selection_mask(this_events[key], eventselection)
                    # note: when one does events = events[mask], the size of events
                    #       in memory actually does not go down significantly!
                    #       under the hood, the full array is kept alive and events[mask]
                    #       just contains a selected view of the full array.
                    #       this leads to memory issues as the masked-out events
                    #       are not actually fully thrown away...
                    #       seems to be solved by the trick below (check with events.nbytes).
                    # note: if the selection efficiency is not tiny (e.g. for signal),
                    #       this "repacking" trick can actually increase the memory usage,
                    #       because some builtin uproot efficiencies are lost.
                    #       but probably not problematic, memory still well within limits.
                    temp = this_events[key][mask]
                    temp2 = ak.from_iter(temp.tolist())
                    del this_events
                    this_events = {key: temp2}
                    nselected = len(this_events[key])
                    print(f'  - Category {key}: selected {nselected} out of {nevents} events.')

                # add to events
                if key not in events.keys(): events[key] = this_events[key]
                else: events[key] = ak.concatenate([events[key], this_events[key]])
                print(f'Cumulative number of events for this sample so far: {len(events[key])}')
                print(f'Size of events array for this sample so far: {events[key].nbytes/1e9} GB')

                # explicitly free up some memory (is this needed? or even useful?)
                del this_events

                #break # break after 1 batch (for testing)

        # end of processing this sample, go to next one.

    # add new variables
    print('Adding new variables...')
    for key in events.keys(): add_variables(events[key])

    # do inference
    scores = {}
    for key in events.keys():
        print(f'Running inference for event category {key}...')
        scores[key] = infer_events(events[key], args.model, prepdict,
                            translation=translation,
                            do_add_variables=False,
                            batch_size=1000)

    # get total weights
    weights = {}
    for key in events.keys():
        #if xsecweighting:
        #    weights[key] = np.multiply(events[key]['lumiwgt'].to_numpy(),
        #      np.multiply(events[key]['genWeight'].to_numpy(),
        #      events[key]['xsecWeight'].to_numpy()))
        #else:
        weights[key] = np.ones(len(events[key]))

    # get labels
    labels = {}
    for key in events.keys():
        labels[key] = {}
        for cat_name, cat_settings in categories.items():
            labels[key][cat_name] = get_selection_mask(events[key], cat_settings['selection']).to_numpy().astype(bool)

    # remove superfluous layer of dict
    events = events['_']
    scores = scores['_']
    weights = weights['_']
    labels = labels['_']

    # loop over regions
    if regions is None: regions = {'all': []}
    for region_name, region_cuts in regions.items():
        print(f'Now evaluating on region {region_name}...')

        # define output directory
        outputdir = os.path.join(args.outputdir, region_name)

        # make a mask for this region
        nevents = len(events)
        mask = get_selection_mask(events, region_cuts)
        nselected = np.sum(mask)
        print(f'  - Category {key}: selected {nselected} out of {nevents} events.')
        this_scores = {key: val[mask] for key, val in scores.items()}
        this_weights = weights[mask]
        this_labels = {key: val[mask] for key, val in labels.items()}

        # plot score distribution and ROC for multiple categories together
        print(f'    Plotting combined ROCs...')
        plot_scores_multi(
                categories,
                outputdir = outputdir,
                scores = this_scores,
                labels = this_labels)
        plot_roc_multi(
                categories,
                outputdir = outputdir,
                scores = this_scores,
                labels = this_labels)
