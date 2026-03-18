# Make score distributions and ROC curves
# for samples with already inferred scores
# (do not re-evaluate the model on the fly)

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
from analysis.external_variables import read_external_variables
from evaluation.plot_roc_multi import plot_scores_multi, plot_roc_multi


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples', required=True, nargs='+')
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('--scoredir', default=None)
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
            'score': 'score_isB', # note: actual definition is hard-coded below
            'color': 'red',
            'label': 'bb'
        },
        'cc': {
            'selection': 'genEventType==4',
            'score': 'score_isC', # note: actual definition is hard-coded below
            'color': 'blue',
            'label': 'cc'
        },
        'other': {
            'selection': 'genEventType<4',
            'score': 'score_isUDSG', # note: actual definition is hard-coded below
            'color': 'green',
            'label': 'other'
        }
    }

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

    # load the regions dict and parse selections
    regions = None
    if args.regions is not None:
        regions = load_eventselection(args.regions)

    # define variables to read
    branches_to_read = []
    # add category selection and score variables
    for cat_settings in categories.values():
        branches_to_read += get_variable_names(cat_settings['selection'])
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
    external_variable_names = None
    for process_key, files in sampledict.items():
        print(f'Reading sample {process_key}...')
        
        # to avoid out-of-memory issues: split per file and read sequentially
        print(f'Found {len(files)} files for this sample.')
        for fidx, file in enumerate(files):
            print(f'Reading file {file} ({fidx+1}/{len(files)})...')
            this_sampledict = {process_key: [file]}
            nevents = read_num_entries(this_sampledict, treename=args.treename, verbose=False)[process_key][file]
            print(f'Found {nevents} entries in this file.')
            this_events = read_sampledict(this_sampledict,
                            treename=args.treename,
                            branches=branches_to_read,
                            verbose=False
                          )

            # read external variables (i.e. per-jet scores)
            if args.scoredir is not None:
                print(f'Reading external variables from {args.scoredir}...')
                external_vars = read_external_variables(
                                  this_sampledict[process_key],
                                  args.scoredir
                                )
                # add to events
                for key, val in external_vars.items():
                    this_events[process_key][key] = val
                # store names
                if external_variable_names is None: external_variable_names = list(external_vars.keys())

            # do object selection
            # note: put before adding new variables for speed,
            # but assumes that the selection does not depend on new variables.
            if objectselection is not None:
                print('Doing object selection...')
                this_events[process_key] = apply_objectselection(this_events[process_key], objectselection[0], objectselection[1])

            # do event selection
            if eventselection is not None:
                print('Doing event selection...')
                nevents = len(this_events[process_key])
                mask = get_selection_mask(this_events[process_key], eventselection)
                this_events = {process_key: this_events[process_key][mask]}
                nselected = len(this_events[process_key])
                print(f'  - Category {process_key}: selected {nselected} out of {nevents} events.')

            # add to events
            if process_key not in events.keys(): events[process_key] = this_events[process_key]
            else: events[process_key] = ak.concatenate([events[process_key], this_events[process_key]])
            print(f'Cumulative number of events for this sample so far: {len(events[process_key])}')

            # explicitly free up some memory (is this needed? or even useful?)
            del this_events

        # end of processing this sample, go to next one.

    # get scores
    # (and also define per-event scores from per-jet scores;
    #  to make more flexible and robust)
    scores = {}
    for process_key in events.keys():
        scores[process_key] = {}
        for cat in ['isB', 'isC', 'isUDSG']:
            jet_scores = events[process_key][f'Jets_score_{cat}']
            event_scores = np.prod(jet_scores, axis=1) # product
            #event_scores = np.mean(jet_scores, axis=1) # average
            #event_scores = np.amin(jet_scores, axis=1) # minimum
            #event_scores = np.amax(jet_scores, axis=1) # maximum
            scores[process_key][f'score_{cat}'] = event_scores

    # get total weights
    weights = {}
    for process_key in events.keys():
        weights[process_key] = np.ones(len(events[process_key]))

    # get labels
    labels = {}
    for process_key in events.keys():
        labels[process_key] = {}
        for cat_name, cat_settings in categories.items():
            labels[process_key][cat_name] = get_selection_mask(events[process_key], cat_settings['selection']).to_numpy().astype(bool)

    # get region masks
    region_masks = {}
    if regions is None: regions = {'all': []}
    for process_key in events.keys():
        region_masks[process_key] = {}
        for region_name, region_cuts in regions.items():
            region_masks[process_key][region_name] = get_selection_mask(events[process_key], region_cuts).to_numpy().astype(bool)

    # remove superfluous layer of dict
    del events
    scores = scores['_']
    weights = weights['_']
    labels = labels['_']
    region_masks = region_masks['_']

    # loop over regions
    for region_name, region_cuts in regions.items():
        print(f'Now evaluating on region {region_name}...')

        # define output directory
        outputdir = os.path.join(args.outputdir, region_name)

        # apply the mask for this region
        nevents = len(weights)
        mask = region_masks[region_name]
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
