# Make score distributions and ROC curves per jet from pre-calculated scores.
# This is intended as a sanity check, to compare against the ROC curves obtained in the training framework.

import os
import sys
import json
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)

from tools.samplelisttools import find_files
from tools.samplelisttools import read_sampledict
from tools.samplelisttools import read_num_entries
from tools.plottools import merge_sampledict
from analysis.eventselection import load_eventselection
from analysis.eventselection import get_selection_mask
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
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--entry_start', default=-1, type=int)
    parser.add_argument('--entry_stop', default=-1, type=int)
    args = parser.parse_args()

    # other settings (hard-coded for now, maybe read as json files later)
    categories = {
        'bb': {
            'selection': 'recojet_isB',
            'score': 'score_isB',
            'color': 'red',
            'label': 'b'
        },
        'cc': {
            'selection': 'recojet_isC',
            'score': 'score_isC',
            'color': 'blue',
            'label': 'c'
        },
        'other': {
            'selection': 'recojet_isUDSG',
            'score': 'score_isUDSG',
            'color': 'green',
            'label': 'udsg'
        }
    }

    # load the event selection dict and parse selection
    eventselection = None
    if args.eventselection is not None:
        eventselection = load_eventselection(args.eventselection, nexpect=1)
        firstkey = list(eventselection.keys())[0]
        eventselection = eventselection[firstkey]

    # define variables to read
    branches_to_read = []
    # add category selection and score variables
    for cat_settings in categories.values():
        branches_to_read += get_variable_names(cat_settings['selection'])
    # add branches needed for selection
    if eventselection is not None:
        branches_to_read += get_variable_names(eventselection)
    # add a variable used in some steps below
    branches_to_read.append('recojet_pt')
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

            # read events
            this_events = read_sampledict(this_sampledict,
                            treename=args.treename,
                            branches=branches_to_read,
                            verbose=False,
                          )
            nevents = len(this_events[process_key])
            print(f'Read {nevents} entries from this file.')

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

            # do event selection
            if eventselection is not None:
                print('Doing event selection...')
                nevents = len(this_events[process_key])
                mask = get_selection_mask(this_events[process_key], eventselection)
                this_events = {process_key: this_events[process_key][mask]}
                nselected = len(this_events[process_key])
                print(f'  - Category {process_key}: selected {nselected} out of {nevents} jets.')

            # add to events
            if process_key not in events.keys(): events[process_key] = this_events[process_key]
            else: events[process_key] = ak.concatenate([events[process_key], this_events[process_key]])
            print(f'Cumulative number of events for this sample so far: {len(events[process_key])}')

            # explicitly free up some memory (is this needed? or even useful?)
            del this_events

        # end of processing this sample, go to next one.

    # printouts for checking
    njets = len(events[process_key])
    print(f'Selected {njets} jets in total.')

    # get scores
    scores = {}
    for process_key in events.keys():
        scores[process_key] = {}
        for cat in ['isB', 'isC', 'isUDSG']:
            jet_scores = events[process_key][f'score_{cat}']
            scores[process_key][f'score_{cat}'] = jet_scores.to_numpy()

    # get labels
    labels = {}
    for process_key in events.keys():
        labels[process_key] = {}
        for cat_name, cat_settings in categories.items():
            jet_labels = get_selection_mask(events[process_key], cat_settings['selection'])
            labels[process_key][cat_name] = jet_labels.to_numpy().astype(bool)

    # remove superfluous layer of dict
    del events
    scores = scores['_']
    labels = labels['_']

    # plot score distribution and ROC for multiple categories together
    print(f'Plotting combined ROCs...')
    plot_scores_multi(
        categories,
        outputdir = args.outputdir,
        scores = scores,
        labels = labels)
    plot_roc_multi(
        categories,
        outputdir = args.outputdir,
        scores = scores,
        labels = labels)
