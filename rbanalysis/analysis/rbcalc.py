# Calculate efficiencies and correlations for Rb measurement

import os
import sys
import json
import copy
import pickle
import uproot
import pandas as pd
import argparse
import itertools
import numpy as np
import awkward as ak
from fnmatch import fnmatch
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)

from tools.variabletools import read_variables
from tools.variabletools import HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist, read_sampledict, find_files
from tools.lumitools import get_lumidict, get_sqrtsdict
from tools.plottools import make_hist_from_events
from tools.plottools import merge_events, merge_sampledict
from tools.plottools import make_batches
from tools.processinfo import ProcessInfoCollection, ProcessCollection
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.eventselection import get_selection_mask, get_selection_masks
from analysis.objectselection import load_objectselection
from analysis.external_variables import read_external_variables
from analysis.external_variables import find_external_files
from analysis.plot import make_events
from plotting.plot import plot
import tools.condortools as ct

# global pyplot settings
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def calculate_efficiencies(tags, cats, jet1_tag_idx, jet2_tag_idx, event_cat_idx, n_bootstrap=0):
    '''
    Calculate per-event and per-jet efficiencies and correlations
    for a given categorization.
    Input arguments:
    - tags: list of tags, e.g. ["Q", "S", "X", "C", "L"]
    - cats: list of categories, e.g. [("Q", ""), ("Q", "Q")]
    - jet1_tag_idx: numpy array of first jet tag (index in tags)
    - jet2_tag_idx: same for second jet
    - event_cat_idx: numpy array of event category (index in cats)
    '''

    # initializations
    rng = np.random.default_rng()
    n_events = len(event_cat_idx)

    # --------------
    # central values
    # --------------

    # calculate observed number of events and per-event efficiency in each category
    num_events = {}
    event_efficiency = {}
    for cat_idx, cat in enumerate(cats):
        mask = (event_cat_idx == cat_idx)
        numerator = int(np.sum(mask).astype(int))
        num_events[cat] = numerator
        event_efficiency[cat] = numerator / n_events

    # concatenate jets
    jet_tag_idx = np.concatenate((jet1_tag_idx, jet2_tag_idx))
    n_jets = len(jet_tag_idx)

    # calculate per-jet efficiencies
    jet_efficiency = {}
    denominator = len(jet1_tag_idx)
    for tag_idx, tag in enumerate(tags):
        mask = (jet_tag_idx == tag_idx)
        numerator = int(np.sum(mask).astype(int))
        jet_efficiency[tag] = numerator / n_jets

    # calculate correlations
    correlations = {}
    for cat_idx, cat in enumerate(cats):
        if cat[0]=="" or cat[1]=="": continue
        event_eff = event_efficiency[cat]
        jet1_eff = jet_efficiency[cat[0]]
        jet2_eff = jet_efficiency[cat[1]]
        combinatorial_factor = 1 if cat[0]==cat[1] else 2
        denominator = jet1_eff * jet2_eff * combinatorial_factor
        if denominator > 0: correlations[cat] = event_eff / denominator - 1
        else: correlations[cat] = 0

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    bootstrap_jet_eff = {tag: [] for tag in tags}
    bootstrap_corr = {cat: [] for cat in correlations}

    for _ in range(n_bootstrap):

        # resample events with replacement
        indices = rng.integers(0, n_events, size=n_events)
        bs_jet1 = jet1_tag_idx[indices]
        bs_jet2 = jet2_tag_idx[indices]
        bs_event_cat = event_cat_idx[indices]

        # bootstrap event efficiencies
        bs_event_eff = {}
        for cat_idx, cat in enumerate(cats):
            mask = (bs_event_cat == cat_idx)
            bs_event_eff[cat] = np.sum(mask) / n_events

        # bootstrap jet efficiencies
        bs_jet_tags = np.concatenate((bs_jet1, bs_jet2))
        bs_n_jets = len(bs_jet_tags)
        bs_jet_eff = {}
        for tag_idx, tag in enumerate(tags):
            mask = (bs_jet_tags == tag_idx)
            eff = np.sum(mask) / bs_n_jets
            bs_jet_eff[tag] = eff
            bootstrap_jet_eff[tag].append(eff)

        # bootstrap correlations
        for cat in correlations:
            jet1_eff = bs_jet_eff[cat[0]]
            jet2_eff = bs_jet_eff[cat[1]]
            combinatorial_factor = 1 if cat[0] == cat[1] else 2
            denom = jet1_eff * jet2_eff * combinatorial_factor
            if denom > 0: corr = bs_event_eff[cat] / denom - 1
            else: corr = np.nan
            bootstrap_corr[cat].append(corr)

    # ------------------------------------------------------------------
    # Uncertainties = bootstrap standard deviation
    # ------------------------------------------------------------------

    jet_efficiency_unc = {
        tag: np.nanstd(vals, ddof=1)
        for tag, vals in bootstrap_jet_eff.items()
    }

    correlations_unc = {
        cat: np.nanstd(vals, ddof=1)
        for cat, vals in bootstrap_corr.items()
    }

    # ------------------------------------------------------------------
    # make result
    # ------------------------------------------------------------------

    res = {
        "num_events": num_events,
        "event_efficiency": event_efficiency,
        "jet_efficiency": jet_efficiency,
        "jet_efficiency_unc": jet_efficiency_unc,
        "correlations": correlations,
        "correlations_unc": correlations_unc,
    }

    return res


def load_and_tag_events(args, extra_branches=None):
    '''
    Read events for the Rb multi-tag analysis, apply jet-level tag thresholds,
    and assign per-jet and per-event tag/category indices.
    Input arguments:
    - args: an argparse namespace (or any object with the same attributes) with fields
      sim, data, thresholds, objectselection, eventselection, external_variables,
      files_per_batch, year, luminosity, sqrts, xsections, merge, split
      (i.e. the same fields as the rbcalc.py command-line interface).
    - extra_branches: optional list of additional branch names to read, on top of those
      already implied by the object/event selection and splitting instructions
      (e.g. for callers that need extra kinematic variables not otherwise read here).
    Returns:
    - events_combined: dict of the form {dtype: {process: events}}, with the following
      fields added to each events collection: 'Jet_tag_idx', 'Jet_mask_<tag>' (one per
      tag), and 'Event_cat_idx'.
    - tags: list of tag names, as found in the thresholds file.
    - cats: list of tag-pair categories: singly tagged ("tag", "") and doubly
      tagged (tag1, tag2), for all tags and all unordered pairs of tags.
    '''

    # parse arguments
    if args.data is not None and len(args.data)==0: args.data = None

    # read regions
    regions = None

    # read extra object selection to apply
    objectselection = None
    if args.objectselection is not None:
        objectselection = []
        for f in args.objectselection:
            this_objectselection = load_objectselection(f)
            objectselection.append(this_objectselection)
            print('Found following extra object selection to apply:')
            print(this_objectselection[0])
            print('(to the following branches):')
            print(this_objectselection[1])

    # read extra selection to apply
    event_selection_name = None
    eventselection = None
    select_processes = None
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
    variablelist = ['genEventType']

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
    # add selection
    if objectselection is not None:
        for this_objectselection in objectselection:
            branches_to_read += get_variable_names(this_objectselection[0])
    if eventselection is not None:
        branches_to_read += get_variable_names(eventselection)
    # add variables to plot
    branches_to_read += variablelist[:]
    # add variables needed for splitting
    if splitdict is not None:
        for splitkey, this_splitdict in splitdict.items():
            for selection_string in this_splitdict.values():
                branches_to_read += get_variable_names(selection_string)
    # add any extra branches requested by the caller
    if extra_branches is not None:
        branches_to_read += list(extra_branches)
    # remove potential duplicates
    branches_to_read = list(set(branches_to_read))
    print('Found following branches to read:')
    print(branches_to_read)

    # read events
    dtypedict = {'sim': sampledict_sim, 'data': sampledict_data}
    events_combined = make_events(dtypedict,
                       branches_to_read = branches_to_read,
                       objectselection = objectselection,
                       eventselection = eventselection,
                       external_variables = args.external_variables,
                       splitdict = splitdict,
                       lumi = luminosity,
                       xsections = xsections)

    # load and apply thresholds per jet
    thresholds = load_eventselection(args.thresholds)
    tags = list(thresholds.keys())
    for dtype in events_combined.keys():
        for process in events_combined[dtype].keys():
            events = events_combined[dtype][process]
            tag_masks = get_selection_masks(events, thresholds)
            events['Jet_tag_idx'] = ak.ones_like(events['Jets_score_isB']) * -1
            for tag_idx, (tag, mask) in enumerate(tag_masks.items()):
                events[f'Jet_mask_{tag}'] = mask
                events['Jet_tag_idx'] = ak.where(mask, tag_idx, events['Jet_tag_idx'])

    # define categories
    cats = [(tag, '') for tag in tags] + list(itertools.combinations_with_replacement(tags, 2))

    # categorize events based on jet tags
    for dtype in events_combined.keys():
        for process, events in events_combined[dtype].items():
            events['Event_cat_idx'] = np.ones(len(events)) * -1
            jet1tag = events['Jet_tag_idx'][:, 0].to_numpy()
            jet2tag = events['Jet_tag_idx'][:, 1].to_numpy()
            for cat_idx, cat in enumerate(cats):
                tag1idx = tags.index(cat[0]) if cat[0]!= '' else -1
                tag2idx = tags.index(cat[1]) if cat[1]!= '' else -1
                mask = (((jet1tag == tag1idx) & (jet2tag == tag2idx))
                        | ((jet2tag == tag1idx) & (jet1tag == tag2idx)))
                events['Event_cat_idx'] = np.where(mask, cat_idx, events['Event_cat_idx'])

    return events_combined, tags, cats


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-d', '--data', default=None, nargs='+')
    parser.add_argument('-t', '--thresholds', required=True)
    parser.add_argument('--objectselection', default=None, nargs='+')
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--external_variables', default=None)
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--sqrts', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('-r', '--runmode', default='local', choices=['local', 'condor'])
    args = parser.parse_args()

    # if requested, submit this exact command (with runmode switched to 'local')
    # as a condor job instead of running locally
    if args.runmode == 'condor':
        env_script = os.path.join(topdir, 'setup.sh')
        ct.submitScriptAsCondorJob('cjob_rbcalc', sys.argv, env_script=env_script)
        sys.exit()

    events_combined, tags, cats = load_and_tag_events(args)

    # calculate efficiencies and correlations
    nevents = {}
    event_efficiency = {}
    jet_efficiency = {}
    correlations = {}
    jet_efficiency_unc = {}
    correlations_unc = {}
    for dtype in events_combined.keys():
        for process, events in events_combined[dtype].items():

            # get the efficiencies and correlations
            jet1tag = events['Jet_tag_idx'][:, 0].to_numpy()
            jet2tag = events['Jet_tag_idx'][:, 1].to_numpy()
            eventcat = events['Event_cat_idx']
            n_bootstrap = 100 if process not in ['data'] else 0
            temp = calculate_efficiencies(tags, cats, jet1tag, jet2tag, eventcat, n_bootstrap=n_bootstrap)
            nevents[process] = temp['num_events']
            event_efficiency[process] = temp['event_efficiency']
            jet_efficiency[process] = temp['jet_efficiency']
            correlations[process] = temp['correlations']
            jet_efficiency_unc[process] = temp['jet_efficiency_unc']
            correlations_unc[process] = temp['correlations_unc']
            
    # printouts for testing
    print('Number of events:')
    df = pd.DataFrame(nevents); print(df)
    print('Per-event efficiency:')
    df = pd.DataFrame(event_efficiency); print(df)
    print('Per-jet efficiency:')
    df = pd.DataFrame(jet_efficiency); print(df)
    print('Per-jet efficiency uncertainty:')
    df = pd.DataFrame(jet_efficiency_unc); print(df)
    print('Correlations:')
    df = pd.DataFrame(correlations); print(df)
    print('Correlations uncertainty:')
    df = pd.DataFrame(correlations_unc); print(df)

    # make output directory
    outputdir = 'output'
    if not os.path.exists(outputdir): os.makedirs(outputdir)

    # parse and write number of observed events
    if 'data' in nevents.keys():
        num_events = {}
        for key, val in nevents['data'].items():
            newkey = ''.join(key)
            num_events[newkey] = val
        with open(os.path.join(outputdir, 'observed.json'), 'w') as f:
            json.dump(num_events, f, indent=2)

    # parse and write other parameters
    params = []
    scale = len(events_combined['data']['data'])
    params.append({'name': 'scale', 'initial_value': scale})
    params.append({'name': 'Rb', 'initial_value': 0.2158})
    params.append({'name': 'Rc', 'initial_value': 0.172})
    for process in jet_efficiency.keys():
        if process=='data': continue
        for tag, eff in jet_efficiency[process].items():
            params.append({'name': f'eff_{tag}_{process}', 'initial_value': eff})
    for process in correlations.keys():
        if process=='data': continue
        for cat, corr in correlations[process].items():
            if cat[0]=='' or cat[1]=='': continue
            params.append({'name': f'corr_{cat[0]}_{cat[1]}_{process}', 'initial_value': corr})
    with open(os.path.join(outputdir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    # parse and write statistical uncertainties
    mcstats = {}
    for process in jet_efficiency_unc.keys():
        if process not in ['c', 'uds']: continue
        for tag, unc in jet_efficiency_unc[process].items():
            if tag not in ['Q']: continue
            name = f'eff_{tag}_{process}'
            mcstats[name] = {name: [unc, unc]}
    for process in correlations_unc.keys():
        if process == 'data': continue
        for cat, unc in correlations_unc[process].items():
            if cat[0]=='' or cat[1]=='': continue
            name = f'corr_{cat[0]}_{cat[1]}_{process}'
            mcstats[name] = {name: [unc, unc]}
    with open(os.path.join(outputdir, 'systematics_mcstats.json'), 'w') as f:
        json.dump(mcstats, f, indent=2)
