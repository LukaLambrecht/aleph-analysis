import os
import sys
import json
import numpy as np
import awkward as ak
from fnmatch import fnmatch

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.variabletools import HistogramVariable
from tools.variabletools import DoubleHistogramVariable
from analysis.eventselection import eval_expression


def make_hist(values, variable, secondary_values=None, **kwargs):
    # helper function to switch between make_hist_single and make_hist_double.
    if isinstance(variable, DoubleHistogramVariable):
        (hist, staterrors) = make_hist_double(values, secondary_values, variable, **kwargs)
    else:
        (hist, staterrors) = make_hist_single(values, variable, **kwargs)
    return (hist, staterrors)

def make_hist_single(values, variable, weights=None, clipmin=None):
    # make a 1D histogram from an array of values and a HistogramVariable instance.
    if weights is None: weights = np.ones(len(values)).astype(float)
    hist = np.histogram(values, bins=variable.bins, weights=weights)[0]
    staterrors = np.sqrt(np.histogram(values, bins=variable.bins,
                   weights=np.square(weights))[0])
    if clipmin is not None: hist = np.clip(hist, a_min=clipmin, a_max=None)
    return (hist, staterrors)

def make_hist_double(primary_values, secondary_values, variable, weights=None, clipmin=None):
    # make a flattened 2D histogram from arrays of primary and secondary values
    # and a DoubleHistogramVariable instance.
    if weights is None: weights = np.ones(len(primary_values)).astype(float)
    hist = np.histogram2d(primary_values, secondary_values,
            bins=(variable.primary.bins, variable.secondary.bins),
            weights=weights)[0]
    staterrors = np.sqrt(np.histogram2d(primary_values, secondary_values,
                   bins=(variable.primary.bins, variable.secondary.bins),
                   weights=np.square(weights))[0])
    if clipmin is not None: hist = np.clip(hist, a_min=clipmin, a_max=None)
    hist = hist.flatten(order='F')
    staterrors = staterrors.flatten(order='F')
    return (hist, staterrors)

def make_hist_from_events(events, variable,
        weights=None, weightkey=None,
        maskname=None, mask=None,
        verbose=False, flatten=False,
        **kwargs):
    # make a histogram from an events array and a HistogramVariable or DoubleHistogramVariable instance.

    # set mask
    if mask is None: mask = np.ones(len(events)).astype(bool)
    if maskname is not None:
        named_mask = events[maskname] if maskname in events.fields else eval_expression(events, maskname)
        named_mask = named_mask.to_numpy().astype(bool)
        mask = ((mask) & (named_mask))
    # set weights
    if weights is not None and weightkey is not None:
        raise Exception('Weights and weightkey cannot both be specified.')
    if weights is None: weights = np.ones(len(events)).astype(float)[mask]
    else: weights = weights[mask]
    if weightkey is not None: weights = events[weightkey].to_numpy().astype(float)[mask]
    # get values and make hist
    if isinstance(variable, DoubleHistogramVariable):
        primary_values = eval_expression(events, variable.primary.variable).to_numpy().astype(float)[mask]
        secondary_values = eval_expression(events, variable.secondary.variable).to_numpy().astype(float)[mask]
        (hist, staterrors) = make_hist_double(primary_values, secondary_values, variable,
                               weights=weights, **kwargs)
    else:
        values = eval_expression(events, variable.variable)[mask]
        if flatten:
            weights = ak.broadcast_arrays(weights, values)[0]
            weights = ak.flatten(weights, axis=None)
            values = ak.flatten(values, axis=None)
        values = values.to_numpy().astype(float)
        weights = weights.to_numpy().astype(float)
        (hist, staterrors) = make_hist_single(values, variable, weights=weights, **kwargs)
    # do some printouts for testing
    if verbose:
        msg = f'Making histogram with {len(weights)} entries (unweighted),'
        msg += f' {np.sum(weights)} events (weighted).'
        print(msg)
    return (hist, staterrors)

def merge_events(events, mergedict, verbose=False):
    merged_events = {}
    all_mkeys = []
    for key, val in mergedict.items():
        mkeys = []
        for pattern in val: mkeys += [k for k in events.keys() if fnmatch(k, pattern)]
        mkeys = list(set(mkeys))
        if len(mkeys)==0: continue
        all_mkeys += mkeys
        if verbose: print(f'  - Creating merged sample {key} from original samples {mkeys}')
        merged_events[key] = ak.concatenate([events[k] for k in mkeys])
    for key in events.keys():
        if key not in all_mkeys:
            if verbose: print(f'  - Keeping sample {key} without merging')
            merged_events[key] = events[key]
    return merged_events

def merge_sampledict(sampledict, mergedict, verbose=False):
    merged_sampledict = {}
    all_mkeys = []
    for key, val in mergedict.items():
        mkeys = []
        for pattern in val: mkeys += [k for k in sampledict.keys() if fnmatch(k, pattern)]
        mkeys = list(set(mkeys))
        if len(mkeys)==0: continue
        all_mkeys += mkeys
        if verbose: print(f'  - Creating merged sample {key} from original samples {mkeys}')
        merged_sampledict[key] = [f for mkey in mkeys for f in sampledict[mkey]]
    for key in sampledict.keys():
        if key not in all_mkeys:
            if verbose: print(f'  - Keeping sample {key} without merging')
            merged_sampledict[key] = sampledict[key]
    return merged_sampledict

def make_batches(orig_list, batch_size=None):
    if batch_size is None: return [orig_list]
    counter = 0
    batches = []
    while counter < len(orig_list):
        start = counter
        end = min(len(orig_list), counter + batch_size)
        batches.append(orig_list[start:end])
        counter += batch_size
    return batches
