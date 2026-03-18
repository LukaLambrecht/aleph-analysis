import os
import sys
import json
from fnmatch import fnmatch

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from analysis.eventselection import get_selection_mask


def load_objectselection(selectionjson):

    # read the json file and check its contents
    with open(selectionjson, 'r') as f:
        selection = json.load(f)
    keys = list(selection.keys())
    expect = ['selection', 'application']
    for expected_key in expect:
        if expected_key not in keys:
            msg = f'Expected key {expected_key} not found in {selectionjson}.'
            raise Exception(msg)

    # format selection
    sel = selection['selection']
    if isinstance(sel, str): pass
    elif isinstance(sel, list):
        if len(sel)==0: sel = ''
        else: sel = ' & '.join([f'({s})' for s in sel])
    else:
        msg = 'Unexpected type of selection: expected str or list'
        msg += f' but found {type(sel)} for selection {sel}.'
        raise Exception(msg)
    if len(sel)>0: sel = f'({sel})'

    # format branches to apply selection to
    branches = selection['application']

    return (sel, branches)


def apply_objectselection_mask(events, mask, branches):
    apply_branches = []
    for available_branch in events.fields:
        for pattern in branches:
            if fnmatch(available_branch, pattern):
                apply_branches.append(available_branch)
    for branch in apply_branches:
        events[branch] = events[branch][mask]
    return events


def apply_objectselection(events, selection, branches):
    object_mask = get_selection_mask(events, selection)
    return apply_objectselection_mask(events, object_mask, branches)
