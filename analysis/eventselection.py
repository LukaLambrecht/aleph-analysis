import ast
import math
import json
import numpy as np
import awkward as ak


def load_eventselection(selectionjson, expect=None, nexpect=None):
    with open(selectionjson, 'r') as f:
        selections = json.load(f)
    keys = list(selections.keys())
    if expect is not None:
        for expected_key in expect:
            if expected_key not in keys:
                msg = f'Expected key {expected_key} not found in {selectionjson}.'
                msg += f' Options are {keys}.'
                raise Exception
    if nexpect is not None:
        if len(keys)!=nexpect:
            msg = f'Found unexpected number of selections in {selectionjson}:'
            msg += f' expected {nexpect} selection names, but found {keys}.'
            raise Exception(msg)
    for key, sel in selections.items():
        if isinstance(sel, str): pass
        elif isinstance(sel, list):
            if len(sel)==0: sel = ''
            else: sel = ' & '.join([f'({s})' for s in sel])
        else:
            msg = 'Unexpected type of selection: expected str or list'
            msg += f' but found {type(sel)} for selection {sel}.'
            raise Exception(msg)
        if len(sel)>0: sel = f'({sel})'
        selections[key] = sel
    return selections


def get_variable_names(expression,
        exclude=['math', 'awkward', 'ak', 'numpy', 'np', 'len']):
    # get variable names in an expression in string format
    # helper function to eval_expression
    root = ast.parse(expression)
    variables = sorted({node.id for node in ast.walk(root) if isinstance(
        node, ast.Name) and not node.id.startswith('_')} - set(exclude))
    variables = [v.split('[',1)[0] for v in variables]
    return variables


def eval_expression(events, expression,
        substitute={'math': math, 'np': np, 'numpy': np, 'ak': ak, 'awkward': ak, 'len': len}):
    # evaluate an expression on events
    exclude = list(substitute.keys())
    variable_names = get_variable_names(expression, exclude=exclude)
    for name in variable_names:
        if name not in events.fields:
            msg = f'Trying to evaluate expression "{expression}",'
            msg += f' but variable "{name}" not found in events.'
            raise Exception(msg)
    substitutions = {name: events[name] for name in variable_names}
    substitutions.update(substitute)
    return eval(expression, substitutions)


def get_selection_mask(events, selection):
    '''
    Get a mask corresponding to a given event selection.
    Input arguments:
      - events: events dict of the form {'<variable name>': awkward array, ...}
      - selection: a string representing a selection, of the form
        "<variable name> <operator> <value>",
        or a list of such strings (in which case the logical & will be taken).
    Returns:
      - a boolean mask
    '''

    # default case of no selection
    if selection is None or len(selection)==0:
        # (note: len 0 can refer to both empty list or empty string)
        mask = ak.values_astype(np.ones(len(events)), 'bool')
        return mask

    # case where selection is a list
    if isinstance(selection, list):
        selection = ' & '.join([f'({s})' for s in selection])

    # apply selection
    selection = f'({selection})'
    mask = ak.values_astype(eval_expression(events, selection), 'bool')
    return mask


def get_selection_masks(events, selections):
    '''
    Get masks for multiple event selections.
    Input arguments:
      - events: events dict of the form {'<variable name>': awkward array, ...}
      - selection: a dict of the following form {'<selection name>': selection},
        where selection is a string or list of strings representing selections,
        see get_selection_mask
    Returns:
      - a dict of masks of the form {'<selection name>': mask}
    '''
    masks = {}
    for selection_name, selection in selections.items():
        masks[selection_name] = get_selection_mask(events, selection)
    return masks


def get_cutflow(events, selection, split=False):
    '''
    Get a cutflow table corresponding to a given event selection.
    Input arguments:
      - events: events dict of the form {'<variable name>': awkward array, ...}
      - selection: a string representing a selection, of the form
        "<variable name> <operator> <value>",
        or a list of such strings.
      - split: split selection (or every string in selection if it is a list)
        by the "&" character.
    Returns:
      - a dict of the form {<selection>: number of passing events}
    '''

    # initialize result
    cutflow = {"initial": int(len(events))}
    mask = np.ones(len(events)).astype(bool)
    if selection is None: return cutflow

    # parse selection
    if isinstance(selection, str): selection = [selection]
    if split:
        split_selection = []
        for selection_string in selection:
            selection_parts = selection_string.split('&')
            for part in selection_parts:
                part = part.strip(' \t\n()')
                split_selection.append(part)
        selection = split_selection

    # loop over selections
    if isinstance(selection, str): selection = [selection]
    for selection_string in selection:
        mask = ((mask) & (get_selection_mask(events, selection_string).to_numpy()))
        cutflow[selection_string] = int(np.sum(mask.astype(int)))

    # return result
    return cutflow
