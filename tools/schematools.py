###########################################################
# Tools for reading ntuples with a different naming scheme #
###########################################################

# The ntuples of some productions use different branch names than our own event-level ntuples
# (e.g. "jet_pT" instead of "Jets_pt", "pfcand_*" instead of "JetsConstituents_*").
# These tools translate such ntuples to our own naming convention while reading them,
# so that the rest of the code does not need to know about the different naming schemes.

# Each naming scheme is defined by a json file in the schemas folder, of the following format:
"""
{
  "detect": [<branches that identify this scheme, i.e. a file is translated if it has all of them>],
  "prefixes": {"<our prefix>": "<their prefix>", ...},
  "branches": {"<our name>": "<their name, or an expression in terms of their names>", ...}
}
"""
# Our branch names are translated as follows:
#   - if the name is in "branches", use the corresponding branch name or expression
#     (expressions can use numpy (np), awkward (ak) and math);
#   - else, if the name starts with one of the prefixes in "prefixes", replace the prefix;
#   - else, use the name as is.
# Branches that cannot be found in a file are skipped (with a warning) when reading.

# A new naming scheme can be supported by adding a json file to the schemas folder.

# Use as a script to make a translation file for model inference
# (see evaluation/inference.py), mapping the input variable names of a model
# trained on ntuples with a different naming scheme to our own names:
#   python schematools.py -s <schema name> -p <preprocess.json of the model> -o <output json file>

import os
import re
import ast
import sys
import json
import math
import argparse
import numpy as np
import awkward as ak

thisdir = os.path.abspath(os.path.dirname(__file__))
schemadir = os.path.join(thisdir, 'schemas')


def load_schemas():
    ### load all naming schemes defined in the schemas folder
    # returns a dict of the form {<schema name>: <schema dict>, ...}
    schemas = {}
    for filename in sorted(os.listdir(schemadir)):
        if not filename.endswith('.json'): continue
        with open(os.path.join(schemadir, filename)) as f:
            schemas[os.path.splitext(filename)[0]] = json.load(f)
    return schemas

_schemas = None
def get_schemas():
    ### same as load_schemas, but only load once
    global _schemas
    if _schemas is None: _schemas = load_schemas()
    return _schemas


def detect_schema(branchnames):
    ### find the naming scheme of a file given its branch names
    # returns a tuple (schema name, schema dict), or (None, None) if no translation is needed.
    branchnames = set(branchnames)
    for name, schema in get_schemas().items():
        if all(branch in branchnames for branch in schema['detect']):
            return (name, schema)
    return (None, None)


def get_variable_names(expression):
    ### get the names of the variables in an expression
    root = ast.parse(expression)
    names = {node.id for node in ast.walk(root) if isinstance(node, ast.Name)}
    return sorted(names - {'np', 'numpy', 'ak', 'awkward', 'math'})


def translate_name(name, schema):
    ### translate one of our branch names to a branch name or expression in the given scheme
    if name in schema['branches']: return schema['branches'][name]
    for prefix, newprefix in schema['prefixes'].items():
        if name.startswith(prefix): return newprefix + name[len(prefix):]
    return name


def inverse_translate_name(name, schema):
    ### translate a branch name in the given scheme back to our own name
    # (only for plain branch names, not for expressions)
    for ourname, theirname in schema['branches'].items():
        if theirname == name: return ourname
    for prefix, newprefix in schema['prefixes'].items():
        if name.startswith(newprefix): return prefix + name[len(newprefix):]
    return name


def get_branches_to_read(branches, schema, branchnames):
    ### find the branches to read from a file with the given scheme
    # input arguments:
    # - branches: list of our branch names to read (or None to read all that can be translated)
    # - schema: schema dict
    # - branchnames: names of the branches available in the file
    # returns a tuple (branches to read from the file, dict matching our names to their expressions);
    # our names that cannot be found in the file are not included.
    branchnames = set(branchnames)
    if branches is None:
        # translate all branches in the file, plus all expressions that can be evaluated
        branches = [inverse_translate_name(b, schema) for b in branchnames]
        branches += [b for b in schema['branches'].keys() if b not in branches]
    expressions = {}
    to_read = set()
    for branch in branches:
        if branch in branchnames:
            # branch exists as is (also in this scheme)
            expressions[branch] = branch
            to_read.add(branch)
            continue
        expression = translate_name(branch, schema)
        needed = get_variable_names(expression)
        if not all(name in branchnames for name in needed): continue
        expressions[branch] = expression
        to_read.update(needed)
    return (sorted(to_read), expressions)


def translate_events(events, expressions):
    ### make an array with our branch names from an array read from a file with another scheme
    # input arguments:
    # - events: awkward array with the branches read from the file
    # - expressions: dict matching our names to expressions in terms of the branches in events
    #   (as returned by get_branches_to_read)
    substitutions = {'np': np, 'numpy': np, 'ak': ak, 'awkward': ak, 'math': math}
    substitutions.update({name: events[name] for name in events.fields})
    output = {}
    for name, expression in expressions.items():
        if expression in events.fields: output[name] = events[expression]
        else: output[name] = eval(expression, dict(substitutions))
    return ak.zip(output, depth_limit=1)


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser(description='Make a translation file for model inference')
    parser.add_argument('-s', '--schema', required=True,
      help='Name of the naming scheme that the model was trained on (see the schemas folder).')
    parser.add_argument('-p', '--preprocess', required=True,
      help='The preprocess.json file of the model.')
    parser.add_argument('-o', '--outputfile', required=True)
    args = parser.parse_args()

    # make the translation
    schema = get_schemas()[args.schema]
    with open(args.preprocess) as f: prepdict = json.load(f)
    translation = {}
    for key in prepdict['input_names']:
        for name in prepdict[key]['var_names']:
            translation[name] = inverse_translate_name(name, schema)

    # write output file
    with open(args.outputfile, 'w') as f: json.dump(translation, f, indent=2)
    print(f'Written {len(translation)} translations to {args.outputfile}.')
