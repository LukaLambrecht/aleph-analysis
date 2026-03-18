# Tools for reading and writing ROOT trees with uproot

import os
import sys
import uproot
import awkward as ak


def make_writable_tree(tree, records=None):
    # helper function for write_tree

    # convert tree to format suitable for writing
    # (see here: https://github.com/scikit-hep/uproot5/discussions/903)
    writebranches = dict(zip(ak.fields(tree), ak.unzip(tree)))
    if records is not None:
        for recordname in records:
            tag = recordname + '_'
            recordbranches = {k: v for k, v in writebranches.items() if k.startswith(tag)}
            if len(recordbranches)==0: continue
            writebranches = {k: v for k, v in writebranches.items() if not k.startswith(tag)}
            record = ak.zip({name[len(tag):]: array for name, array in zip(ak.fields(tree), ak.unzip(tree)) if name.startswith(tag)})
            writebranches[recordname] = record

    # need to remove counter branches as they are added automatically,
    # otherwise gives strange dtype errors...
    writebranches = {k: v for k, v in writebranches.items() if not (k[0]=='n' and k[1].isupper())}

    return writebranches


def write_tree(tree, rootfile, treename='Events', records=None, writemode='recreate'):
    '''
    Write a tree to a ROOT file
    Input arguments:
      - tree: tree to write, in the format of an events dict
        of the form {'<variable name>': awkward array}.
      - rootfile: name of the file to write.
    '''

    # convert tree to writable format
    writebranches = make_writable_tree(tree, records=records)

    # write to file
    outputdir = os.path.dirname(rootfile)
    if len(outputdir)>0:
        if not os.path.exists(outputdir): os.makedirs(outputdir)
    writefunc = uproot.recreate
    if writemode=='recreate': pass
    elif writemode=='update': writefunc = uproot.update
    else:
        msg = 'Writemode "{}" not recognized.'.format(writemode)
        raise Exception(msg)
    with writefunc(rootfile) as f:
        f[treename] = writebranches

def write_trees(trees, treenames, rootfile, records=None):
    # write multiple trees.
    # note: this is an attempt to fix corruption warnings and errors
    #       when writing multiple trees by doing write_tree 
    #       with "recreate" and "update" sequentially;
    #       something seems to be going wrong there...
    #       this workaround seems to fix the issue!

    # convert trees to writable format
    writebranches = [make_writable_tree(tree, records=records) for tree in trees]

    # write to file
    outputdir = os.path.dirname(rootfile)
    if len(outputdir)>0:
        if not os.path.exists(outputdir): os.makedirs(outputdir)
    with uproot.recreate(rootfile) as f:
        for tree, treename in zip(writebranches, treenames):
            f[treename] = tree
