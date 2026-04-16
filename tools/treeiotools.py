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

    # need to remove counter branches, as they are anyway added automatically,
    # otherwise gives strange dtype errors...
    writebranches = {k: v for k, v in writebranches.items() if not (k[0]=='n' and k[1].isupper())}

    # the above works fine for 1-level and 2-level awkward arrays,
    # but it does not seem to be possible to write 3-level arrays (e.g. particles per jet per event);
    # so instead need to flatten them to 2-level arrays and store an index
    # to reconstruct the 3-level after reading.
    branchnames = list(writebranches.keys())[:]
    for key in branchnames:
        if writebranches[key].layout.minmax_depth[1]==3:
            val = writebranches.pop(key)
            val_flat = ak.flatten(val, axis=2)
            local_index = ak.local_index(val, axis=1)
            local_index = ak.broadcast_arrays(local_index, val)[0]
            index = ak.flatten(local_index, axis=2)
            writebranches[key] = val_flat
            writebranches[key + '_idx'] = index

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

def reshape_2dto3d_by_index(events):
    # helper function to reshape arrays flattened to 2D back to 3D.
    # todo: make more general.

    # find index arrays and corresponding arrays to be reshaped
    branches = {}
    idx_branches = [b for b in events.fields if b.endswith('_idx')]
    for idx_branch in idx_branches:
        collection = idx_branch.replace('_idx', '')
        branches[idx_branch] = [b for b in events.fields if b.startswith(collection) and b!=idx_branch]

    # loop over index branches and prepare reshaping
    for idx_branch, val in branches.items():
        indices = events[idx_branch]
        particles_per_jet = ak.run_lengths(indices)
        particles_per_jet_flat = ak.flatten(particles_per_jet)
        jets_per_event = ak.num(particles_per_jet)
        # loop over branches to reshape
        for branch in val:
            values = events[branch]
            # reshape
            values_flat = ak.flatten(values)
            temp = ak.unflatten(values_flat, particles_per_jet_flat)
            events[branch] = ak.unflatten(temp, jets_per_event)
