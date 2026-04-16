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
from tools.plottools import merge_sampledict
from tools.treeiotools import write_tree
from tools.treeiotools import reshape_2dto3d_by_index


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-o', '--outputfile', required=True)
    args = parser.parse_args()

    # define variables to read (hard-coded for now)
    branches_to_read = [
        'Event_njets', # per event
        'Jets_pt', # per jet
        'JetsConstituents_pt' # per particle
    ]

    # find samples
    sampledict = find_files(args.inputfile)

    # merge into one
    mergedict = {'_': '*'}
    sampledict = merge_sampledict(sampledict, mergedict)
           
    # read events 
    events = read_sampledict(sampledict,
                treename='events',
                branches=branches_to_read,
                verbose=True
    )['_']

    # print types
    print(type(events))
    for branch in events.fields:
        print(f'  - {branch}: {type(events[branch])}, {events[branch].type}')

    # write to output file
    outputdir = os.path.dirname(args.outputfile)
    if len(outputdir)>0 and not os.path.exists(outputdir): os.makedirs(outputdir)
    write_tree(events, args.outputfile, treename='events', records=['Jets', 'JetsConstituents'], writemode='recreate')

    # re-read output file
    branches_to_read += ['JetsConstituents_idx']
    events_test = read_sampledict({'_': [args.outputfile]},
                treename='events',
                branches=branches_to_read,
                verbose=True
    )['_']

    # reshape
    reshape_2dto3d_by_index(events_test)
    
    # check if branches are the same
    print('Difference:')
    for branch in events.fields:
        diff = np.abs(events[branch] - events_test[branch])
        maxdiff = np.amax(diff)
        print(f'  - {branch}: {maxdiff}')
