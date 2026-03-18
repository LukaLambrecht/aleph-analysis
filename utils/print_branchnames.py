import os
import sys

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.samplelisttools import read_sampledict, read_branchnames


if __name__=='__main__':

    inputfile = sys.argv[1]
    treename = sys.argv[2]

    # read and print branch names
    key = 'temp'
    sampledict = {key: [inputfile]}
    branchnames = read_branchnames(sampledict, treename=treename, verbose=True)
