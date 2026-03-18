# Tools for reading and writing ROOT histograms with uproot

import os
import sys
import uproot
from fnmatch import fnmatch


def load_all_histograms(histfile):
    histograms = {}
    with uproot.open(histfile) as f:
        objs = f.classnames()
        for objname, classname in objs.items():
            ishist = (fnmatch(classname, 'TH1*') or fnmatch(classname, 'TH2*'))
            if not ishist: continue
            obj = f[objname]
            hist = (obj.values(), obj.errors())
            histname = obj.name
            histograms[histname] = hist
    return histograms
