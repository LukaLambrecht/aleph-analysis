# Tools for sample normalization using luminosity and cross-section

import os
import sys
import uproot
import numpy as np
import awkward as ak


def get_normalization_factor(sumgenweights=1, xsec=1, lumi=1):
    # get the normalization factor
    # note: make sure the cross-section and luminosity are expressed in the same unit!
    if sumgenweights is None:
        raise Exception('Cannot calculate normalization factor because sumgenweights is None')
    if xsec is None:
        raise Exception('Cannot calculate normalization factor because xsec is None')
    if lumi is None:
        raise Exception('Cannot calculate normalization factor because lumi is None')
    return xsec * lumi / sumgenweights

def get_normalization_factor_from_tree(runstree, **kwargs):
    # same as above but with utitlity to retrieve sumgenweights from tree
    gensumw = np.sum(runstree['genEventSumw'].array(library='np'))
    return get_normalization_factor(sumgenweights=gensumw, **kwargs)

def get_normalization_factor_from_file(rootfile, **kwargs):
    # same as above but with utitlity to retrieve sumgenweights from file
    with uproot.open(rootfile) as f:
        runstree = rootfile['Runs']
        gensumw = np.sum(runstree['genEventSumw'].array(library='np'))
    return get_normalization_factor(sumgenweights=gensumw, **kwargs)

def get_weights(genweights, sumgenweights=1, xsec=1, lumi=1):
    # get the complete weights for given genweights
    norm = get_normalization_factor(
            sumgenweights=sumgenweights,
            xsec=xsec, lumi=lumi)
    return genweights * norm

def get_weights_from_trees(eventstree, runstree, xsec=1, lumi=1):
    # get the complete weights for given events
    gensumw = np.sum(runstree['genEventSumw'].array(library='np'))
    genweights = eventstree['genWeight'].array(library='np')
    return get_weights(genweights, sumgenweights=gensumw, xsec=xsec, lumi=lumi)
