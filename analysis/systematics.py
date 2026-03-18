import os
import sys
import numpy as np
import awkward as ak


def make_weight_ratio(varied_weights, nominal_weights):
    '''
    Helper function for making ratios of weights
    '''

    # safety for division by zero
    varied_weights = np.where(nominal_weights==0, 0, varied_weights)
    nominal_weights = np.where(nominal_weights==0, 1, nominal_weights)

    # make ratio
    return np.divide(varied_weights, nominal_weights)


def get_weight_variation(events, systematic):
    '''
    Get appropriately varied weights for a given systematic
    '''

    # dummy case for nominal
    if systematic=='nominal': return np.ones(len(events))

    # simple cases of pre-calculated per-event weights stored in ntuples
    elif( systematic.startswith('btagSF')
        or systematic.startswith('trgSF')
        or systematic.startswith('puWeight') ):

        # set nominal weights for this systematic
        if systematic.startswith('btagSF'): nominal_weightkey = 'btagSF_central'
        elif systematic.startswith('trgSF'): nominal_weightkey = 'trgSF_central'
        elif systematic.startswith('puWeight'): nominal_weightkey = 'puWeight'
        else:
            msg = f'ERROR: no nominal weights are defined for systematic {systematic}.'
            raise Exception(msg)

        # get weights
        varied_weightkey = systematic
        nominal_weights = events[nominal_weightkey].to_numpy()
        varied_weights = events[varied_weightkey].to_numpy()
        
        # return ratio
        return make_weight_ratio(varied_weights, nominal_weights)

    # slightly more complicated cases of theory uncertainties
    # (pre-calculated in ntuples but multiple weights per event)
    elif systematic.startswith('LHEScaleWeight'):
        return get_lhescaleweight_variation(events, systematic)
    elif systematic.startswith('PSWeight'):
        return get_psweight_variation(events, systematic)

    # ad-hoc weights for ABCD method
    elif systematic.startswith('abcdWeight'):
        return events[systematic].to_numpy()

    else:
        msg = f'ERROR: systematic {systematic} not recognized.'
        raise Exception(msg)


def get_lhescaleweight_variation(events, systematic):
    # see here: https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2.html

    # do shape check
    npsweights = events['nLHEScaleWeight'].to_numpy()
    if np.any(npsweights!=9):
        raise Exception('Found unexpected number of LHE weights.')
    
    # get weights at correct index
    if systematic=='LHEScaleWeightMuRDown': return events['LHEScaleWeight'][:, 1].to_numpy()
    elif systematic=='LHEScaleWeightMuFDown': return events['LHEScaleWeight'][:, 3].to_numpy()
    elif systematic=='LHEScaleWeightMuFUp': return events['LHEScaleWeight'][:, 4].to_numpy()
    elif systematic=='LHEScaleWeightMuRUp': return events['LHEScaleWeight'][:, 6].to_numpy()
    else: raise Exception(f'Systematic {systematic} not recognized.')


def get_psweight_variation(events, systematic):
    # see here: https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2.html
    
    # do shape check
    npsweights = events['nPSWeight'].to_numpy()
    if np.any(npsweights!=4):
        raise Exception('Found unexpected number of PS weights.')

    # get weights at correct index
    if systematic=='PSWeightISRUp': return events['PSWeight'][:, 0].to_numpy()
    elif systematic=='PSWeightFSRUp': return events['PSWeight'][:, 1].to_numpy()
    elif systematic=='PSWeightISRDown': return events['PSWeight'][:, 2].to_numpy()
    elif systematic=='PSWeightFSRDown': return events['PSWeight'][:, 3].to_numpy()
    else: raise Exception(f'Systematic {systematic} not recognized.')


def format_systematic_name(systematic):
    '''
    Helper function for parsing systematic names
    from how they are defined in the ntuples
    to how they are recognized downstream.
    '''

    newname = systematic[:]
    if '_up' in newname: newname = newname.replace('_up', '') + 'Up'
    if '_down' in newname: newname = newname.replace('_down', '') + 'Down'
    # to be extended as needed...
    return newname
