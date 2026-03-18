import os
import sys
import json
import onnx
import onnxruntime
import numpy as np
import awkward as ak


def add_variables_jetlevel(jets, names_only=False):
    '''
    Build new variables and add to jets dict.
    Note: hard-coded for now, maybe try to automate later if needed.
    Input arguments:
      - jets: jets dict of the form {'<variable name>': awkward array, ...}
      - names_only: return only the names of the variables that are needed as input
        and the names of the variables that will be produced,
        without actually producing them.
        In this case the jets argument is not used and can be None.
    Returns:
      - If names_only is set to True, returns a dictionary of the form:
        {'input_names': [names of input variables],
         'output_names': [names of variables that will be produced]}
      - Else returns the modified jets dict with added variables.
    '''
    if names_only:
        # just return a dict with input and output names
        # (e.g. useful for deciding which branches to read)
        names = {
          'input_names': [
            'pfcand_pt', 'pfcand_e', 'pfcand_thetarel', 'pfcand_phirel'
          ],
          'output_names': [
            'pfcand_mask', 'pfcand_pt_log', 'pfcand_e_log', 'pfcand_drrel'
          ]
        }
        return names
    # add new variables to jets dict
    jets['pfcand_mask'] = ak.ones_like(jets['pfcand_pt'])
    jets['pfcand_pt_log'] = np.log(jets['pfcand_pt'])
    jets['pfcand_e_log'] = np.log(jets['pfcand_e'])
    jets['pfcand_drrel'] = np.hypot(jets['pfcand_thetarel'], jets['pfcand_phirel'])
    return jets
