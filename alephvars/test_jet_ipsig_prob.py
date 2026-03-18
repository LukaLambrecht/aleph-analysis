# Small test script for per-jet probability calculation

import os
import sys
import numpy as np
import awkward as ak

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from alephvars import ipsig_prob, jet_ipsig_prob


if __name__=='__main__':

    track_probs = ak.Array([
      [-0.5, -0.1, -0.9], # only negative probs -> should return 1
      [0.5, -0.5, -0.2, -0.3], # only 1 positive prob -> should return that prob
      [0.5, -0.5, -0.1, 0.5], # expected value: see calculation below
          # p = 0.5*0.5 = 0.25 -> prob = 0.25*(1 + -ln(0.25)) = 0.1875
    ])

    jet_probs = jet_ipsig_prob(None, prob=track_probs)
    print(jet_probs)
