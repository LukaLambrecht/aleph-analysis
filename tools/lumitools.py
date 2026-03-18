import os
import sys

def get_lumidict():
    # note: units are inverse picobarns!

    lumidict = {
      #"1994": 42.639, # from "Preliminary Results on Z Production Cross-Sections and Lepton Forward-Backward Asymmetries using the 1994 Data"
      #"1994": 66.659, # calculated from input files (with LCAL lumi)
      "1994": 57.894 # calculated from input files (with SICAL lumi)
    }
    return lumidict

def get_sqrtsdict():
    # note: units are GeV

    sqrtsdict = {
        "1994": 91.2
    }
    return sqrtsdict
