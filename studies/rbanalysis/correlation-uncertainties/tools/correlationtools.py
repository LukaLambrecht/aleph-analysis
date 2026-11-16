# Small utilities shared between correlation_uncertainty.py and
# plot_efficiencies.py. Kept in its own module (rather than having one script
# import from the other) so that correlation_uncertainty.py can import plotting
# functions from plot_efficiencies.py without creating an import cycle.

import numpy as np


def binomial_error(eps, n):
    '''
    Rough estimate of the statistical uncertainty on an efficiency (or a
    background-subtracted/reweighted equivalent of one): the usual binomial
    proportion standard error, sqrt(eps*(1-eps)/n). This ignores correlations
    between the own- and opposite-hemisphere tagging decisions (i.e. the very effect
    being studied) and, for weighted samples, treats the sum of weights in a bin as
    an effective number of entries -- good enough to get an idea of the size of the
    statistical fluctuations, not a precise uncertainty.
    '''
    eps = np.asarray(eps, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        err = np.where(n > 0, np.sqrt(np.clip(eps*(1.-eps), a_min=0., a_max=None)/n), 0.)
    return err
