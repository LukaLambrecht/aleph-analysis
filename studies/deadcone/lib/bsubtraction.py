#############################################################
# Removal of heavy-hadron (B) decay products from xi_p spectra #
#############################################################

# Each method gives the flavour spectra of the charged particles *without* B decay products,
# normalized per jet, in simulation (true event flavour) and in data (tag-corrected, see
# Spectra.corrected_data in plot_results.py), as (values, errors) tuples.
#
# There are two kinds of methods:
#   - TrackRemoval: a per-track selection, applied in data and simulation alike.
#     It is a particle selection of lib/variables.py (histograms xi_<sample>_<particles>).
#     Examples: 'prompt' (displaced tracks removed, impact parameter based);
#     later e.g. a selection on a per-track classifier score.
#   - TemplateSubtraction: the spectrum of the tracks from B decays (truth flag, see the 'bdecay'
#     group in lib/derived.py) is taken from simulation only, per flavour, and subtracted at
#     distribution level (not per jet) from the spectrum of all charged tracks,
#     in simulation and in data.
#
# To add a method: add an entry to METHODS (and, for a track removal, the particle selection
# to lib/variables.py and the inputs it needs to lib/derived.py).
# Methods whose histograms are not in the input file are skipped (see Method.available).

import numpy as np


class Method(object):

    def __init__(self, label):
        # label: short description, shown as a note on the plots
        self.label = label

    def variables(self):
        ### histogram variables needed (with {sample} placeholder)
        raise NotImplementedError()

    def available(self, spectra):
        return all(v.format(sample='incl') in spectra.bins for v in self.variables())

    def sim(self, spectra, flavour):
        ### per-jet spectrum without B decay products for a true flavour in simulation
        raise NotImplementedError()

    def data(self, spectra, flavour):
        ### per-jet spectrum without B decay products for a flavour in data (tag-corrected)
        raise NotImplementedError()


class TrackRemoval(Method):
    ### keep only the tracks passing a per-track selection, in data and simulation

    def __init__(self, label, particles):
        super().__init__(label)
        self.var = 'xi_{sample}_' + particles

    def variables(self): return [self.var]

    def sim(self, spectra, flavour): return spectra.truth(self.var, flavour)

    def data(self, spectra, flavour): return spectra.corrected_data(self.var, flavour)


class TemplateSubtraction(Method):
    ### subtract the spectrum of the tracks from B decays in simulation (per flavour)
    ### from the spectrum of all charged tracks, in simulation and in data

    def __init__(self, label, base='charged', template='fromb'):
        super().__init__(label)
        self.base = 'xi_{sample}_' + base
        self.template = 'xi_{sample}_' + template

    def variables(self): return [self.base, self.template]

    def sim(self, spectra, flavour):
        (base, base_err) = spectra.truth(self.base, flavour)
        (tmpl, tmpl_err) = spectra.truth(self.template, flavour)
        # the template tracks are a subset of the base tracks in the same events,
        # so the difference is statistically independent of the template
        return base - tmpl, np.sqrt(np.clip(base_err**2 - tmpl_err**2, 0, None))

    def data(self, spectra, flavour):
        (base, base_err) = spectra.corrected_data(self.base, flavour)
        (tmpl, tmpl_err) = spectra.truth(self.template, flavour)
        return base - tmpl, np.sqrt(base_err**2 + tmpl_err**2)


# baseline: no removal (all charged constituents)
NOMINAL = TrackRemoval('All charged constituents', 'charged')

# registry of B decay product removal methods: key -> method
METHODS = {
    'ip': TrackRemoval('Displaced tracks removed', 'prompt'),
    'mctemplate': TemplateSubtraction('B decay products subtracted\nvia simulation template'),
}
