##########################################################
# Histogram variable definitions for the dead cone study #
##########################################################

# The variables are written to json files in the format of tools/variabletools.py.
# Each variable is an expression on (derived) branches, in which a jet sample and
# (for constituents) a particle selection are applied through masks:
#   - jet samples: 'incl' (all jets), 'btag' / 'lighttag' (jets for which the *other* jet
#     in the event is b-tagged / light-tagged, i.e. hemisphere-opposite tagging);
#   - particle selections (momentum spectra): 'charged' (all charged constituents),
#     'prompt' (charged constituents that are not displaced).
# For each jet sample, a one-bin counter variable gives the number of jets (for normalization).
#
# Naming scheme:
#   xi_<sample>_<particles>                momentum spectrum (xi = ln(1/x_p))
#   lund<cut>_<sample>_E<low>-<high>       ln(1/theta) of splittings with radiator energy in [low, high),
#                                          with splitting cut <cut> ('' is the default kt cut,
#                                          others are in LUND_CUT_VARIATIONS)
#   lnkt_<sample>_E<low>-<high>            ln(kt) of splittings with radiator energy in [low, high) (no cut)
#   njets_<sample>                         number of jets
#   nrad_<sample>_E<low>-<high>            number of jets with at least one splitting with radiator energy
#                                          in [low, high), i.e. number of radiators (for normalization)
#
# Run as a script to (re)write configs/variables_xi.json, configs/variables_lund.json
# and configs/variables_lund_variations.json.

import os
import json


JET_SAMPLES = {
    'incl': None,
    'btag': 'Jets_otherBTagged',
    'lighttag': 'Jets_otherLightTagged',
}
PARTICLES = {
    'charged': '(np.abs(JetsConstituents_charge) > 0)',
    'prompt': '(np.abs(JetsConstituents_charge) > 0) & (~JetsConstituents_isDisplaced)',
}
# radiator energy bins (GeV) and minimum kt (GeV) for the declustering
ERAD_BINS = [(5, 10), (10, 15), (15, 20), (20, 30), (30, 50)]
KTMIN = 0.2
# splitting cuts: default (written to variables_lund.json) and variations (variables_lund_variations.json)
LUND_CUTS = {'': f'JetsSplittings_kt > {KTMIN}'}
LUND_CUT_VARIATIONS = {
    'kt0': None,
    'kt0p5': 'JetsSplittings_kt > 0.5',
    'kt1': 'JetsSplittings_kt > 1',
    'z0p1': 'JetsSplittings_z > 0.1',
    'zemit0p1': 'JetsSplittings_zemit > 0.1',
}
LUND_CUT_LABELS = {'': f'$k_T$ > {KTMIN} GeV', 'kt0': 'no cut', 'kt0p5': '$k_T$ > 0.5 GeV',
                   'kt1': '$k_T$ > 1 GeV', 'z0p1': 'z > 0.1', 'zemit0p1': '$z_{emit}$ > 0.1'}

XI_BINNING = {'nbins': 30, 'xlow': 0., 'xhigh': 6.}
LUND_BINNING = {'nbins': 24, 'xlow': -0.5, 'xhigh': 5.5}
LNKT_BINNING = {'nbins': 32, 'xlow': -5., 'xhigh': 3.}


def _masked(branch, masks):
    masks = [m for m in masks if m is not None]
    if len(masks) == 0: return branch
    return f'{branch}[' + ' & '.join(f'({m})' for m in masks) + ']'


def counter_variables():
    return [{'name': f'njets_{sample}', 'variable': _masked('Jets_e', [mask]),
             'axtitle': 'Jet energy (counter)', 'unit': 'GeV', 'nbins': 1, 'xlow': 0, 'xhigh': 1000}
            for sample, mask in JET_SAMPLES.items()]


def xi_variables():
    variables = []
    for sample, mask in JET_SAMPLES.items():
        for particles, pmask in PARTICLES.items():
            variables.append({'name': f'xi_{sample}_{particles}',
                              'variable': _masked('JetsConstituents_xi', [mask, pmask]),
                              'axtitle': r'$\xi_p = \ln(1/x_p)$', **XI_BINNING})
    return variables + counter_variables()


def _erad_mask(low, high):
    return f'(JetsSplittings_erad >= {low}) & (JetsSplittings_erad < {high})'


def lund_variables(cuts=LUND_CUTS):
    variables = []
    for cut, cutexpr in cuts.items():
        for sample, mask in JET_SAMPLES.items():
            for (low, high) in ERAD_BINS:
                variables.append({'name': f'lund{cut}_{sample}_E{low}-{high}',
                                  'variable': _masked('JetsSplittings_lnInvTheta', [mask, _erad_mask(low, high), cutexpr]),
                                  'axtitle': r'$\ln(1/\theta)$' + f' ({low} $<E_{{rad}}<$ {high} GeV)',
                                  **LUND_BINNING})
    return variables


def radiator_counters():
    variables = []
    for sample, mask in JET_SAMPLES.items():
        for (low, high) in ERAD_BINS:
            hasrad = f'ak.any({_erad_mask(low, high)}, axis=-1)'
            variables.append({'name': f'nrad_{sample}_E{low}-{high}', 'variable': _masked('Jets_e', [mask, hasrad]),
                              'axtitle': 'Jet energy (counter)', 'unit': 'GeV', 'nbins': 1, 'xlow': 0, 'xhigh': 1000})
    return variables


def lnkt_variables():
    variables = []
    for sample, mask in JET_SAMPLES.items():
        for (low, high) in ERAD_BINS:
            variables.append({'name': f'lnkt_{sample}_E{low}-{high}',
                              'variable': _masked('JetsSplittings_lnKt', [mask, _erad_mask(low, high)]),
                              'axtitle': r'$\ln(k_T/\mathrm{GeV})$' + f' ({low} $<E_{{rad}}<$ {high} GeV)',
                              **LNKT_BINNING})
    return variables


if __name__ == '__main__':
    configdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs')
    for name, variables in [('xi', xi_variables()),
                            ('lund', lund_variables() + radiator_counters() + counter_variables()),
                            ('lund_variations', lund_variables(LUND_CUT_VARIATIONS) + lnkt_variables())]:
        outputfile = os.path.join(configdir, f'variables_{name}.json')
        with open(outputfile, 'w') as f: json.dump(variables, f, indent=2)
        print(f'Written {len(variables)} variables to {outputfile}')
