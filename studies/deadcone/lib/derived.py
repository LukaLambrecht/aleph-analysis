#####################################################
# Derived variables for the dead cone study         #
#####################################################

# Derived variables are branches computed on the fly from the ntuple branches
# (see the derived_variables argument of analysis.plot.make_histograms).
# They are organized in groups; each group has a function that returns a dict of new branches,
# the list of new branch names, and the expressions it depends on (to find the branches to read).
#
# Settings (see configs/settings_default.json):
#   - sqrts: centre-of-mass energy (GeV), used for x_p = 2p/sqrt(s).
#   - displaced: expression selecting displaced constituents (candidate heavy-hadron decay products).
#   - btag / lighttag: jet-level expressions defining b-tagged and light-tagged jets.
#     They can use any jet-level branch, including external variables (e.g. classifier scores)
#     and Jets_nDisplaced (number of displaced constituents in the jet).
#   - bdecay_truth: expression selecting constituents from B hadron decays (truth flag, simulation only),
#     used for the simulation template of the B decay products (see lib/bsubtraction.py).
#   - lund_constituents: expression selecting the constituents used in the declustering.
#   - lund_merge_displaced: if true, the displaced constituents of each jet are merged
#     into one pseudo-particle (heavy-hadron proxy) before reclustering,
#     and the declustering follows the branch containing it.

import os
import sys
import numpy as np
import awkward as ak

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../../'))
sys.path.append(topdir)
sys.path.append(thisdir)

from analysis.eventselection import eval_expression, get_variable_names
from lund import decluster_jets, SPLITTING_FIELDS


MOMENTUM = ['JetsConstituents_px', 'JetsConstituents_py', 'JetsConstituents_pz']
JET_MOMENTUM = ['Jets_px', 'Jets_py', 'Jets_pz']


def _norm(x, y, z): return np.sqrt(x**2 + y**2 + z**2)


def angle_to_jet(events, settings):
    ### angle between each constituent and the axis of its jet
    jet = [events[b] for b in JET_MOMENTUM]
    part = [events[b] for b in MOMENTUM]
    # (jet arrays are broadcast to the constituent dimension)
    cosangle = (jet[0]*part[0] + jet[1]*part[1] + jet[2]*part[2]) / (_norm(*jet) * _norm(*part))
    cosangle = ak.where(cosangle > 1, 1, ak.where(cosangle < -1, -1, cosangle))
    return {'JetsConstituents_angleToJet': np.arccos(cosangle)}


def momentum_fraction(events, settings):
    ### scaled momentum x_p = 2p/sqrt(s) and xi = ln(1/x_p) of each constituent
    xp = 2 * _norm(*[events[b] for b in MOMENTUM]) / settings['sqrts']
    return {'JetsConstituents_xp': xp, 'JetsConstituents_xi': -np.log(xp)}


def jet_tags(events, settings):
    ### displaced constituents, jet tags, and tags of the other jet (hemisphere tagging)
    displaced = ak.values_astype(eval_expression(events, settings['displaced']), bool)
    events = ak.with_field(events, displaced, 'JetsConstituents_isDisplaced')
    events = ak.with_field(events, ak.sum(displaced, axis=-1), 'Jets_nDisplaced')
    res = {'JetsConstituents_isDisplaced': displaced, 'Jets_nDisplaced': events['Jets_nDisplaced']}
    # tag of the other jet: only defined for events with exactly two jets
    two_jets = ak.num(events['Jets_px'], axis=1) == 2
    for tag in ['btag', 'lighttag']:
        name = 'Jets_isBTagged' if tag == 'btag' else 'Jets_isLightTagged'
        other = 'Jets_otherBTagged' if tag == 'btag' else 'Jets_otherLightTagged'
        tagged = ak.values_astype(eval_expression(events, settings[tag]), bool)
        res[name] = tagged
        res[other] = ak.where(two_jets, tagged[:, ::-1], ak.zeros_like(tagged))
    return res


def bdecay_truth(events, settings):
    ### constituents from B hadron decays according to the truth flag (only meaningful in simulation)
    return {'JetsConstituents_isFromBDecay': ak.values_astype(eval_expression(events, settings['bdecay_truth']), bool)}


def lund_splittings(events, settings):
    ### iterative declustering of each jet (see lib/lund.py)
    use = ak.values_astype(eval_expression(events, settings['lund_constituents']), bool)
    px, py, pz = (events[b][use] for b in MOMENTUM)
    e = events['JetsConstituents_e'][use]
    tagged = ak.zeros_like(px, dtype=bool)
    if settings['lund_merge_displaced']:
        # replace the displaced constituents by their sum (heavy-hadron proxy)
        disp = events['JetsConstituents_isDisplaced'][use]
        proxy = [ak.sum(x[disp], axis=-1) for x in (px, py, pz, e)]
        has_proxy = ak.any(disp, axis=-1)
        def merged(x, p): return ak.concatenate([x[~disp], ak.singletons(ak.mask(p, has_proxy), axis=-1)], axis=-1)
        tagged = merged(ak.zeros_like(e, dtype=bool), has_proxy)
        px, py, pz, e = (merged(x, p) for x, p in zip((px, py, pz, e), proxy))
    splittings = decluster_jets(px, py, pz, e, tagged)
    res = {f'JetsSplittings_{f}': splittings[f] for f in SPLITTING_FIELDS}
    with np.errstate(divide='ignore'):
        res['JetsSplittings_lnInvTheta'] = -np.log(splittings['theta'])
        res['JetsSplittings_lnKt'] = np.log(splittings['kt'])
    return res


# registry of groups: name -> (function, new branches, input expressions or setting keys)
GROUPS = {
    'angle': (angle_to_jet, ['JetsConstituents_angleToJet'],
              JET_MOMENTUM + MOMENTUM),
    'xi': (momentum_fraction, ['JetsConstituents_xp', 'JetsConstituents_xi'],
           MOMENTUM),
    'tags': (jet_tags, ['JetsConstituents_isDisplaced', 'Jets_nDisplaced',
                        'Jets_isBTagged', 'Jets_isLightTagged',
                        'Jets_otherBTagged', 'Jets_otherLightTagged'],
             ['Jets_px', '@displaced', '@btag', '@lighttag']),
    'bdecay': (bdecay_truth, ['JetsConstituents_isFromBDecay'], ['@bdecay_truth']),
    'lund': (lund_splittings, [f'JetsSplittings_{f}' for f in SPLITTING_FIELDS] + ['JetsSplittings_lnInvTheta', 'JetsSplittings_lnKt'],
             MOMENTUM + ['JetsConstituents_e', '@lund_constituents']),
}


def get_derived_variables(groups, settings):
    '''
    Get the derived variables for a list of group names.
    Returns:
    - dict of the form {group name: function(events)}, to pass to make_histograms,
    - list of names of all derived branches,
    - list of ntuple branches needed to compute them.
    '''
    funcs = {}
    derived = []
    inputs = []
    # the heavy-hadron proxy in the declustering needs the displaced constituents
    groups = list(groups)
    if 'lund' in groups and settings.get('lund_merge_displaced', False):
        if 'tags' in groups: groups.remove('tags')
        groups.insert(groups.index('lund'), 'tags')
    for group in groups:
        func, names, deps = GROUPS[group]
        funcs[group] = (lambda f: (lambda events: f(events, settings)))(func)
        derived += names
        for dep in deps:
            if dep.startswith('@'): inputs += get_variable_names(settings[dep[1:]])
            else: inputs.append(dep)
    inputs = sorted(set(inputs) - set(derived))
    return funcs, derived, inputs
