# Study of data/MC uncertainties in the hemisphere-hemisphere tagging correlations
# used in the Rb double-tag method, following the method of:
# - D. Buskulic et al. (ALEPH Collab.), Phys. Lett. B 401 (1997) 150, Section 4.2.2
# - D. Buskulic et al. (ALEPH Collab.), Phys. Lett. B 401 (1997) 163, Section 6.3
#
# In this analysis, "hemispheres" are the two jets per event (jet index 0 and 1),
# and the tags are the same mutually exclusive NN-score-based tags (Q, S, L, C, X)
# already used in rbcalc.py.
#
# This script computes, for each (tag pair, truth flavour, kinematic variable)
# combination, the differential hemisphere-hemisphere correlation curve as a function
# of that variable, both from simulation (using truth flavour) and from data (using a
# loose "soft" per-flavour tag to isolate an enriched, MC-purity-corrected, sample).
# The comparison between the two is the raw material for assigning a correlation
# systematic uncertainty; combining/weighting the individual results into a single
# number per correlation is left for a later step (see README.md in this folder).

import os
import sys
import json
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)

from analysis.eventselection import eval_expression, get_selection_mask
from calibration.calibration import build_tag_selection_expression
from rbanalysis.analysis.rbcalc import load_and_tag_events
from plot_efficiencies import plot_curves, plot_rho_summary
import tools.condortools as ct

# Note: MC-based background composition estimates below (purity reweighting and
# background subtraction) deliberately do *not* reweight the b/c/uds truth-split
# subsamples by external Rb/Rc branching-fraction constants. They are all truth
# splits (via genEventType) of a single generated 'qqb' sample, so their relative
# sizes already reflect the generator's true branching fractions (up to the modest,
# and here relevant, effect of flavour-dependent selection efficiency); reapplying
# Rb/Rc on top would double-count that.

# kinematic variables sensitive to the various sources of hemisphere-hemisphere
# correlation (gluon radiation, detector acceptance), see Section 3.2 of
# Phys. Lett. B 401 (1997) 150. 'level'='jet' means the expression yields one value
# per jet (i.e. is evaluated once per hemisphere); 'level'='event' means the
# expression yields a single value per event (shared by both hemispheres).
# 'nbins' bin edges are *not* fixed here: they are resolved at runtime as
# quantile-based (equal-population) bins from the simulated sample actually used
# (see compute_quantile_bins), since several of these variables (e.g. jet momentum,
# strongly peaked near the kinematic edge; y3, steeply falling) are poorly suited to
# equal-width binning -- most bins would otherwise be nearly empty while remaining
# statistics pile up in one or two bins.
# Note: there is no explicit thrust axis available in these ntuples; the leading
# jets' own direction is used as a stand-in, which is a good approximation given
# that events are required to have exactly 2 (necessarily back-to-back) jets.
VARIABLES = {
    'jet_momentum': {
        'expression': 'np.sqrt(np.square(Jets_pt) + np.square(Jets_pz))',
        'level': 'jet',
        'axtitle': 'Jet momentum',
        'unit': 'GeV',
        'nbins': 8,
        # unlike the other variables, this one has no selection cut enforcing an
        # upper bound, so a single mismeasured jet can stretch the outermost bin
        # edge arbitrarily far; place the internal edges from the "core" 1st-99th
        # percentile range instead (see compute_quantile_bins), while still
        # covering the full range with no events excluded.
        'outlier_quantile': 0.01,
    },
    'y3': {
        'expression': 'Event_dmerge2 / np.square(91.2)',
        'level': 'event',
        'axtitle': r'$y_3$ (2 $\rightarrow$ 3 jet transition)',
        'unit': '',
        'nbins': 8,
    },
    'cos_theta': {
        'expression': 'np.cos(Jets_theta)',
        'level': 'jet',
        'axtitle': r'Jet $\cos(\theta)$',
        'unit': '',
        'nbins': 8,
    },
    'phi': {
        'expression': 'Jets_phi',
        'level': 'jet',
        'axtitle': r'Jet $\phi$',
        'unit': 'rad',
        'nbins': 8,
    },
}

# extra branches needed on top of what load_and_tag_events already reads
EXTRA_BRANCHES = ['Jets_pt', 'Jets_pz', 'Jets_theta', 'Jets_phi', 'Event_dmerge2']

# colors for the truth-flavour subsamples, matching the palette used elsewhere in the
# repo for bb/cc/uds processes (see tools.plottools.default_flavour_labels)
FLAVOUR_COLORS = {'b': 'darkorchid', 'c': 'slateblue', 'uds': 'paleturquoise'}


def get_variable_values(events, variable_config):
    '''
    Evaluate a variable (as defined in VARIABLES) on a set of events.
    Returns a tuple (v0, v1) of numpy arrays (one entry per event each), representing
    the variable's value "as seen by" jet 0 and jet 1 respectively. For an event-level
    variable, v0 and v1 are identical (the same value, shared by both hemispheres).
    '''
    expression = variable_config['expression']
    values = eval_expression(events, expression)
    if variable_config['level'] == 'jet':
        v0 = values[:, 0].to_numpy().astype(float)
        v1 = values[:, 1].to_numpy().astype(float)
    else:
        v = values.to_numpy().astype(float)
        v0 = v
        v1 = v
    return v0, v1


def compute_quantile_bins(events_per_flavour, variable_config, n_bins, outlier_quantile=0.):
    '''
    Compute quantile-based (equal-population) bin edges for a variable, from the
    pooled (all truth flavours, both jets) simulated distribution -- i.e. the
    inclusive qqbar sample, not reweighted by flavour (see the note on
    Rb/Rc reweighting near the top of this file). Used instead of fixed equal-width
    bins so that each bin carries comparable statistics.
    Input arguments:
    - events_per_flavour: dict {flavour: events}, one per simulated truth flavour.
    - variable_config: entry of VARIABLES (only 'expression' and 'level' are used).
    - n_bins: number of quantile bins to produce.
    - outlier_quantile: if > 0, the *internal* bin edges are placed using quantiles
      of the "core" distribution between this quantile and 1 minus this quantile,
      while the outermost bin edges are still set to the true min/max (so no events
      are excluded, only the *placement* of the bin edges ignores the extreme
      tails). Use for variables with no selection cut enforcing a hard boundary
      (e.g. jet momentum, where a single mismeasured jet can otherwise stretch a
      bin edge arbitrarily far); leave at 0 (default) for variables that already
      have a hard cut-off (e.g. cos(theta), y3, phi).
    Returns a 1D array of n_bins+1 bin edges.
    '''
    all_v = []
    for events in events_per_flavour.values():
        v0, v1 = get_variable_values(events, variable_config)
        all_v.append(v0)
        all_v.append(v1)
    v_pooled = np.concatenate(all_v)
    if outlier_quantile > 0:
        inner_quantiles = np.linspace(outlier_quantile, 1.-outlier_quantile, n_bins-1)
        inner_edges = np.quantile(v_pooled, inner_quantiles)
        edges = np.concatenate([[v_pooled.min()], inner_edges, [v_pooled.max()]])
    else:
        quantiles = np.linspace(0, 1, n_bins+1)
        edges = np.quantile(v_pooled, quantiles)
    edges = np.unique(edges)
    if len(edges) < n_bins+1:
        msg = f'WARNING: only {len(edges)-1} distinct quantile bins could be formed'
        msg += f' (requested {n_bins}); the variable may have many repeated values.'
        print(msg)
    return edges


def plot_variable_distributions(events_per_flavour, variables, outputdir, n_fine_bins=40):
    '''
    Diagnostic plot (simulation only): for each variable, show the pooled
    (jet 0 + jet 1) distribution per truth flavour, both at fine resolution (to see
    the natural shape) and with the actual analysis bin edges overlaid as dashed
    lines, annotated with the number of entries per analysis bin. Useful for judging
    whether the chosen binning is reasonable and where statistics are limited.
    These are plotted directly from the events as loaded by load_and_tag_events,
    i.e. after the same baseline object/event selection as the rest of the analysis
    (and as rbcalc.py itself).
    Input arguments:
    - events_per_flavour: dict {flavour: events}, one per simulated truth flavour.
    - variables: dict {variable_name: variable_config}, with variable_config already
      resolved (i.e. containing a concrete 'bins' entry, see compute_quantile_bins).
    - outputdir: directory to write the figures to.
    - n_fine_bins: number of bins for the fine-grained shape histogram.
    '''
    flavours = list(events_per_flavour.keys())

    for variable_name, variable_config in variables.items():
        bins = np.asarray(variable_config['bins'], dtype=float)
        fine_bins = np.linspace(bins[0], bins[-1], n_fine_bins+1)

        fig, axs = plt.subplots(nrows=1, ncols=len(flavours), figsize=(4.5*len(flavours), 4), squeeze=False)
        print(f'Entries per analysis bin for variable "{variable_name}" (pooled hemispheres):')
        for col_idx, flavour in enumerate(flavours):
            ax = axs[0, col_idx]
            v0, v1 = get_variable_values(events_per_flavour[flavour], variable_config)
            v_own = np.concatenate([v0, v1])
            color = FLAVOUR_COLORS.get(flavour, 'grey')

            ax.hist(v_own, bins=fine_bins, color=color, alpha=0.6,
                    label=f'MC {flavour}-jets ({len(v_own)} entries)')
            for edge in bins:
                ax.axvline(x=edge, color='black', linestyle='dashed', linewidth=0.7)

            n_bin, _ = np.histogram(v_own, bins=bins)
            print(f'  {flavour}: {n_bin.tolist()} (total {len(v_own)})')
            centers = 0.5*(bins[:-1] + bins[1:])
            # give extra headroom above the histogram so the per-bin entry counts and
            # the legend can both be placed without overlapping the bars or each other
            barmax = ax.get_ylim()[1]
            ax.set_ylim(0, barmax*1.5)
            label_y = barmax*1.03
            for c, n in zip(centers, n_bin):
                ax.text(c, label_y, str(int(n)), ha='center', va='bottom', fontsize=7, rotation=90)

            unit = variable_config['unit']
            xlabel = variable_config['axtitle']
            if len(unit) > 0: xlabel += f' [{unit}]'
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Jets', fontsize=11)
            ax.legend(fontsize=9, loc='upper right')
            ax.tick_params(labelsize=9)

        fig.tight_layout()
        figpath = os.path.join(outputdir, f'distribution_{variable_name}.png')
        fig.savefig(figpath, bbox_inches='tight')
        plt.close(fig)
        print(f'Figure saved to {figpath}.')


def plot_soft_tag_control_plots(control_data, variable_config, variable_name, outputdir):
    '''
    Standard stacked-simulation-vs-data control plots for the soft-tag-selected
    samples, one panel per soft tag (light/c/b-enriched) side by side in a single
    figure -- the same 1xN grid layout as plot_variable_distributions.
    Each panel shows the pooled (jet 0 + jet 1) distribution of the analysis
    variable, with simulation stacked by truth flavour and scaled by that soft tag's
    norm_factor (the same overall data/MC normalization actually used by
    compute_correlation_curves_subtracted -- so this plot shows exactly what that
    method assumes), and data overlaid as points. Meant to check, for each soft tag,
    whether the resulting sample has adequate statistics and purity (i.e. whether the
    target flavour dominates the stack).
    Input arguments:
    - control_data: dict {soft_flavour: {'data_events':..., 'mc_events_per_flavour':...,
      'norm_factor':...}}, one entry per soft tag, with data_events/mc_events_per_flavour
      already restricted to that soft tag's selection (see build_soft_tag_mask), and
      norm_factor as computed and used in compute_correlation_curves_subtracted.
    - variable_config, variable_name: as elsewhere.
    - outputdir: directory to write the figure to.
    '''
    soft_flavours = list(control_data.keys())
    bins = np.asarray(variable_config['bins'], dtype=float)
    centers = 0.5*(bins[:-1] + bins[1:])
    widths = bins[1:] - bins[:-1]

    fig, axs = plt.subplots(nrows=1, ncols=len(soft_flavours),
                 figsize=(4.5*len(soft_flavours), 4.5), squeeze=False)
    for col_idx, soft_flavour in enumerate(soft_flavours):
        ax = axs[0, col_idx]
        entry = control_data[soft_flavour]
        data_events = entry['data_events']
        mc_events_per_flavour = entry['mc_events_per_flavour']
        norm_factor = entry['norm_factor']

        dv0, dv1 = get_variable_values(data_events, variable_config)
        n_data, _ = np.histogram(np.concatenate([dv0, dv1]), bins=bins)
        data_err = np.sqrt(np.clip(n_data.astype(float), a_min=0., a_max=None))

        bottom = np.zeros(len(centers))
        for flavour, events in mc_events_per_flavour.items():
            v0, v1 = get_variable_values(events, variable_config)
            n_mc, _ = np.histogram(np.concatenate([v0, v1]), bins=bins)
            n_mc_scaled = n_mc.astype(float)*norm_factor
            color = FLAVOUR_COLORS.get(flavour, 'grey')
            label = f'{flavour}-jets' + (' (target)' if flavour == soft_flavour else '')
            ax.bar(centers, n_mc_scaled, width=widths, bottom=bottom, color=color,
                   alpha=0.8, label=label, align='center')
            bottom += n_mc_scaled

        ax.errorbar(centers, n_data, yerr=data_err, linestyle='None', marker='o',
                    color='black', markersize=4, label='data')

        unit = variable_config['unit']
        xlabel = variable_config['axtitle']
        if len(unit) > 0: xlabel += f' [{unit}]'
        ax.set_xlabel(xlabel, fontsize=11)
        if col_idx == 0: ax.set_ylabel('Jets', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1]*1.3)
        ax.text(0.05, 0.95, f'{soft_flavour} soft selection', ha='left', va='top',
                fontsize=13, transform=ax.transAxes)
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    figpath = os.path.join(outputdir, f'controlplot_{variable_name}.png')
    fig.savefig(figpath, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {figpath}.')


def compute_tag_curves(v0, v1, tag0, tag1, bins, weights=None):
    '''
    Compute the "own-hemisphere" and "opposite-hemisphere" tagging efficiency curves
    for a single tag, as a function of a variable v, from a sample of events.
    Input arguments:
    - v0, v1: variable value as seen by jet 0 / jet 1 (see get_variable_values).
    - tag0, tag1: boolean arrays, whether jet 0 / jet 1 passes the tag's selection.
    - bins: bin edges for v.
    - weights: optional per-event weights (e.g. MC-purity weights for a data sample);
      defaults to unit weight per event.
    Returns a dict with:
    - 'f_v': normalized population per bin (from pooling both jets' v).
    - 'eps_own': fraction of hemispheres in each v-bin that pass the tag
      (efficiency of a hemisphere as a function of its own v).
    - 'eps_oppo': fraction of the *partner* hemisphere passing the tag, as a function
      of the v-bin of *this* hemisphere (the correlation-sensitive curve: if hemispheres
      tagged independently, this would not depend on v).
    - 'mean_eff': overall (marginal) tagging efficiency for this tag in this sample.
    '''
    if weights is None: weights = np.ones(len(v0))
    n_bin, sum_own, sum_oppo = compute_raw_bin_sums(v0, v1, tag0, tag1, bins, weights=weights)
    return curves_from_raw_sums(n_bin, sum_own, sum_oppo)


def compute_raw_bin_sums(v0, v1, tag0, tag1, bins, weights=None):
    '''
    Compute the raw (unnormalized) per-bin population and same-/opposite-hemisphere
    tagged sums that underlie compute_tag_curves. Exposed separately so that several
    samples (e.g. data and multiple MC flavours) can have their raw sums combined
    (e.g. subtracted) *before* normalizing into efficiency curves -- see
    compute_correlation_curves_subtracted.
    Returns (n_bin, sum_own, sum_oppo), each a 1D array of length len(bins)-1.
    '''
    if weights is None: weights = np.ones(len(v0))
    v_own = np.concatenate([v0, v1])
    tag_own = np.concatenate([tag0, tag1]).astype(float)
    tag_oppo = np.concatenate([tag1, tag0]).astype(float)
    w = np.concatenate([weights, weights])

    bins = np.asarray(bins, dtype=float)
    n_bin, _ = np.histogram(v_own, bins=bins, weights=w)
    sum_own, _ = np.histogram(v_own, bins=bins, weights=w*tag_own)
    sum_oppo, _ = np.histogram(v_own, bins=bins, weights=w*tag_oppo)
    return n_bin, sum_own, sum_oppo


def curves_from_raw_sums(n_bin, sum_own, sum_oppo):
    '''
    Normalize raw per-bin sums (as returned by compute_raw_bin_sums, possibly after
    combining/subtracting several samples) into the same curve dict format returned
    by compute_tag_curves.
    '''
    with np.errstate(invalid='ignore', divide='ignore'):
        eps_own = np.where(n_bin > 0, sum_own/n_bin, 0.)
        eps_oppo = np.where(n_bin > 0, sum_oppo/n_bin, 0.)
    total = n_bin.sum()
    f_v = n_bin/total if total > 0 else np.zeros_like(n_bin)
    mean_eff = sum_own.sum()/total if total > 0 else 0.

    return {
        'f_v': f_v, 'eps_own': eps_own, 'eps_oppo': eps_oppo, 'mean_eff': mean_eff,
        'n_bin': n_bin,
    }


def correlation_from_curves(f_v, curves_I, curves_J):
    '''
    Combine the per-tag curves (as returned by compute_tag_curves) for a tag pair
    (I, J) into the differential correlation ("contribution to rho") for one variable,
    generalizing Eq. (4) of Phys. Lett. B 401 (1997) 150 to a pair of possibly
    different tags, following Section 6.3 of Phys. Lett. B 401 (1997) 163.
    '''
    numerator = np.sum(f_v * (
        curves_I['eps_own']*curves_J['eps_oppo'] + curves_J['eps_own']*curves_I['eps_oppo']
    ))
    denominator = 2 * curves_I['mean_eff'] * curves_J['mean_eff']
    if denominator <= 0: return 0.
    return float(numerator/denominator - 1)


def curves_to_json(curves):
    '''
    Convert a dict {tag: curve_dict} (as returned by compute_tag_curves /
    curves_from_raw_sums) into a JSON-serializable dict, dropping the per-tag 'f_v'
    and 'n_bin' (identical for every tag, so stored once at the caller's level).
    '''
    return {tag: {
        'eps_own': c['eps_own'].tolist(),
        'eps_oppo': c['eps_oppo'].tolist(),
        'mean_eff': c['mean_eff'],
    } for tag, c in curves.items()}


def compute_correlation_curves(events, variable_config, tags, weights=None):
    '''
    For a single sample of events (e.g. one truth flavour's simulation, or a
    soft-tag-selected data sample), compute the differential same-/opposite-
    hemisphere efficiency curve for every tag (each tag's curve is a property of
    that tag alone, computed once), plus the resulting aggregate correlation value
    ('rho') for every unordered pair of tags, for one variable.
    Returns a dict with 'bins', 'f_v', 'n_bin' (shared by all tags), 'curves'
    (dict {tag: {eps_own, eps_oppo, mean_eff}}), and 'rho' (dict {'tagI_tagJ': value}).
    '''
    v0, v1 = get_variable_values(events, variable_config)
    bins = variable_config['bins']

    tag_curves = {}
    for tag in tags:
        tag0 = events[f'Jet_mask_{tag}'][:, 0].to_numpy().astype(bool)
        tag1 = events[f'Jet_mask_{tag}'][:, 1].to_numpy().astype(bool)
        tag_curves[tag] = compute_tag_curves(v0, v1, tag0, tag1, bins, weights=weights)

    # f_v and n_bin are identical for all tags (same v0, v1, bins); take from any one
    f_v = next(iter(tag_curves.values()))['f_v']
    n_bin = next(iter(tag_curves.values()))['n_bin']

    rho = {}
    for tagI, tagJ in itertools.combinations_with_replacement(tags, 2):
        rho[f'{tagI}_{tagJ}'] = correlation_from_curves(f_v, tag_curves[tagI], tag_curves[tagJ])

    return {
        'bins': list(bins),
        'f_v': f_v.tolist(),
        'n_bin': n_bin.tolist(),
        'curves': curves_to_json(tag_curves),
        'rho': rho,
    }


def build_soft_tag_mask(events, cuts):
    '''
    Build an event-level mask requiring *both* jets to pass a loose ("soft") per-jet
    selection, used to isolate a flavour-enriched (but not necessarily pure) data
    sample, following the "soft b tag" approach of Section 4.2.2 of
    Phys. Lett. B 401 (1997) 150.
    Input arguments:
    - events: events collection.
    - cuts: list of {"branch":..., "min"/"max":...} dicts, as in tools/thresholds_soft.json.
    '''
    expr0 = build_tag_selection_expression(cuts, 0)
    expr1 = build_tag_selection_expression(cuts, 1)
    mask0 = get_selection_mask(events, expr0).to_numpy().astype(bool)
    mask1 = get_selection_mask(events, expr1).to_numpy().astype(bool)
    return mask0 & mask1


def get_pooled_bin_counts(events_per_flavour, variable_config, bins):
    '''
    For each flavour's events, evaluate the variable and histogram the pooled
    (jet 0 and jet 1) values into the given bins. Shared helper for the purity-
    reweighting and background-subtraction methods below.
    Returns a dict {flavour: n_bin} (each a 1D array of length len(bins)-1).
    '''
    n_bin_per_flavour = {}
    for flavour, events in events_per_flavour.items():
        if len(events) == 0:
            n_bin_per_flavour[flavour] = np.zeros(len(bins)-1)
            continue
        fv0, fv1 = get_variable_values(events, variable_config)
        v_own = np.concatenate([fv0, fv1])
        n_bin, _ = np.histogram(v_own, bins=bins)
        n_bin_per_flavour[flavour] = n_bin.astype(float)
    return n_bin_per_flavour


def compute_purity_weights(v0, v1, mc_events_per_flavour, soft_flavour, variable_config):
    '''
    Estimate, from simulation, the purity of the soft-tag-selected sample as a
    function of the variable v (i.e. the fraction of soft-tag-selected hemispheres
    that are genuinely of flavour soft_flavour, per v-bin), and return the
    corresponding per-hemisphere weight for each entry of the (v0, v1) data sample
    (the weight to apply is the purity of the bin that entry falls into).
    This is "method 2" (purity reweighting) of Section 4.2.2 of
    Phys. Lett. B 401 (1997) 150; see compute_correlation_curves_subtracted for
    "method 1" (bin-by-bin background subtraction).
    Input arguments:
    - v0, v1: variable values (data sample already restricted to the soft tag).
    - mc_events_per_flavour: dict {flavour: events}, each already restricted to the
      SAME soft-tag selection (see build_soft_tag_mask), for all considered flavours.
    - soft_flavour: the flavour the soft tag is designed to select.
    - variable_config: as in VARIABLES.
    '''
    bins = np.asarray(variable_config['bins'], dtype=float)
    n_bin_per_flavour = get_pooled_bin_counts(mc_events_per_flavour, variable_config, bins)

    total = sum(n_bin_per_flavour.values())
    with np.errstate(invalid='ignore', divide='ignore'):
        purity_per_bin = np.where(total > 0, n_bin_per_flavour[soft_flavour]/total, 0.)

    # assign each data entry (pooling jet 0 and jet 1) the purity of its own v-bin
    v_own = np.concatenate([v0, v1])
    bin_ids = np.clip(np.digitize(v_own, bins) - 1, 0, len(purity_per_bin)-1)
    weights_own = purity_per_bin[bin_ids]
    n = len(v0)
    return weights_own[:n], weights_own[n:], purity_per_bin


def compute_correlation_curves_subtracted(data_events, mc_events_per_flavour, target_flavour,
        variable_config, tags):
    '''
    Alternative to compute_correlation_curves for a soft-tag-selected data sample:
    instead of reweighting data by an MC-estimated purity (compute_purity_weights,
    "method 2"), estimate the true-target_flavour data yield by subtracting the
    other flavours' MC-predicted contribution bin-by-bin, after normalizing the
    (truth-split) MC sample's overall soft-tag-selected yield to that observed in
    data with a single scale factor (same idea as the tag-and-probe normalization
    in calibration/calibration.py). This is "method 1" of Section 4.2.2 of
    Phys. Lett. B 401 (1997) 150.
    Input arguments:
    - data_events: data events, already restricted to the soft-tag selection.
    - mc_events_per_flavour: dict {flavour: events}, each already restricted to the
      SAME soft-tag selection, for all considered flavours (including target_flavour).
    - target_flavour: the flavour the soft tag is designed to select.
    - variable_config, tags: as elsewhere.
    Returns the same dict format as compute_correlation_curves, with an extra
    top-level 'norm_factor' entry recording the data/MC normalization used.
    '''
    v0, v1 = get_variable_values(data_events, variable_config)
    bins = variable_config['bins']
    mc_vs = {flavour: get_variable_values(events, variable_config)
             for flavour, events in mc_events_per_flavour.items()}

    # overall data/MC normalization factor (single scalar, matching the total
    # soft-tag-selected yield in data to the total predicted by simulation)
    bins_arr = np.asarray(bins, dtype=float)
    n_bin_per_flavour = get_pooled_bin_counts(mc_events_per_flavour, variable_config, bins_arr)
    n_data_total = len(v0) + len(v1)
    n_mc_total = sum(n.sum() for n in n_bin_per_flavour.values())
    norm_factor = n_data_total/n_mc_total if n_mc_total > 0 else 0.

    # raw per-bin sums, per tag, for data and for each MC flavour; then background-subtract
    signal_curves = {}
    for tag in tags:
        tag0 = data_events[f'Jet_mask_{tag}'][:, 0].to_numpy().astype(bool)
        tag1 = data_events[f'Jet_mask_{tag}'][:, 1].to_numpy().astype(bool)
        n_bin_data, sum_own_data, sum_oppo_data = compute_raw_bin_sums(v0, v1, tag0, tag1, bins)

        n_bin_bkg = np.zeros_like(n_bin_data)
        sum_own_bkg = np.zeros_like(sum_own_data)
        sum_oppo_bkg = np.zeros_like(sum_oppo_data)
        for flavour, mc_events in mc_events_per_flavour.items():
            if flavour == target_flavour: continue
            mv0, mv1 = mc_vs[flavour]
            mtag0 = mc_events[f'Jet_mask_{tag}'][:, 0].to_numpy().astype(bool)
            mtag1 = mc_events[f'Jet_mask_{tag}'][:, 1].to_numpy().astype(bool)
            n_bin_g, sum_own_g, sum_oppo_g = compute_raw_bin_sums(mv0, mv1, mtag0, mtag1, bins)
            n_bin_bkg += n_bin_g
            sum_own_bkg += sum_own_g
            sum_oppo_bkg += sum_oppo_g

        n_bin_signal = n_bin_data - norm_factor*n_bin_bkg
        sum_own_signal = sum_own_data - norm_factor*sum_own_bkg
        sum_oppo_signal = sum_oppo_data - norm_factor*sum_oppo_bkg
        # a negative predicted population is a sign of a statistical fluctuation in a
        # low-population bin; clip it to avoid ill-defined efficiencies there.
        n_bin_signal = np.clip(n_bin_signal, a_min=0, a_max=None)
        signal_curves[tag] = curves_from_raw_sums(n_bin_signal, sum_own_signal, sum_oppo_signal)

    # f_v and n_bin (background-subtracted population) are identical for every tag
    f_v = next(iter(signal_curves.values()))['f_v']
    n_bin = next(iter(signal_curves.values()))['n_bin']

    rho = {}
    for tagI, tagJ in itertools.combinations_with_replacement(tags, 2):
        rho[f'{tagI}_{tagJ}'] = correlation_from_curves(f_v, signal_curves[tagI], signal_curves[tagJ])

    return {
        'bins': list(bins),
        'f_v': f_v.tolist(),
        'n_bin': n_bin.tolist(),
        'curves': curves_to_json(signal_curves),
        'rho': rho,
        'norm_factor': norm_factor,
    }


if __name__=='__main__':

    # read command line args
    # (the sample/selection arguments below are forwarded as-is to
    #  rbanalysis.analysis.rbcalc.load_and_tag_events)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-d', '--data', required=True, nargs='+')
    parser.add_argument('-t', '--thresholds', required=True)
    parser.add_argument('--soft_thresholds',
      default=os.path.join(thisdir, 'tools', 'thresholds_soft.json'))
    parser.add_argument('--objectselection', default=None, nargs='+')
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--external_variables', default=None)
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--sqrts', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--variables', default=list(VARIABLES.keys()), nargs='+',
      choices=list(VARIABLES.keys()))
    parser.add_argument('-o', '--outputdir', default='output')
    parser.add_argument('-r', '--runmode', default='local', choices=['local', 'condor'])
    args = parser.parse_args()

    # if requested, submit this exact command (with runmode switched to 'local')
    # as a condor job instead of running locally
    if args.runmode == 'condor':
        env_script = os.path.join(topdir, 'setup.sh')
        ct.submitScriptAsCondorJob('cjob_correlation_uncertainty', sys.argv, env_script=env_script)
        sys.exit()

    # read events and assign tags (reusing rbcalc.py)
    events_combined, tags, cats = load_and_tag_events(args, extra_branches=EXTRA_BRANCHES)
    flavours = [p for p in events_combined['sim'].keys()]
    print(f'Found following truth flavours in simulation: {flavours}')

    # read soft tag thresholds
    with open(args.soft_thresholds, 'r') as f:
        soft_thresholds = json.load(f)
    print('Found following soft tag thresholds:')
    print(json.dumps(soft_thresholds, indent=2))

    # make output directory
    if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

    # resolve quantile-based bin edges per requested variable, from the pooled
    # (all truth flavours) simulated distribution
    resolved_variables = {}
    for variable_name in args.variables:
        variable_config = dict(VARIABLES[variable_name])
        n_bins = variable_config.pop('nbins')
        outlier_quantile = variable_config.pop('outlier_quantile', 0.)
        bins = compute_quantile_bins(events_combined['sim'], variable_config, n_bins,
                 outlier_quantile=outlier_quantile)
        variable_config['bins'] = list(bins)
        resolved_variables[variable_name] = variable_config
        print(f'Resolved bins for variable "{variable_name}": {list(bins)}')

    # diagnostic: distributions of the variables used, simulation only
    plot_variable_distributions(events_combined['sim'], resolved_variables, args.outputdir)

    # main loop over variables
    results = {}
    for variable_name in args.variables:
        print(f'Now processing variable "{variable_name}"...')
        variable_config = resolved_variables[variable_name]
        results[variable_name] = {'mc': {}, 'data_soft': {}, 'data_soft_subtracted': {}}
        results[variable_name]['_meta'] = {
            'axtitle': variable_config['axtitle'], 'unit': variable_config['unit'],
        }

        # MC side: straightforward, truth flavour is known exactly
        for flavour in flavours:
            print(f'  MC truth flavour {flavour}...')
            results[variable_name]['mc'][flavour] = compute_correlation_curves(
                events_combined['sim'][flavour], variable_config, tags)

        # data side: soft-tag flavour isolation + MC-purity reweighting
        control_data = {}
        for soft_flavour, cuts in soft_thresholds.items():
            if soft_flavour not in flavours:
                print(f'  WARNING: soft tag flavour "{soft_flavour}" not found among'
                      f' simulated flavours {flavours}; skipping.')
                continue
            print(f'  Data, soft tag for flavour {soft_flavour}...')

            # apply the soft tag to data and to every simulated flavour
            data_events = events_combined['data']['data']
            data_mask = build_soft_tag_mask(data_events, cuts)
            data_soft = data_events[data_mask]

            mc_soft = {}
            for flavour in flavours:
                mc_events = events_combined['sim'][flavour]
                mc_mask = build_soft_tag_mask(mc_events, cuts)
                mc_soft[flavour] = mc_events[mc_mask]
                n_pass = int(np.sum(mc_mask))
                print(f'    {flavour}: {n_pass} / {len(mc_mask)} pass the soft tag.')
            n_data_pass = int(np.sum(data_mask))
            print(f'    data: {n_data_pass} / {len(data_mask)} pass the soft tag.')

            if n_data_pass == 0:
                print(f'    WARNING: no data events pass the soft tag for {soft_flavour}; skipping.')
                continue

            # compute per-hemisphere MC-purity weights for the data sample
            v0, v1 = get_variable_values(data_soft, variable_config)
            w0, w1, purity_per_bin = compute_purity_weights(
                v0, v1, mc_soft, soft_flavour, variable_config)
            # compute_tag_curves expects one weight per event (applied to both its
            # hemispheres identically); since v0/v1-specific purity weights were
            # derived per-hemisphere above, recombine into a single symmetric
            # per-event weight by averaging the two hemispheres' purities
            event_weights = 0.5*(w0 + w1)

            results[variable_name]['data_soft'][soft_flavour] = compute_correlation_curves(
                data_soft, variable_config, tags, weights=event_weights)
            results[variable_name]['data_soft'][soft_flavour]['_purity_per_bin'] = (
                purity_per_bin.tolist())

            # alternative ("method 1"): bin-by-bin background subtraction instead of
            # purity reweighting
            results[variable_name]['data_soft_subtracted'][soft_flavour] = (
                compute_correlation_curves_subtracted(
                    data_soft, mc_soft, soft_flavour, variable_config, tags))

            # collect inputs for the grouped control plot (all soft tags side by
            # side), made once below after this loop over soft tags completes
            norm_factor = (
                results[variable_name]['data_soft_subtracted'][soft_flavour]['norm_factor'])
            control_data[soft_flavour] = {
                'data_events': data_soft, 'mc_events_per_flavour': mc_soft,
                'norm_factor': norm_factor,
            }

        # diagnostic: stacked-simulation-vs-data control plots for all soft tags,
        # using the same data/MC normalization the subtraction method above uses
        if len(control_data) > 0:
            plot_soft_tag_control_plots(control_data, variable_config, variable_name,
                args.outputdir)

    # write output
    outputfile = os.path.join(args.outputdir, 'correlation_curves.json')
    with open(outputfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote results to {outputfile}.')

    # diagnostic plots: own/opposite hemisphere efficiency curves, and a summary of
    # the aggregate correlation per tag pair, MC vs. data-driven, for each variable
    for variable_name in args.variables:
        plot_curves(results, variable_name, args.outputdir, tags)
        plot_curves(results, variable_name, args.outputdir, tags, mc_only=True)
        plot_rho_summary(results, variable_name, args.outputdir, tags)
        plot_rho_summary(results, variable_name, args.outputdir, tags, mc_only=True)
