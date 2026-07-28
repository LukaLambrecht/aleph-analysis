# Post-hoc plotting of the output of correlation_uncertainty.py.
#
# Two kinds of diagnostic plots are made per variable:
# - "curves": for each truth flavour, a 1xN grid (one panel per tag) showing the
#   own-hemisphere and opposite-hemisphere tagging efficiency as a function of the
#   variable (both from MC truth, and from the soft-tag-selected, purity-reweighted
#   data sample where available). A flat eps_oppo(v) means no detected correlation
#   via that variable; structure in it is the correlation signal itself.
# - "rho_summary": a bar chart of the aggregate (bin-integrated) correlation value
#   per tag pair, comparing MC to the data-driven estimate, for a given flavour.

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
sys.path.append(os.path.join(thisdir, 'tools'))

from correlationtools import binomial_error


def plot_curves(results, variable_name, outputdir, tags, mc_only=False):
    '''
    Make a grid of eps_own / eps_oppo curves, one row per truth flavour, one column
    per tag, for a given variable. Curves are drawn as step functions (matplotlib
    stairs) over the actual bin edges, with a semi-transparent band showing a rough
    binomial statistical uncertainty.
    If mc_only is True, only the simulation curves are drawn (no data-driven
    overlays), and the figure is saved under a separate 'curves_mc_...' name; useful
    as a simpler, didactic first look before comparing to the data-driven estimates.
    '''
    variable_config = results[variable_name]['_meta']
    mc_results = results[variable_name]['mc']
    data_results = {} if mc_only else results[variable_name].get('data_soft', {})
    data_sub_results = {} if mc_only else results[variable_name].get('data_soft_subtracted', {})
    flavours = list(mc_results.keys())
    nrows, ncols = len(flavours), len(tags)

    def draw_stairs(ax, values, err, bins, color, **kwargs):
        ax.stairs(values, edges=bins, color=color, linewidth=1.5, **kwargs)
        ax.stairs(values+err, baseline=values-err, edges=bins, fill=True, color=color, alpha=0.2)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2*ncols, 3*nrows), squeeze=False)

    for row_idx, flavour in enumerate(flavours):
        for col_idx, tag in enumerate(tags):
            ax = axs[row_idx, col_idx]
            # each tag's own/opposite curve is a property of that tag alone (not of a
            # pair), stored once; just look it up directly
            entry = mc_results[flavour]
            bins = np.array(entry['bins'])
            curve = entry['curves'][tag]
            n_bin = np.array(entry['n_bin'])
            own = np.array(curve['eps_own'])
            oppo = np.array(curve['eps_oppo'])
            mean_eff = curve['mean_eff']

            ax.axhline(y=mean_eff, color='grey', linestyle='dashed', linewidth=1)
            draw_stairs(ax, own, binomial_error(own, n_bin), bins, 'limegreen',
                        label='MC, same hemisphere')
            draw_stairs(ax, oppo, binomial_error(oppo, n_bin), bins, 'red',
                        label='MC, opposite hemisphere')

            has_purity = flavour in data_results
            has_subtracted = flavour in data_sub_results

            if has_purity:
                dentry = data_results[flavour]
                dn_bin = np.array(dentry['n_bin'])
                dcurve = dentry['curves'][tag]
                dsame = np.array(dcurve['eps_own'])
                doppo = np.array(dcurve['eps_oppo'])

            if has_subtracted:
                sentry = data_sub_results[flavour]
                sn_bin = np.array(sentry['n_bin'])
                scurve = sentry['curves'][tag]
                ssame = np.array(scurve['eps_own'])
                soppo = np.array(scurve['eps_oppo'])

            # draw grouped by same/opposite (rather than by method), so the two
            # data-driven methods can be compared directly within each group
            if has_purity:
                draw_stairs(ax, dsame, binomial_error(dsame, dn_bin), bins, 'deepskyblue',
                            label='data (purity-reweighted), same')
            if has_subtracted:
                draw_stairs(ax, ssame, binomial_error(ssame, sn_bin), bins, 'skyblue',
                            linestyle='dotted', label='data (background-subtracted), same')
            if has_purity:
                draw_stairs(ax, doppo, binomial_error(doppo, dn_bin), bins, 'darkviolet',
                            label='data (purity-reweighted), opposite')
            if has_subtracted:
                draw_stairs(ax, soppo, binomial_error(soppo, sn_bin), bins, 'violet',
                            linestyle='dotted', label='data (background-subtracted), opposite')

            if row_idx == 0: ax.set_title(tag, fontsize=14)
            if col_idx == 0: ax.set_ylabel(f'{flavour}\nefficiency', fontsize=11)
            if row_idx == nrows-1:
                unit = variable_config['unit']
                xlabel = variable_config['axtitle']
                if len(unit) > 0: xlabel += f' [{unit}]'
                ax.set_xlabel(xlabel, fontsize=11)
            ax.tick_params(labelsize=9)

            if row_idx == 0 and col_idx == ncols-1:
                ax.legend(fontsize=13, loc='upper left', bbox_to_anchor=(1.02, 1.0))

    suptitle = f'Same vs. opposite hemisphere efficiency vs. {variable_name}'
    if mc_only: suptitle += ' (simulation only)'
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    figname = f'curves_mc_{variable_name}.png' if mc_only else f'curves_{variable_name}.png'
    figpath = os.path.join(outputdir, figname)
    fig.savefig(figpath, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {figpath}.')


def plot_rho_summary(results, variable_name, outputdir, tags, mc_only=False):
    '''
    Make a bar chart comparing the aggregate (bin-integrated) rho value from MC and
    from the data-driven estimate, for every tag pair, one panel per truth flavour.
    If mc_only is True, only the MC bars are drawn (no data-driven overlays), and the
    figure is saved under a separate 'rho_summary_mc_...' name; useful as a simpler,
    didactic first look before comparing to the data-driven estimates.
    '''
    mc_results = results[variable_name]['mc']
    data_results = {} if mc_only else results[variable_name].get('data_soft', {})
    data_sub_results = {} if mc_only else results[variable_name].get('data_soft_subtracted', {})
    flavours = list(mc_results.keys())
    pairs = [f'{tagI}_{tagJ}' for tagI, tagJ in itertools.combinations_with_replacement(tags, 2)]

    fig, axs = plt.subplots(nrows=len(flavours), ncols=1, figsize=(10, 3*len(flavours)), squeeze=False)
    for row_idx, flavour in enumerate(flavours):
        ax = axs[row_idx, 0]
        mc_vals = [mc_results[flavour]['rho'][p] for p in pairs]
        x = np.arange(len(pairs))
        width = 0.25
        mc_x = x if mc_only else x - width
        mc_width = 2*width if mc_only else width
        ax.bar(mc_x, mc_vals, width=mc_width, color='grey', label='MC')
        if flavour in data_results:
            data_vals = [data_results[flavour]['rho'][p] for p in pairs]
            ax.bar(x, data_vals, width=width, color='orange', label='data (purity-reweighted)')
        if flavour in data_sub_results:
            sub_vals = [data_sub_results[flavour]['rho'][p] for p in pairs]
            ax.bar(x + width, sub_vals, width=width, color='blue', label='data (background-subtracted)')
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xticks(x, labels=pairs, rotation=90, fontsize=8)
        ax.set_ylabel(f'{flavour}\nrho', fontsize=11)
        if row_idx == 0: ax.legend(fontsize=9)

    suptitle = f'Aggregate correlation per tag pair, variable: {variable_name}'
    if mc_only: suptitle += ' (simulation only)'
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    figname = f'rho_summary_mc_{variable_name}.png' if mc_only else f'rho_summary_{variable_name}.png'
    figpath = os.path.join(outputdir, figname)
    fig.savefig(figpath, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {figpath}.')


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True,
      help='Path to a correlation_curves.json file produced by correlation_uncertainty.py')
    parser.add_argument('-o', '--outputdir', required=True)
    args = parser.parse_args()

    with open(args.inputfile, 'r') as f:
        results = json.load(f)

    if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

    for variable_name in results.keys():
        # recover the tag list from the first available flavour's curves
        first_flavour = next(iter(results[variable_name]['mc'].keys()))
        tags = list(results[variable_name]['mc'][first_flavour]['curves'].keys())

        plot_curves(results, variable_name, args.outputdir, tags)
        plot_curves(results, variable_name, args.outputdir, tags, mc_only=True)
        plot_rho_summary(results, variable_name, args.outputdir, tags)
        plot_rho_summary(results, variable_name, args.outputdir, tags, mc_only=True)
