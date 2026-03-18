# Plot score distribution and ROC for multiple processes

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def format_table_txt(table, colwidth=15, firstcolwidth=30):
    colfmtstr = '{0: <' + str(colwidth) + '}'
    firstcolfmtstr = '{0: <' + str(firstcolwidth) + '}'

    header = firstcolfmtstr.format('Signal efficiency:')
    for sig_eff in table['sig_effs']: header += colfmtstr.format(sig_eff)
    length = len(header)
    divider = '-'*length
    lines = []
    lines.append(divider)
    for key, val in table.items():
        if key=='sig_effs': continue
        label = key.replace('$', '').replace('\\', '')
        line = firstcolfmtstr.format(label)
        for el in val:
            elstr = str(el)
            if el > 0.01: elstr = '{:.3f}'.format(el)
            else: elstr = '{:.3e}'.format(el)
            line += colfmtstr.format(elstr)
        lines.append(line)
    lines.append(divider)
    txt = '\n'.join([header] + lines)
    return txt

def format_table_txt_latex(table, colwidth=30, firstcolwidth=30):
    colfmtstr = '{0: <' + str(colwidth) + '}'
    firstcolfmtstr = '{0: <' + str(firstcolwidth) + '}'

    header = firstcolfmtstr.format('Signal efficiency:')
    for sig_eff in table['sig_effs']: header += colfmtstr.format(sig_eff)
    length = len(header)
    divider = '-'*length
    lines = []
    lines.append(divider)
    for key, val in table.items():
        if key=='sig_effs': continue
        label = key.replace('-', ' ')
        label = label.replace('$b$', r'\PQb')
        label = label.replace('$c$', r'\PQc')
        label = label.replace('$s$', r'\PQs')
        label = label.replace('$ud$', r'$\PQu\PQd$')
        label = label.replace('$uds$', r'$\PQu\PQd\PQs$')
        line = firstcolfmtstr.format(label + ' &')
        for el in val:
            elstr = str(el)
            if el > 0.01: elstr = '{:.2g}'.format(el)
            else:
                elstr = '{:.1e}'.format(el)
                elstr = '$ ' + elstr.replace('e-0', 'e-').replace('e', r'\times 10^{') + '} $'
            line += colfmtstr.format(elstr + ' &')
        line = line.strip(' &') + r' \\'
        lines.append(line)
    lines.append(divider)
    txt = '\n'.join([header] + lines)
    return txt


def plot_scores_multi(events,
            categories,
            outputdir = None):

    # get mask for each category
    cat_masks = {}
    for category_name, category_settings in categories.items():
        branch = category_settings['label_branch']
        mask = events[branch].astype(bool)
        cat_masks[category_name] = mask

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # loop over different scores to plot
    for score_name, score_branch in categories.items():
        score_branch = score_branch['score_branch']

        # retrieve score
        scores = events[score_branch]

        # initialize figure
        fig, ax = plt.subplots()

        # loop over categories
        for category_name, category_settings in categories.items():
            cat_mask = cat_masks[category_name]

            # get scores
            this_values = scores[cat_mask]
            
            # make a histogram
            label = category_settings['label']
            bins = np.linspace(0, 1, num=41)
            hist = np.histogram(this_values, bins=bins)[0]
            norm = np.sum( np.multiply(hist, np.diff(bins) ) )
            staterrors = np.sqrt(np.histogram(this_values, bins=bins)[0])
            ax.stairs(hist/norm, edges=bins,
                  color = category_settings['color'],
                  label = label,
                  linewidth=2)
            ax.stairs((hist+staterrors)/norm, baseline=(hist-staterrors)/norm,
                        color = category_settings['color'],
                        edges=bins, fill=True, alpha=0.15)
        
        ax.set_xlabel(f'Classifier output score ({score_name})', fontsize=12)
        ax.set_ylabel('Events (normalized)', fontsize=12)
        ax.set_title(f'Score distribution', fontsize=12)
        ylim_default= ax.get_ylim()
        ax.set_ylim((0., ylim_default[1]*1.3))
        leg = ax.legend(fontsize=10)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
            lh._sizes = [30]
        fig.tight_layout()
        figname = os.path.join(outputdir, f'{score_branch}.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')
    
        # same with log scale
        ax.autoscale()
        ax.set_yscale('log')
        fig.tight_layout()
        figname = os.path.join(outputdir, f'{score_branch}_log.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')
        plt.close()


def make_roc_curves(events,
            signal_categories,
            background_categories,
            do_bootstrap = False):

    # check arguments
    all_categories = {**signal_categories, **background_categories}

    # get mask for each category
    masks = {}
    for category_name, category_settings in all_categories.items():
        branch = category_settings['label_branch']
        if '|' in branch:
            # simple ad-hoc parsing of complex labels
            parts = [part.strip(' ') for part in branch.split('|')]
            mask = np.zeros(len(events[parts[0]])).astype(bool)
            for part in parts: mask = ((mask) | (events[part].astype(bool)))
        else:
            mask = events[branch].astype(bool)
        masks[category_name] = mask

    # loop over pairs of categories
    # update: loop over all pairs, not just signal vs background
    roc_curves = {}
    aucs = {}
    for sidx, (signal_category_name, signal_category_settings) in enumerate(all_categories.items()):
        for bidx, (background_category_name, background_category_settings) in enumerate(all_categories.items()):
                if bidx <= sidx: continue

                # get scores for signal and background
                sig_score_branch = signal_category_settings['score_branch']
                bkg_score_branch = background_category_settings['score_branch']

                # do binarization
                #scores = np.divide(events[sig_score_branch], events[sig_score_branch] + events[bkg_score_branch])
                # alternative: just use the signal score
                scores = events[sig_score_branch]

                # make scores and weights for signal and background
                scores_sig = scores[masks[signal_category_name]]
                scores_bkg = scores[masks[background_category_name]]
                weights_sig = np.ones(len(scores_sig))
                weights_bkg = np.ones(len(scores_bkg))

                # safety for no passing events
                if len(scores_sig)==0 or len(scores_bkg)==0:
                    continue

                # calculate AUC
                this_scores = np.concatenate((scores_sig, scores_bkg))
                this_weights = np.concatenate((weights_sig, weights_bkg))
                this_labels = np.concatenate((np.ones(len(scores_sig)), np.zeros(len(scores_bkg))))
                auc = roc_auc_score(this_labels, this_scores, sample_weight=np.abs(this_weights))

                # calculate signal and background efficiency
                thresholds = np.concatenate((
                    np.linspace(np.amin(this_scores), np.amax(this_scores)*0.9, num=100),
                    np.linspace(np.amax(this_scores)*0.9, np.amax(this_scores)*0.98, num=300),
                    np.linspace(np.amax(this_scores)*0.98, np.amax(this_scores), num=300)
                ))

                def roc_curve(scores, weights, labels, thresholds):
                    scores_sig = scores[labels==1]
                    scores_bkg = scores[labels==0]
                    weights_sig = weights[labels==1]
                    weights_bkg = weights[labels==0]
                    efficiency_sig = np.zeros(len(thresholds))
                    efficiency_bkg = np.zeros(len(thresholds))
                    for idx, threshold in enumerate(thresholds):
                        eff_s = np.sum(weights_sig[scores_sig > threshold])
                        efficiency_sig[idx] = eff_s
                        eff_b = np.sum(weights_bkg[scores_bkg > threshold])
                        efficiency_bkg[idx] = eff_b
                    efficiency_sig /= np.sum(weights_sig)
                    efficiency_bkg /= np.sum(weights_bkg)
                    return (efficiency_sig, efficiency_bkg)

                # make roc curve
                efficiency_sig, efficiency_bkg = roc_curve(this_scores, this_weights, this_labels, thresholds)

                # new: add uncertainty from bootstrapping
                eff_s_lo = None
                eff_s_hi = None
                if do_bootstrap:
                    n_bootstrap = 100
                    effs_s = []
                    for _ in range(n_bootstrap):
                        idx = np.random.randint(0, len(this_scores), len(this_scores))
                        eff_s, eff_b = roc_curve(this_scores[idx], this_weights[idx], this_labels[idx], thresholds)
                        this_eff_s = np.interp(efficiency_bkg[::-1], eff_b[::-1], eff_s[::-1])
                        effs_s.append(this_eff_s[::-1])
                    effs_s = np.array(effs_s)
                    eff_s_med = np.median(effs_s, axis=0)
                    eff_s_lo = np.percentile(effs_s, 16, axis=0)
                    eff_s_hi = np.percentile(effs_s, 84, axis=0)

                # add to dict
                key = (signal_category_name, background_category_name)
                val = (efficiency_sig, efficiency_bkg, eff_s_lo, eff_s_hi)
                roc_curves[key] = val
                aucs[key] = auc

    return (roc_curves, aucs)

    
def plot_roc_multi(events,
            signal_categories,
            background_categories,
            outputdir = None,
            doRb = False,
            doAFB = False,
            do_bootstrap = False):

    # check arguments
    all_categories = {**signal_categories, **background_categories}

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # initialize figure
    fig, ax = plt.subplots()

    # initialize colors (automatic)
    nlines = int(len(all_categories)*(len(all_categories)-1)/2)
    cmap = plt.get_cmap('cool', nlines)
    cidx = 0

    # initialize colors (ad hoc, hard-coded)
    cmap = {}
    cmap[('b', 'c')] = 'blueviolet'
    cmap[('b', 'uds')] = 'crimson'
    cmap[('b', 'udsc')] = 'mediumvioletred'
    cmap[('c', 'uds')] = 'dodgerblue'

    # initialize a table
    table = {}
    table['sig_effs'] = [0.2, 0.4, 0.6, 0.8]

    # make roc curves
    roc_curves, aucs = make_roc_curves(events, signal_categories, background_categories, do_bootstrap=do_bootstrap)

    # loop over pairs of categories
    # update: loop over all pairs, not just signal vs background
    for sidx, (signal_category_name, signal_category_settings) in enumerate(all_categories.items()):
        for bidx, (background_category_name, background_category_settings) in enumerate(all_categories.items()):
                if bidx <= sidx: continue
               
                # get roc curve 
                key = (signal_category_name, background_category_name)
                efficiency_sig, efficiency_bkg, eff_s_lo, eff_s_hi = roc_curves[key]
                auc = aucs[key]
                
                # make a plot of the ROC curve
                label = signal_category_settings['label'] + ' vs. '
                label += background_category_settings['label']
                label += ' (AUC: {:.2f})'.format(auc)
                #color = cmap(cids)
                color = cmap[(signal_category_name, background_category_name)]
                ax.plot(efficiency_bkg, efficiency_sig,
                  color=color, linewidth=3, label=label)
                cidx += 1

                # experimental: add uncertainty
                if eff_s_lo is not None and eff_s_hi is not None:
                    ax.fill_between(efficiency_bkg, eff_s_lo, eff_s_hi,
                        color=color, alpha=0.3)

                # make a table entry
                table_entry = []
                for sig_eff in table['sig_effs']:
                    idx = np.nonzero(efficiency_sig[::-1] > sig_eff)[0][0]
                    bkg_eff = efficiency_bkg[::-1][idx]
                    table_entry.append(bkg_eff)
                label = signal_category_settings['label'] + ' vs. '
                label += background_category_settings['label']
                table[label] = table_entry

    # ad-hoc addition (maybe clean up later):
    # add Rb reference
    if doRb:
        ref = r'\textit{Phys. Lett. B} \textbf{401} (1997) 163-175'
        ax.scatter(0.00216, 0.1957, s=40, color=cmap[('b', 'c')], edgecolor=None, label=f'$b$-jets vs. $c$-jets ({ref})')
        ax.scatter(0.00043, 0.1957, s=40, color=cmap[('b', 'uds')], edgecolors=None, label=f'$b$-jets vs. $uds$-jets ({ref})')

    # ad-hoc addition (maybe clean up later):
    # add A_FB reference
    if doAFB:
        # read file (hard-coded for now)
        filepath = '../../purity/digitized_roc_curve/roc.csv'
        df = pd.read_csv(filepath)
        sig_eff = df['sig_eff'].values[:-1]
        bkg_eff = df['bkg_eff'].values[:-1]
        ref = r'\textit{Eur. Phys. J. C} \textbf{22} (2001) 201-215'
        ax.plot(bkg_eff, sig_eff, color=cmap[('b', 'udsc')], linewidth=2, linestyle='dotted', label=f'$b$-jets vs. $udsc$-jets ({ref})')

    # add random guessing line
    dummy_efficiency = np.linspace(0, 1, num=101)
    ax.plot(dummy_efficiency, dummy_efficiency,
      color='darkblue', linewidth=3, linestyle='--', label='Random guessing')

    # add aleph logo
    docms = True
    extracmstext = 'Archived Sim.'
    if docms:
        cmstext = r'$\bf{ALEPH}$'
        if extracmstext is not None:
            for part in extracmstext.split(' '): cmstext += r' $\it{' + f' {part}' + r'}$'
        cmstext_in_box = True # maybe later add as argument
        if cmstext_in_box:
            text = ax.text(0.02, 0.98, cmstext,
                    ha='left', va='top', fontsize=20, transform=ax.transAxes)
            text.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
            # modify the axis range to accommodate the CMS text
            #yscale = ax.get_ylim()[1] - ax.get_ylim()[0]
            #ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + yscale*0.2)
        else:
            ax.text(0., 1., cmstext,
                    ha='left', va='bottom', fontsize=20, transform=ax.transAxes)

    # other plot settings
    ax.set_xlabel('Background pass-through', fontsize=22)
    ax.set_ylabel('Signal efficiency', fontsize=22)
    ax.tick_params(axis='both', labelsize=17)
    ax.grid(which='both')
    ax.set_ylim((-0.05, 1.1))
    legend_in_box = False # maybe later add as argument
    if legend_in_box: leg = ax.legend(fontsize=17)
    else:
        leg = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=17)
        #w, h = fig.get_size_inches()
        #fig.set_size_inches(w*1.75, h, forward=True)

    # save the figure
    figname = os.path.join(outputdir, 'roc.png')
    fig.savefig(figname, bbox_extra_artists=(leg,), bbox_inches='tight')
    fig.savefig(figname.replace('.png', '.pdf'), bbox_extra_artists=(leg,), bbox_inches='tight')
    print(f'Saved figure {figname}.')

    # same with log scale on x-axis
    ax.set_xscale('log')
    ax.set_xlim((1e-5, 1))
    figname = os.path.join(outputdir, 'roc_log.png')
    fig.savefig(figname, bbox_extra_artists=(leg,), bbox_inches='tight')
    fig.savefig(figname.replace('.png', '.pdf'), bbox_extra_artists=(leg,), bbox_inches='tight')
    print(f'Saved figure {figname}.')
    plt.close()

    # print table
    print('Results table:')
    table_txt = format_table_txt(table)
    print(table_txt)
    table_txt_latex = format_table_txt_latex(table)
    print(table_txt_latex)

    # store table to json and file
    filename = os.path.join(outputdir, 'table.json')
    with open(filename, 'w') as f:
        json.dump(table, f, indent=2)
    filename = os.path.join(outputdir, 'table.txt')
    with open(filename, 'w') as f:
        f.write(table_txt)
