# Plot score distribution and ROC for multiple processes

import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def plot_scores_multi(categories, scores, labels,
        outputdir = None):

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # loop over different scores to plot
    for score_name, score_branch in categories.items():
        score_branch = score_branch['score']

        # retrieve score
        this_scores = scores[score_branch]

        # initialize figure
        fig, ax = plt.subplots()

        # loop over categories
        for category_name, category_settings in categories.items():
            cat_mask = labels[category_name]

            # get scores
            this_values = this_scores[cat_mask]
            
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


def plot_roc_multi(categories, scores, labels,
        outputdir = None):

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # initialize figure
    fig, ax = plt.subplots()
    nlines = int(len(categories)*(len(categories)-1)/2)
    cmap = plt.get_cmap('cool', nlines)
    cidx = 0

    # loop over pairs of categories
    for sidx, (signal_category_name, signal_category_settings) in enumerate(categories.items()):
        for bidx, (background_category_name, background_category_settings) in enumerate(categories.items()):
                if bidx <= sidx: continue

                # get scores for signal and background
                sig_score_branch = signal_category_settings['score']
                bkg_score_branch = background_category_settings['score']

                # binarization
                #this_scores = np.divide(scores[sig_score_branch],
                #                 scores[sig_score_branch] + scores[bkg_score_branch])
                # alternative: just use signal score
                this_scores = scores[sig_score_branch]

                this_scores = np.nan_to_num(this_scores, nan=0, posinf=0, neginf=0)
                scores_sig = this_scores[labels[signal_category_name]]
                scores_bkg = this_scores[labels[background_category_name]]
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
                    np.linspace(np.amax(this_scores)*0.9, np.amax(this_scores), num=500),
                ))
                efficiency_sig = np.zeros(len(thresholds))
                efficiency_bkg = np.zeros(len(thresholds))
                for idx, threshold in enumerate(thresholds):
                    w_sig = weights_sig[scores_sig > threshold]
                    efficiency_sig[idx] = np.sum(w_sig)
                    w_bkg = weights_bkg[scores_bkg > threshold]
                    efficiency_bkg[idx] = np.sum(w_bkg)
                efficiency_sig /= np.sum(weights_sig)
                efficiency_bkg /= np.sum(weights_bkg)

                # make a plot of the ROC curve
                label = signal_category_settings['label'] + ' vs. '
                label += background_category_settings['label']
                label += ' (AUC: {:.2f})'.format(auc)
                ax.plot(efficiency_bkg, efficiency_sig,
                  color=cmap(cidx), linewidth=3, label=label)
                cidx += 1
    
    # other plot settings
    dummy_efficiency = np.linspace(0, 1, num=101)
    ax.plot(dummy_efficiency, dummy_efficiency,
      color='darkblue', linewidth=3, linestyle='--')
    ax.set_xlabel('Background pass-through', fontsize=12)
    ax.set_ylabel('Signal efficiency', fontsize=12)
    ax.grid(which='both')
    leg = ax.legend()

    # save figure
    fig.tight_layout()
    figname = os.path.join(outputdir, 'roc.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')

    # same with log scale on x-axis
    ax.set_xscale('log')
    ax.set_xlim((1e-5, 1))
    fig.tight_layout()
    figname = os.path.join(outputdir, 'roc_log.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')
    plt.close()
