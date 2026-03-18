import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__=='__main__':

    # set output dir 
    outputdir = '.'

    # read file
    df = pd.read_csv('plot-data.csv')
    x = df['x'].values
    y = df[' y'].values

    # split in b-distribution and background distribution
    sig_counts = y[0::2]
    bkg_counts = y[1::2] - sig_counts

    # calculate signal and background efficiency
    efficiency_sig = 1 - np.cumsum(sig_counts) / np.sum(sig_counts)
    efficiency_bkg = 1 - np.cumsum(bkg_counts) / np.sum(bkg_counts)

    # printouts for testing
    print('Signal efficiency:')
    print(efficiency_sig)
    print('Background efficiency:')
    print(efficiency_bkg)

    # write to csv file
    data = {'sig_eff': efficiency_sig, 'bkg_eff': efficiency_bkg}
    outdf = pd.DataFrame(data)
    outdf.to_csv('roc.csv')

    # make plot of score distribution
    fig, ax = plt.subplots()
    ax.stairs(sig_counts, label='Signal', color='g', linewidth=2)
    ax.stairs(bkg_counts, label='Background', color='r', linewidth=2)
    ax.stairs(sig_counts + bkg_counts, label='Sum', color='k', linewidth=1, linestyle='--')

    # plot aesthetics
    ax.grid(which='both', axis='both')
    ax.legend(fontsize=12)

    # save figure
    fig.tight_layout()
    figname = os.path.join(outputdir, 'score_dist.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')

    # make a plot of the ROC curve
    fig, ax = plt.subplots()
    ax.scatter(efficiency_bkg, efficiency_sig, s=10,
        color='dodgerblue', linewidth=3)

    # other plot settings
    dummy_efficiency = np.linspace(0, 1, num=101)
    ax.plot(dummy_efficiency, dummy_efficiency,
      color='darkblue', linewidth=3, linestyle='--')
    ax.set_xlabel('Background pass-through', fontsize=12)
    ax.set_ylabel('Signal efficiency', fontsize=12)
    ax.grid(which='both')
    #leg = ax.legend()

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
