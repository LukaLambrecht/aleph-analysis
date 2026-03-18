import os
import sys
import json
#import mplhep
import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.plottools import merge_events
from tools.plottools import make_hist
from tools.variabletools import read_variables
from tools.variabletools import HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.eventselection import get_selection_mask

# local version of mplhep
import plotting.mplhep as mplhep

def plot(sig=None, bkg=None, data=None,
         systematics=None,
         variable=None,
         fig=None, ax=None,
         stacklist=None,
         colordict=None, labeldict=None, styledict=None, multdict=None,
         normalize=False, normalizesum=False, normalizesim=False, logscale=False,
         docms=True, extracmstext=None, lumiheader=None,
         yaxtitle=None, dolegend=False,
         secondary_bin_label_height=0.7,
         ratios=None, ratio_yaxtitles=None):
    '''
    Make prediction vs data histogram plot
    Input arguments:
    - sig and bkg: dicts of the form {<label>: (np array, np array), ...},
      where the first array in the tuple represents the bin contents,
      and the second one the (statistical) uncertainties.
    - data: dict of the same form as sig and bkg, but with only one item.
    - systematics: dict of the same form as sig and bkg, but where the second item
      in each tuple represents the total systematic uncertainty for the given process.
    - variable: an instance of type HistogramVariable or DoubleHistogramVariable.
    - ratios: list specifying which ratio plots to make.
              should be of the following form:
              [ [numerator, denominator], [...] ]
              a separate ratio pad will be made for each numerator/denominator combo.
    '''

    # concatenate dicts of sig and bkg histograms
    # (distinction between sig and bkg is not used in this function for now)
    all_hists = {}
    if bkg is not None:
        for key, val in bkg.items(): all_hists[key] = deepcopy(val)
    if sig is not None:
        for key, val in sig.items(): all_hists[key] = deepcopy(val)

    # split data tuple in actual histogram and uncertainties,
    # and some other data properties
    if data is not None:
        if len(data)!=1:
            msg = f'Unexptected number of items in data dict: {data}'
            raise Exception(msg)
        data_key = list(data.keys())[0]
        data_hist = np.copy(data[data_key][0])
        data_staterrors = np.copy(data[data_key][1])
        data_label = labeldict.get(data_key, 'Data')
        data_markersize = 4

    # make lists for sig and bkg histograms
    # and corresponding settings
    hists = [all_hists[key][0] for key in all_hists.keys()]
    staterrors = [all_hists[key][1] for key in all_hists.keys()]
    systerrors = []
    for key in all_hists.keys():
        if systematics is not None and key in systematics.keys(): systerrors.append(systematics[key][1])
        else: systerrors.append(np.zeros(hists[0].shape))
    stack_ids = []
    nostack_ids = list(range(len(hists)))
    if stacklist is not None:
        stack_ids = [idx for idx, key in enumerate(all_hists.keys()) if key in stacklist]
        nostack_ids = [idx for idx, key in enumerate(all_hists.keys()) if key not in stacklist]
    colors = [None]*len(hists)
    if colordict is not None:
        colors = [colordict.get(key, 'grey') for key in all_hists.keys()]
    labels = [None]*len(hists)
    if labeldict is not None:
        labels = [labeldict.get(key, '') for key in all_hists.keys()]
    styles = ['step']*len(hists)
    if styledict is not None:
        styles = [styledict.get(key, 'step') for key in all_hists.keys()]
    mults = [1]*len(hists)
    if multdict is not None:
        mults = [multdict.get(key, None) for key in all_hists.keys()]

    # calculate bin widths
    binwidths = None
    if variable is not None:
        if isinstance(variable, DoubleHistogramVariable):
            # just use the primary bin widths on repeat,
            # i.e. the secondary variable is treated as a categorical variable;
            # anyway it shouldn't really matter as long as this is only used for normalization.
            binwidths = variable.primary.bins[1:]-variable.primary.bins[:-1]
            binwidths = np.repeat(binwidths, variable.secondary.nbins)
        else: binwidths = variable.bins[1:]-variable.bins[:-1]

    # get bin edges and bin centers
    bin_edges = None
    bin_centers = None
    if variable is not None:
        bin_edges = variable.bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # for stacked histograms, calculate nominal sum and total error
    histsum = np.zeros(hists[0].shape)
    staterrorsum = np.zeros(hists[0].shape)
    systerrorsum = np.zeros(hists[0].shape)
    for idx in stack_ids:
        histsum += hists[idx]
        staterrorsum += np.square(staterrors[idx])
        systerrorsum += np.square(systerrors[idx])
    staterrorsum = np.sqrt(staterrorsum)
    systerrorsum = np.sqrt(systerrorsum)

    # normalize to unit surface area or unit sum if requested
    # note: default (with argument normalize) is to normalize to unit surface area,
    #       taking into account the bin widths (if provided);
    #       alternatively (with argument normalizesum), normalize to unit sum of bin contents,
    #       ignoring bin widths.
    #       if no bin widths are provided, normalize and normalizesum have the same effect.
    if( normalize or normalizesum ):
        # data
        if data is not None:
            integral = np.sum(data_hist)
            if( not normalizesum and binwidths is not None ):
                integral = np.sum( np.multiply(data_hist, binwidths) )
            if integral > 0:
                data_hist /= integral
                data_staterrors /= integral
        # predictions
        for idx, (hist, staterror, systerror) in enumerate(zip(hists, staterrors, systerrors)):
            # for non-stacked histograms, the normalization factor
            # is just the integral of the original histogram
            if idx not in stack_ids:
                integral = np.sum(hist)
                if( not normalizesum and binwidths is not None ):
                    integral = np.sum( np.multiply(hist, binwidths) )
            # for stacked histograms, the normalization factor
            # is the integral of the sum of stacked histograms
            else:
                integral = np.sum(histsum)
                if( not normalizesum and binwidths is not None ):
                    integral = np.sum( np.multiply(histsum, binwidths) )
            if integral > 0:
                hists[idx] /= integral
                staterrors[idx] /= integral
                systerrors[idx] /= integral
        # sum of stacked histograms
        integral = np.sum(histsum)
        if( not normalizesum and binwidths is not None ):
            integral = np.sum( np.multiply(histsum, binwidths) )
        if integral > 0:
            histsum /= integral
            staterrorsum /= integral
            systerrorsum /= integral

    # normalize sim to data if requested
    if( normalizesim ):
        # data
        if data is None: raise Exception('Option normalizesim is not well defined without data.')
        data_integral = np.sum(data_hist)
        if( binwidths is not None ):
            data_integral = np.sum( np.multiply(data_hist, binwidths) )
        # predictions
        for idx, (hist, staterror, systerror) in enumerate(zip(hists, staterrors, systerrors)):
            # for non-stacked histograms, the normalization factor
            # is just the integral of the original histogram
            if idx not in stack_ids:
                integral = np.sum(hist)
                if( binwidths is not None ):
                    integral = np.sum( np.multiply(hist, binwidths) )
            # for stacked histograms, the normalization factor
            # is the integral of the sum of stacked histograms
            else:
                integral = np.sum(histsum)
                if( binwidths is not None ):
                    integral = np.sum( np.multiply(histsum, binwidths) )
            if integral > 0 and data_integral > 0:
                hists[idx] /= (integral / data_integral)
                staterrors[idx] /= (integral / data_integral)
                systerrors[idx] /= (integral / data_integral)
        # sum of stacked histograms
        integral = np.sum(histsum)
        if( binwidths is not None ):
            integral = np.sum( np.multiply(histsum, binwidths) )
        if integral > 0 and data_integral > 0:
            histsum /= (integral / data_integral)
            staterrorsum /= (integral / data_integral)
            systerrorsum /= (integral / data_integral)

    # check settings for multiplicative factors
    if multdict is not None:
        for key, val in multdict.items():
            if val is None or val==1.: continue
            if key in stacklist:
                msg = 'Multiplicative factors have only been implemented for non-stacked histograms so far.'
                raise Exception(msg)

    # make the base figure
    if fig is None or ax is None:
        if ratios is None or len(ratios)==0:
            fig, ax = plt.subplots(figsize=(8,6))
            axs = [ax]
        else:
            nratiopads = len(ratios)
            fig, axs = plt.subplots(figsize=(8,6), nrows=(nratiopads+1),
                         height_ratios = [4] + [1]*nratiopads)

    # upper pad
    ax = axs[0]

    # plot stacked histograms
    if len(stack_ids)>0:

        # nominal
        mplhep.histplot(
          [hists[idx] for idx in stack_ids],
          stack = True,
          bins = bin_edges,
          histtype = styles[stack_ids[0]],
          color = [colors[idx] for idx in stack_ids],
          edgecolor = [colors[idx] for idx in stack_ids],
          label = [labels[idx] for idx in stack_ids],
          ax=ax)

        # statistical error
        ax.stairs(histsum+staterrorsum, baseline=histsum-staterrorsum,
                  edges=bin_edges,
                  fill=True, color='grey', alpha=0.3)

        # statistical + systematic error
        toterrorsum = np.sqrt(np.square(staterrorsum) + np.square(systerrorsum))
        ax.stairs(histsum+toterrorsum, baseline=histsum-toterrorsum,
                  edges=bin_edges,
                  fill=True, color='grey', alpha=0.15)

    # plot non-stacked histograms
    for idx in nostack_ids:

        # set transparency
        alpha = 1
        if styles[idx]=='fill': alpha = 0.7

        # set multiplicative factor
        factor = 1
        if multdict is not None: factor = mults[idx]

        # nominal
        mplhep.histplot(
          hists[idx]*factor,
          stack = False,
          bins = bin_edges,
          histtype = styles[idx],
          color = colors[idx],
          alpha = alpha,
          linewidth = 3,
          edgecolor = colors[idx],
          label = labels[idx],
          ax=ax
        )

        # statistical error
        ax.stairs((hists[idx]+staterrors[idx])*factor, baseline=(hists[idx]-staterrors[idx])*factor,
                  edges=bin_edges,
                  fill=True, color=colors[idx], alpha=0.3)

        # statistical + systematic error
        toterror = np.sqrt(np.square(staterrors[idx]) + np.square(systerrors[idx]))
        ax.stairs((hists[idx]+toterror)*factor, baseline=(hists[idx]-toterror)*factor,
                  edges=bin_edges,
                  fill=True, color=colors[idx], alpha=0.15)

    # plot data
    if data is not None:
        ax.errorbar(bin_centers, data_hist,
            xerr=None, yerr=data_staterrors,
            linestyle="None", color="black",
            marker="o", markersize=data_markersize, label=data_label)

    # draw vertical dashed lines between secondary bins of double histogram
    if isinstance(variable, DoubleHistogramVariable):
        for xval in np.arange(0, bin_edges[-1]+1, step=variable.primary.nbins):
            ax.axvline(x=xval, ymax=0.65, color='grey', linestyle='dashed')

    # draw secondary bin labels for double histogram
    if isinstance(variable, DoubleHistogramVariable):
        for idx in range(variable.secondary.nbins):
            xlow = variable.secondary.bins[idx]
            xhigh = variable.secondary.bins[idx+1]
            if xlow.is_integer(): xlow = int(xlow)
            if xhigh.is_integer(): xhigh = int(xhigh)
            label = f'{xlow} - {xhigh}'
            pos = variable.primary.nbins*(0.5 + idx)
            pos, _ = (ax.transData + ax.transAxes.inverted()).transform((pos, 0))
            #pos = (pos - ax.get_xlim()[0])/ax.get_xlim()[1]
            ax.text(pos, secondary_bin_label_height-0.05, label,
                    transform=ax.transAxes, ha='center', va='bottom')
            if idx==0:
                label = variable.secondary.axtitle
                if variable.secondary.unit is not None and len(variable.secondary.unit)>0:
                    label += f' [{variable.secondary.unit}]'
                label += ':'
                ax.text(pos, secondary_bin_label_height, label,
                        transform=ax.transAxes, ha='center', va='bottom')

    # set primary bin labels for double histograms
    if isinstance(variable, DoubleHistogramVariable):
        xtick_pos = [0]
        xtick_labels = [variable.primary.bins[0]]
        for xval in np.arange(variable.primary.nbins, bin_edges[-1], step=variable.primary.nbins):
            xtick_pos.append(xval)
            xlow = variable.primary.bins[0]
            if xlow.is_integer(): xlow = int(xlow)
            xhigh = variable.primary.bins[-1]
            if xhigh.is_integer(): xhigh - int(xhigh)
            xtick_labels.append(f'{xhigh:>5} | {xlow:<5}')
        xtick_pos.append(bin_edges[-1])
        xtick_labels.append(variable.primary.bins[-1])
        ax.set_xticks(xtick_pos, labels=xtick_labels)

    # some plot aesthetics
    ax.tick_params(axis='both', which='both', labelsize=15)
    ax.minorticks_on()
    if logscale: ax.set_yscale('log')
    if docms:
        cmstext = r'$\bf{ALEPH}$'
        if extracmstext is not None:
            for part in extracmstext.split(' '): cmstext += r' $\it{' + f' {part}' + r'}$'
        cmstext_in_box = False # maybe later add as argument
        if cmstext_in_box:
            ax.text(0.02, 0.98, cmstext,
                    ha='left', va='top', fontsize=20, transform=ax.transAxes)
            # modify the axis range to accommodate the CMS text
            if logscale:
                yscale = ax.get_ylim()[1]/ax.get_ylim()[0]
                ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*yscale**(0.2))
            else:
                yscale = ax.get_ylim()[1] - ax.get_ylim()[0]
                ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + yscale*0.2)
        else:
            ax.text(0., 1., cmstext,
                    ha='left', va='bottom', fontsize=20, transform=ax.transAxes)
    if lumiheader is not None:
        ax.text(1., 1., lumiheader,
                ha='right', va='bottom', fontsize=20, transform=ax.transAxes)
    
    # set x-axis title
    xaxtitle = None
    if variable is not None:
        if isinstance(variable, DoubleHistogramVariable):
            xaxtitle = variable.primary.axtitle
            unit = variable.primary.unit
        else:
            xaxtitle = variable.axtitle
            unit = variable.unit
        if xaxtitle is not None and len(xaxtitle)>0:
            if unit is not None and len(unit)>0:
                xaxtitle += f' [{unit}]'
            ax.set_xlabel(xaxtitle, fontsize=22)
    
    # set y-axis title
    if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=22)
    
    # make legend
    if dolegend:
        # if number of labels is large, put legend outside the axes
        bbox_to_anchor = None
        loc = None
        if len(hists) > 5:
            loc = 'upper left'
            bbox_to_anchor = (1., 1.)
        ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=17)

    # lower pad
    if ratios is not None and len(ratios)>0:

        # some plot aesthetics: modify the upper pad to remove the title and labels
        axs[0].set_xlabel('', fontsize=0)
        axs[0].xaxis.set_ticklabels([])
        
        # loop over ratios to plot
        for ratio_idx, ratio in enumerate(ratios):
        
            ax = axs[ratio_idx+1]

            # note: ratio is of assumed to be of the form:
            # [numerator, denominator],
            # where both numerator and denominator can be either a string
            # (used as key in the hist dictionaries provides as input)
            # or a list of such strings (in which case the sum is taken).

            # define denominator
            denominator_tags = ratio[1]
            if isinstance(denominator_tags, str):
                denominator_tags = [denominator_tags]
            if data is not None and denominator_tags == [data_key]:
                denominator = np.copy(data_hist)
                denominator_staterrors = np.copy(data_staterrors)
                denominator_systerrors = np.zeros(data_hist.shape)
            else:
                denominator = np.zeros(hists[0].shape)
                denominator_staterrors = np.zeros(hists[0].shape)
                denominator_systerrors = np.zeros(hists[0].shape)
                for tag in denominator_tags:
                    denominator += all_hists[tag][0]
                    denominator_staterrors += np.square(all_hists[tag][1])
                    if systematics is not None and tag in systematics.keys():
                        denominator_systerrors += np.square(systematics[tag][1])
                denominator_staterrors = np.sqrt(denominator_staterrors)
                denominator_systerrors = np.sqrt(denominator_systerrors)
            denominator_mask = (denominator>0).astype(int)
            denominator = np.where(denominator>0, denominator, 1)

            # plot ratio of denominator to itself
            # (for the uncertainties)
            denominator_ratio = np.ones(len(denominator))
            denominator_staterrors_ratio = np.divide(denominator_staterrors, denominator)
            denominator_staterrors_ratio = np.multiply(denominator_staterrors_ratio, denominator_mask)
            denominator_systerrors_ratio = np.divide(denominator_systerrors, denominator)
            denominator_systerrors_ratio = np.multiply(denominator_systerrors_ratio, denominator_mask)
            denominator_toterrors_ratio = np.sqrt(np.square(denominator_staterrors_ratio)
                                            + np.square(denominator_systerrors_ratio))
            ax.stairs(denominator_ratio+denominator_staterrors_ratio,
                  baseline=denominator_ratio-denominator_staterrors_ratio,
                  edges=bin_edges,
                  fill=True, color='grey', alpha=0.3)
            ax.stairs(denominator_ratio+denominator_toterrors_ratio,
                  baseline=denominator_ratio-denominator_toterrors_ratio,
                  edges=bin_edges,
                  fill=True, color='grey', alpha=0.15)

            # define numerator
            numerator_tags = ratio[0]
            if isinstance(numerator_tags, str):
                numerator_tags = [numerator_tags]
            if data is not None and numerator_tags == [data_key]:
                numerator = np.copy(data_hist)
                numerator_staterrors = np.copy(data_staterrors)
                numerator_systerrors = np.zeros(data_hist.shape)
            else:
                numerator = np.zeros(hists[0].shape)
                numerator_staterrors = np.zeros(hists[0].shape)
                numerator_systerrors = np.zeros(hists[0].shape)
                for tag in numerator_tags:
                    numerator += all_hists[tag][0]
                    numerator_staterrors += np.square(all_hists[tag][1])
                    if systematics is not None and tag in systematics.keys():
                        numerator_systerrors += np.square(systematics[tag][1])
                numerator_staterrors = np.sqrt(numerator_staterrors)
                numerator_systerrors = np.sqrt(numerator_systerrors)

            # make ratio
            hist_ratio = np.divide(numerator, denominator)
            hist_ratio = np.multiply(hist_ratio, denominator_mask)
            staterrors_ratio = np.divide(numerator_staterrors, denominator)
            staterrors_ratio = np.multiply(staterrors_ratio, denominator_mask)
            systerrors_ratio = np.divide(numerator_systerrors, denominator)
            systerrors_ratio = np.multiply(systerrors_ratio, denominator_mask)

            # plot ratio
            if data is not None and numerator_tags == [data_key]:
                # case of ratio of data to something else
                ax.errorbar(bin_centers, hist_ratio,
                    xerr=None, yerr=staterrors_ratio,
                    linestyle="None", color="black",
                    marker="o", markersize=data_markersize, label=data_label)
            else:
                # general case
                tag = numerator_tags[0]
                # nominal
                mplhep.histplot(
                  hist_ratio,
                  stack = False,
                  bins = bin_edges,
                  histtype = styledict.get(tag, 'step'),
                  color = colordict.get(tag, 'grey'),
                  edgecolor = colordict.get(tag, 'grey'),
                  label = labeldict.get(tag, ''),
                  ax=ax
                )

                # statistical error
                ax.stairs(hist_ratio+staterrors_ratio, baseline=hist_ratio-staterrors_ratio,
                  edges=bin_edges,
                  fill=True, color=colordict.get(tag, 'grey'), alpha=0.3)

                # statistical + systematic error
                toterror = np.sqrt(np.square(staterrors_ratio) + np.square(systerrors_ratio))
                ax.stairs(hist_ratio+toterror, baseline=hist_ratio-toterror,
                  edges=bin_edges,
                  fill=True, color=colordict.get(tag, 'grey'), alpha=0.15)

            # draw vertical dashed lines between secondary bins of double histogram
            if isinstance(variable, DoubleHistogramVariable):
                for xval in np.arange(0, bin_edges[-1]+1, step=variable.primary.nbins):
                    ax.axvline(x=xval, color='grey', linestyle='dashed')

            # get correct y-axis title
            yaxtitle = 'Ratio'
            if ratio_yaxtitles is not None:
                if len(ratios) != len(ratio_yaxtitles): raise Exception('Incompatible lists.')
                yaxtitle = ratio_yaxtitles[ratio_idx]

            # some plot aesthetics
            ax.set_ylim((0.5, 1.5))
            ax.tick_params(axis='both', which='both', labelsize=15)
            if yaxtitle is not None: ax.set_ylabel(yaxtitle, fontsize=22)
            ax.axhline(y=1, color='grey', linestyle='dashed')

            # some plot aesthetics: remove the title and labels
            if ratio_idx!=len(ratios)-1:
                ax.set_xlabel('', fontsize=0)
                ax.xaxis.set_ticklabels([])

        # add title and labels to the lowest pad
        if xaxtitle is not None and len(xaxtitle)>0:
            axs[-1].set_xlabel(xaxtitle, fontsize=22)
        if isinstance(variable, DoubleHistogramVariable):
            axs[-1].set_xticks(xtick_pos, labels=xtick_labels)

    if ratios is None: return fig, ax
    else: return fig, axs


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--signals', default=None)
    parser.add_argument('-b', '--backgrounds', default=None)
    parser.add_argument('-v', '--variables', required=True)
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('-t', '--treename', default=None)
    parser.add_argument('--selection', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--colors', default=None)
    parser.add_argument('--labels', default=None)
    parser.add_argument('--styles', default=None)
    parser.add_argument('--dolog', default=False, action='store_true')
    parser.add_argument('--extracmstext', default=None)
    parser.add_argument('--lumiheader', default=None)
    parser.add_argument('--entry_start', default=None)
    parser.add_argument('--entry_stop', default=None)
    args = parser.parse_args()

    # read variables
    variables = read_variables(args.variables)
    variablelist = [v.variable for v in variables]

    # read merge dict
    mergedict = None
    if args.merge is not None:
        with open(args.merge, 'r') as f:
            mergedict = json.load(f)
        print('Found following instructions for merging samples:')
        print(json.dumps(mergedict, indent=2))

    # load the selection dict
    # and parse selection
    selection = None
    if args.selection is not None:
        selection = load_eventselection(args.selection)
        if len(selection.keys())!=1:
            msg = 'Ambiguous number of selections; only one is supported.'
            raise Exception(msg)
        key = list(selection.keys())[0]
        selection = selection[key]
        print('Will apply the following selection:')
        print(json.dumps(selection, indent=2))

    # set branches to read
    branches_to_read = variablelist
    if args.weighted:
        branches_to_read.append('genWeight')
        branches_to_read.append('xsecWeight')
        branches_to_read.append('lumiwgt')
    if selection is not None:
        branches_to_read += get_variable_names(selection)

    # read all files
    signal_events = {}
    if args.signals is not None:
        print('Reading signal samples...')
        signal_events = read_samplelist(args.signals,
                          treename=args.treename,
                          branches=branches_to_read,
                          entry_start=args.entry_start,
                          entry_stop=args.entry_stop)
    background_events = {}
    if args.backgrounds is not None:
        print('Reading background samples...')
        background_events = read_samplelist(args.backgrounds,
                              treename=args.treename,
                              branches=branches_to_read,
                              entry_start=args.entry_start,
                              entry_stop=args.entry_stop)

    # do merging
    if mergedict is not None:
        print('Merging samples...')
        signal_events = merge_events(signal_events, mergedict, verbose=True)
        background_events = merge_events(background_events, mergedict, verbose=True)
        
    # merge signal and background events for easier processing
    # (can split again later using the keys)
    signal_keys = list(signal_events.keys())
    background_keys = list(background_events.keys())
    events = signal_events
    events.update(background_events)
    del signal_events
    del background_events

    # do selection
    if selection is not None:
        print('Doing selection...')
        for key in events.keys():
            nevents = len(events[key])
            mask = get_selection_mask(events[key], selection)
            events[key] = events[key][mask]
            nselected = len(events[key])
            print(f'  - Event category {key}: selected {nselected} out of {nevents} entries.')

    # read colors
    colordict = None
    if args.colors is not None:
        with open(args.colors, 'r') as f:
            colordict = json.load(f)

    # read labels
    labeldict = None
    if args.labels is not None:
        with open(args.labels, 'r') as f:
            labeldict = json.load(f)

    # read styles
    styledict = None
    if args.styles is not None:
        with open(args.styles, 'r') as f:
            styledict = json.load(f)

    # make output directory
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # loop over variables
    for varidx, variable in enumerate(variables):
        print(f'Plotting variable {variable.name}')
        
        # make histograms
        hists = {}
        for key, sample in events.items():
            values = sample[variable.variable].to_numpy().astype(float)
            weights = None
            if args.weighted:
                weights = (sample['genWeight'].to_numpy().astype(float)
                           * sample['xsecWeight'].to_numpy().astype(float)
                           * sample['lumiwgt'].to_numpy().astype(float))
            hists[key] = make_hist(values, variable, weights=weights)

        # split in signal and background
        signal_hists = {key: val for key, val in hists.items() if key in signal_keys}
        background_hists = {key: val for key, val in hists.items() if key in background_keys}

        # do plotting
        yaxtitle = 'Events'
        if args.normalize: yaxtitle += ' (normalized)'
        fig, ax = plot(sig=signal_hists, bkg=background_hists,
                       variable=variable,
                       colordict=colordict,
                       labeldict=labeldict,
                       styledict=styledict,
                       normalize=args.normalize,
                       extracmstext=args.extracmstext,
                       lumiheader=args.lumiheader,
                       yaxtitle=yaxtitle,
                       dolegend=True)

        # save the figure
        fig.tight_layout()
        figname = variable.name + '.png'
        figname = os.path.join(args.outputdir, figname)
        fig.savefig(figname)
        plt.close(fig)

        # same with log scale
        if args.dolog:
            fig, ax = plot(sig=signal_hists, bkg=background_hists,
                       variable=variable,
                       colordict=colordict,
                       labeldict=labeldict,
                       styledict=styledict,
                       normalize=args.normalize,
                       logscale=True,
                       extracmstext=args.extracmstext,
                       lumiheader=args.lumiheader,
                       yaxtitle=yaxtitle,
                       dolegend=True)

            # save the figure
            fig.tight_layout()
            figname = variable.name + '_log.png'
            figname = os.path.join(args.outputdir, figname)
            fig.savefig(figname)
            plt.close(fig)
