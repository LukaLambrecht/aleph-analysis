# Plot analysis results

import os
import sys
import json
import copy
import pickle
import uproot
import argparse
import numpy as np
import awkward as ak
from fnmatch import fnmatch
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.variabletools import read_variables
from tools.variabletools import HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist, read_sampledict, find_files
from tools.lumitools import get_lumidict
from tools.plottools import merge_sampledict
from tools.processinfo import ProcessInfoCollection, ProcessCollection
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.eventselection import get_selection_mask
from analysis.objectselection import load_objectselection
from analysis.plot import make_histograms, make_events, plot_hists_default
from analysis.plot import get_default_style_dicts
from analysis.external_variables import read_external_variables
from tools.plottools import default_flavour_labels
from plotting.plot import plot


def build_tag_selection_expression(cuts, idx):
    '''
    Build a tag jet selection expression from a list of cuts.
    Input arguments:
    - cuts: list of dicts, each with key "branch" and at least one of "min", "max"
      (e.g. [{"branch": "Jets_score_isB", "min": 0.9}]).
    - idx: index of the tag jet within the per-event jet collection.
    '''
    subexpressions = []
    for cut in cuts:
        branch = cut['branch']
        if 'min' not in cut and 'max' not in cut:
            msg = f'Cut {cut} does not specify "min" and/or "max".'
            raise Exception(msg)
        if 'min' in cut: subexpressions.append(f'{branch}[:, {idx}] > {cut["min"]}')
        if 'max' in cut: subexpressions.append(f'{branch}[:, {idx}] < {cut["max"]}')
    return ' & '.join([f'({s})' for s in subexpressions])


def make_summary_plot(hists_combined, variables, tag_selections, outputdir,
        colordict=None, labeldict=None, styledict=None, stacklist=None,
        region_labeldict=None, datatag='data', dolog=True):
    '''
    Make a single grid figure combining all (tag region, probe variable) panels.
    Rows: tag regions (i.e. flavour-enriched regions based on the tag jet).
    Columns: probe jet score variable.
    The ALEPH logo is only drawn on the top-left panel, and the legend only
    on the top-right panel; axis titles are only drawn on the outer edges.
    If dolog is True, an additional version with log-scale y-axes is made.
    '''

    region_names = list(tag_selections.keys())
    nrows = len(region_names)
    ncols = len(variables)

    # get default style dicts (only used for whichever of these were not explicitly provided)
    dummykey = f'{region_names[0]}_{variables[0].name}'
    sim_processes = list(hists_combined['sim'][dummykey].keys())
    default_colordict, default_labeldict, default_styledict, default_stacklist, _ = (
        get_default_style_dicts(sim_processes))
    if colordict is None: colordict = default_colordict
    if labeldict is None: labeldict = default_labeldict
    if styledict is None: styledict = default_styledict
    if stacklist is None: stacklist = default_stacklist

    def make_grid(logscale=False):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                    figsize=(4.2*ncols, 3.6*nrows), squeeze=False)

        for row_idx, region_name in enumerate(region_names):
            for col_idx, variable in enumerate(variables):
                ax = axs[row_idx, col_idx]
                region_variable_key = f'{region_name}_{variable.name}'
                is_top_left = (row_idx==0 and col_idx==0)
                is_top_right = (row_idx==0 and col_idx==ncols-1)
                is_bottom_row = (row_idx==nrows-1)
                is_left_col = (col_idx==0)

                # get nominal histograms for simulation and data
                hists_sim_nominal = {
                    process_key: hists_combined['sim'][region_variable_key][process_key]['nominal']
                    for process_key in hists_combined['sim'][region_variable_key].keys()
                }
                hists_data = {
                    process_key: hists_combined['data'][region_variable_key][process_key]['nominal']
                    for process_key in hists_combined['data'][region_variable_key].keys()
                }
                data = {datatag: hists_data[datatag]}

                # do the actual plotting on the provided axis
                plot(bkg=hists_sim_nominal, data=data,
                     variable=variable,
                     fig=fig, ax=ax,
                     stacklist=stacklist, colordict=colordict, labeldict=labeldict, styledict=styledict,
                     logscale=logscale,
                     docms=is_top_left,
                     yaxtitle=('Jets' if is_left_col else None),
                     dolegend=is_top_right)

                # add headroom at the top of the panel so the region annotation
                # (drawn below) does not overlap the histogram
                if not logscale:
                    ax.set_ylim((0, ax.get_ylim()[1]*1.6))
                else:
                    histarray = np.array([h[0] for h in hists_sim_nominal.values()]
                                          + [data[datatag][0]])
                    if np.any(histarray > 0):
                        ymin = np.min(histarray[np.nonzero(histarray)])
                        ax.set_ylim((ymin, ax.get_ylim()[1]**1.6))

                # remove axis titles except on the outer edges
                if not is_bottom_row:
                    ax.set_xlabel('')
                if not is_left_col:
                    ax.set_ylabel('')

                # add flavour-enriched-region annotation
                if region_labeldict is not None and region_name in region_labeldict:
                    ax.text(0.05, 0.95, region_labeldict[region_name], ha='left', va='top', fontsize=13,
                            transform=ax.transAxes)

        fig.tight_layout()
        return fig

    # make linear-scale version
    fig = make_grid(logscale=False)
    figname = os.path.join(outputdir, 'summary.png')
    fig.savefig(figname)
    fig.savefig(figname.replace('.png', '.pdf'))
    plt.close(fig)
    print(f'Summary figure saved to {figname}.')

    # make log-scale version
    if dolog:
        fig = make_grid(logscale=True)
        figname = os.path.join(outputdir, 'summary_log.png')
        fig.savefig(figname)
        fig.savefig(figname.replace('.png', '.pdf'))
        plt.close(fig)
        print(f'Summary figure saved to {figname}.')


def smooth_values(values, errors, window):
    '''
    Smooth an array of values with a double-sided sliding window of adjustable size,
    weighting each point in the window by the inverse square of its statistical uncertainty
    (i.e. an inverse-variance weighted moving average).
    Input arguments:
    - values: 1D array of values to smooth.
    - errors: 1D array of statistical uncertainties on values (same shape as values).
      Points with a non-positive or infinite uncertainty carry no information and are
      given zero weight (e.g. bins for which no simulation statistics were available).
    - window: half-width of the sliding window, in number of neighboring bins to include
      on either side of the point in question (i.e. the full window size is 2*window+1).
    '''
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    n = len(values)
    with np.errstate(divide='ignore'):
        weights = np.where((errors>0) & np.isfinite(errors), 1./np.square(errors), 0.)
    smoothed = np.copy(values)
    for i in range(n):
        lo = max(0, i-window)
        hi = min(n, i+window+1)
        w = weights[lo:hi]
        wsum = np.sum(w)
        if wsum>0: smoothed[i] = np.sum(w*values[lo:hi])/wsum
    return smoothed


def make_scale_factor_plot(output_struct, outputfile,
        region_labeldict=None, region_colordict=None, axtitledict=None, ylim=(0, 2),
        smooth_window=None):
    '''
    Make a grid figure of the scale factor curves as produced by calibration.py
    (i.e. the content of output.json), with one panel per (tag region, probe score) combination.
    Rows: tag regions (i.e. flavour-enriched regions based on the tag jet).
    Columns: probe jet score variable.
    Input arguments:
    - output_struct: dict of the form
      {region_name: {variable_name: {"bins": [...], "values": [...], "errors": [...]}}}
      (same structure as written to / read from output.json). The "errors" key is optional,
      but required if smooth_window is used.
    - outputfile: path of the file to write the figure to (a companion .pdf is
      written alongside it, next to whatever extension outputfile has, typically .png).
    - region_colordict: optional dict mapping region_name to a line color
      (e.g. the same color used for that flavour's process in the stacked plots).
    - axtitledict: optional dict mapping variable_name to x-axis title.
    - ylim: shared y-axis range for all panels (default (0, 2), matching the clipping range
      applied to the aggregated per-event weight in apply_calibration.py).
    - smooth_window: if provided, smooth the scale factor curves with this half-window size
      before plotting (see smooth_values); the underlying output_struct is left untouched.
    '''

    region_names = list(output_struct.keys())
    variable_names = list(output_struct[region_names[0]].keys())
    nrows = len(region_names)
    ncols = len(variable_names)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                figsize=(4.2*ncols, 3.6*nrows), squeeze=False)

    for row_idx, region_name in enumerate(region_names):
        for col_idx, variable_name in enumerate(variable_names):
            ax = axs[row_idx, col_idx]
            is_top_left = (row_idx==0 and col_idx==0)
            is_bottom_row = (row_idx==nrows-1)
            is_left_col = (col_idx==0)

            bins = np.array(output_struct[region_name][variable_name]['bins'])
            values = np.array(output_struct[region_name][variable_name]['values'])
            if smooth_window is not None:
                if 'errors' not in output_struct[region_name][variable_name]:
                    msg = 'Smoothing was requested, but no "errors" were found in output_struct'
                    msg += f' for region "{region_name}", variable "{variable_name}";'
                    msg += ' regenerate output.json with an updated calibration.py to enable smoothing.'
                    raise Exception(msg)
                errors = np.array(output_struct[region_name][variable_name]['errors'])
                values = smooth_values(values, errors, smooth_window)
            color = 'black'
            if region_colordict is not None and region_name in region_colordict:
                color = region_colordict[region_name]

            # reference line at unity, and the scale factor curve itself
            ax.axhline(y=1, color='grey', linestyle='dashed')
            ax.stairs(values, edges=bins, color=color, linewidth=2.5)

            ax.set_ylim(ylim)
            ax.tick_params(axis='both', which='both', labelsize=15)
            ax.minorticks_on()

            if is_top_left:
                ax.text(0., 1., r'$\bf{ALEPH}$', ha='left', va='bottom', fontsize=20,
                        transform=ax.transAxes)

            if is_bottom_row and axtitledict is not None and variable_name in axtitledict:
                ax.set_xlabel(axtitledict[variable_name], fontsize=22)
            if is_left_col:
                ax.set_ylabel('Scale factor', fontsize=22)

            if region_labeldict is not None and region_name in region_labeldict:
                ax.text(0.05, 0.95, region_labeldict[region_name], ha='left', va='top', fontsize=13,
                        transform=ax.transAxes)

    fig.tight_layout()
    outputdir = os.path.dirname(outputfile)
    if len(outputdir)>0 and not os.path.exists(outputdir): os.makedirs(outputdir)
    fig.savefig(outputfile)
    root, ext = os.path.splitext(outputfile)
    fig.savefig(root + '.pdf')
    plt.close(fig)
    print(f'Scale factor figure saved to {outputfile}.')


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-d', '--data', default=None, nargs='+')
    parser.add_argument('-o', '--outputdir', default=None)
    parser.add_argument('--objectselection', default=None)
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--select_processes', default=[], nargs='+')
    parser.add_argument('--external_variables', default=None)
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--tag_jet_selection', required=True)
    parser.add_argument('--smooth_window', default=None, type=int,
      help='If provided, smooth the scale factor plot with a double-sided sliding window'
           + ' of this half-width (in number of bins on either side), with each point'
           + ' weighted by its statistical uncertainty. Does not affect output.json.')
    args = parser.parse_args()

    # set weight variations to include in the uncertainty band
    # (hard-coded for now, maybe extend later)
    weight_variations = {}

    # parse arguments
    if args.data is not None and len(args.data)==0: args.data = None

    # read extra object selection to apply
    objectselection = None
    if args.objectselection is not None:
        objectselection = load_objectselection(args.objectselection)
        print('Found following extra object selection to apply:')
        print(objectselection[0])
        print('(to the following branches):')
        print(objectselection[1])

    # read extra selection to apply
    event_selection_name = None
    eventselection = None
    select_processes = None
    if args.eventselection is not None:
        eventselection = load_eventselection(args.eventselection, nexpect=1)
        print('Found following extra event selection to apply:')
        print(eventselection)
        if len(args.select_processes)>0:
            select_processes = args.select_processes
            print('(selection will be applied only to the following processes:'
                    + f' {select_processes})')
        event_selection_name = list(eventselection.keys())[0]
        eventselection = eventselection[event_selection_name]

    # read tag jet selection
    with open(args.tag_jet_selection, 'r') as f:
        tag_jet_selection = json.load(f)
    print('Found following tag jet selection:')
    print(json.dumps(tag_jet_selection, indent=2))

    # read cross-sections
    xsections = None
    if args.xsections is not None:
        with open(args.xsections, 'r') as f:
            xsections = json.load(f)
        print('Found following cross-sections:')
        print(json.dumps(xsections, indent=2))

    # read merging instructions
    mergedict = None
    if args.merge is not None:
        with open(args.merge, 'r') as f:
            mergedict = json.load(f)
        print('Found following instructions for merging samples:')
        print(json.dumps(mergedict, indent=2))

    # read splitting instructions
    splitdict = None
    if args.split is not None:
        with open(args.split, 'r') as f:
            splitdict = json.load(f)
        print('Found following instructions for splitting samples:')
        print(json.dumps(splitdict, indent=2))

    # find samples for simulation
    sampledirs_sim = []
    print('Finding sample files for simulation...')
    for sampledir in args.sim:
        # first check if a file 'files.json' is present (i.e. after merging years)
        ffile = os.path.join(sampledir, 'files.json')
        if os.path.exists(ffile): sampledirs_sim.append(ffile)
        # else default case: find all .root files in the given directory
        else: sampledirs_sim.append(sampledir)
    sampledict_sim = find_files(sampledirs_sim, verbose=False)
    #print('Found following sample dict for simulation:')
    #print(json.dumps(sampledict_sim, indent=2))
    nsimfiles = sum([len(v) for v in sampledict_sim.values()])
    print(f'Found {nsimfiles} simulation files.')

    # find samples for data
    sampledict_data = None
    if args.data is not None:
        sampledirs_data = []
        print('Finding sample files for data...')
        for sampledir in args.data:
            # first check if a file 'files.json' is present (i.e. after merging years)
            ffile = os.path.join(sampledir, 'files.json')
            if os.path.exists(ffile): sampledirs_data.append(ffile)
            # else default case: find all .root files in the given directory
            else: sampledirs_data.append(sampledir)
        sampledict_data = find_files(sampledirs_data, verbose=False)
        #print('Found following sample dict for data:')
        #print(json.dumps(sampledict_data, indent=2))
        ndatafiles = sum([len(v) for v in sampledict_data.values()])
        print(f'Found {ndatafiles} data files.')

    # do merging
    if mergedict is not None:
        print('Merging samples...')
        sampledict_sim = merge_sampledict(sampledict_sim, mergedict, verbose=True)
        if sampledict_data is not None:
            print('Merging data...')
            sampledict_data = merge_sampledict(sampledict_data, mergedict, verbose=False)
        # printouts for testing
        print('Number of files for (merged) samples:')
        for sampledict in [sampledict_sim, sampledict_data]:
            if sampledict is None: continue
            for key, val in sampledict.items():
                print(f'  - {key}: {len(val)}')

    # get luminosity from year
    luminosity = args.luminosity
    if args.year is not None:
        lumi_from_year = get_lumidict()[args.year]
        if args.luminosity is None or args.luminosity < 0:
            luminosity = lumi_from_year
        elif luminosity!=lumi_from_year:
            msg = f'WARNING: found inconsistency between provided luminosity ({luminosity})'
            msg += f' and the one corresponding to the provided year ({args.year}: {lumi_from_year}).'
            print(msg)
    if luminosity < 0: luminosity = None

    # define variables to read
    branches_to_read = []
    # add selection
    if objectselection is not None:
        branches_to_read += get_variable_names(objectselection[0])
    if eventselection is not None:
        branches_to_read += get_variable_names(eventselection)
    # add variables needed for splitting
    if splitdict is not None:
        for splitkey, this_splitdict in splitdict.items():
            for selection_string in this_splitdict.values():
                branches_to_read += get_variable_names(selection_string)
    # add jet kinematics
    branches_to_read += ['Jets_pt', 'Jets_theta']
    # remove potential duplicates
    branches_to_read = list(set(branches_to_read))
    print('Found following branches to read:')
    print(branches_to_read)

    # make events
    dtypedict = {'sim': sampledict_sim, 'data': sampledict_data}
    events_combined = make_events(dtypedict,
                       branches_to_read = branches_to_read,
                       objectselection = objectselection,
                       eventselection = eventselection,
                       select_processes = select_processes,
                       external_variables = args.external_variables)

    # define selections for events with good tags
    print('Applying selection on tag jet...')
    (tag_idx, probe_idx) = (0, 1)
    #(tag_idx, probe_idx) = (1, 0) # to see if it makes a difference
    tag_selections = {
        key: build_tag_selection_expression(cuts, tag_idx)
        for key, cuts in tag_jet_selection.items()
    }

    # map each tag region to the corresponding truth-flavour subprocess
    # (as defined in the splitting instructions, e.g. "b_jets" region <-> "bb" subprocess)
    process_of_interest = {
        'b_jets': 'bb',
        'c_jets': 'cc',
        's_jets': 'ss',
        'ud_jets': 'uudd',
    }

    # build annotation labels for each tag region
    # (reusing the same LaTeX flavour labels as used in the legend)
    flavour_labels = default_flavour_labels()
    region_labeldict = {
        region_name: flavour_labels[process_of_interest[region_name]] + '-enriched region'
        for region_name in tag_selections.keys() if region_name in process_of_interest
    }

    # map each tag region to the color used for the corresponding process
    # in the stacked plots (e.g. "b_jets" region <-> "bb" color)
    process_colordict, _, _, _, _ = get_default_style_dicts(list(process_of_interest.values()))
    region_colordict = {
        region_name: process_colordict[process_of_interest[region_name]]
        for region_name in tag_selections.keys() if region_name in process_of_interest
    }

    # make masks
    tag_masknames = {k: 'mask_' + k for k in tag_selections.keys()}
    for dtype in events_combined.keys():
        for process_key in events_combined[dtype].keys():
            for key, val in tag_selections.items():
                mask = get_selection_mask(
                         events_combined[dtype][process_key], val).to_numpy().astype(bool)
                events_combined[dtype][process_key]['mask_' + key] = mask
                nbefore = len(mask)
                nafter = np.sum(mask.astype(int))
                print(f'  - {process_key}: selected {nafter} out of {nbefore} events for tag {key}.')

    # determine global normalization factor based on number of selected events
    # with a good tag jet
    norm_factors = {}
    for key in tag_selections.keys():
        ndata = np.sum(events_combined['data']['data']['mask_' + key].to_numpy().astype(int))
        nsim = np.sum(events_combined['sim']['qqb']['mask_' + key].to_numpy().astype(int))
        norm_factor = float(ndata) / nsim
        norm_factors[key] = norm_factor
        print(f'Normalization factor for tag {key}: {norm_factor}')

    # make weights corresponding to the normalization factors above
    # (note: assumes the tag regions to be disjoint)
    weights = np.ones(len(events_combined['sim']['qqb']))
    for key in tag_selections.keys():
        mask = events_combined['sim']['qqb']['mask_' + key].to_numpy()
        weights = np.where(mask, norm_factors[key], weights)
    events_combined['sim']['qqb']['weight'] = weights

    # make plots of probe jet
    variables = [
        HistogramVariable.fromdict({
            "name": "b_score",
            "variable": f"Jets_score_isB[:, {probe_idx}]",
            "bins": np.linspace(0, 1, num=51),
            "axtitle": "Probe jet classifier b score"
        }),
        HistogramVariable.fromdict({
            "name": "c_score",
            "variable": f"Jets_score_isC[:, {probe_idx}]",
            "bins": np.linspace(0, 1, num=51),
            "axtitle": "Probe jet classifier c score"
        }),
        HistogramVariable.fromdict({
            "name": "s_score",
            "variable": f"Jets_score_isS[:, {probe_idx}]",
            "bins": np.linspace(0, 1, num=51),
            "axtitle": "Probe jet classifier s score"
        }),
        HistogramVariable.fromdict({
            "name": "ud_score",
            "variable": f"Jets_score_isUDG[:, {probe_idx}]",
            "bins": np.linspace(0, 1, num=51),
            "axtitle": "Probe jet classifier udg score"
        })
    ]
    weights = {'qqb': ['weight']}
    hists_combined = make_histograms(events_combined, variables,
                           regions=tag_masknames,
                           splitdict=splitdict,
                           weights=weights)
    plot_hists_default(hists_combined, variables,
          outputdir=args.outputdir,
          regions=tag_selections,
          datatag='data',
          region_labeldict=region_labeldict,
          dolog=True)

    # make summary grid figure combining all regions and variables
    make_summary_plot(hists_combined, variables, tag_selections, args.outputdir,
          region_labeldict=region_labeldict, datatag='data')

    # calculate ratios (and their statistical uncertainties) to store in output
    ratios = {}
    ratio_errors = {}
    for region_name in tag_selections.keys():
        ratios[region_name] = {}
        ratio_errors[region_name] = {}
        poi = process_of_interest.get(region_name, region_name)
        for variable in variables:
            region_variable_key = f'{region_name}_{variable.name}'
            counts_data = hists_combined['data'][region_variable_key]['data']['nominal'][0]
            errors_data = hists_combined['data'][region_variable_key]['data']['nominal'][1]
            counts_sim_this = []
            counts_sim_other = []
            errors_sim_this = []
            errors_sim_other = []
            for process_key in hists_combined['sim'][region_variable_key].keys():
                counts = hists_combined['sim'][region_variable_key][process_key]['nominal'][0]
                errors = hists_combined['sim'][region_variable_key][process_key]['nominal'][1]
                if process_key == poi:
                    counts_sim_this.append(counts)
                    errors_sim_this.append(errors)
                else:
                    counts_sim_other.append(counts)
                    errors_sim_other.append(errors)
            if len(counts_sim_this)!=1:
                msg = f'Exactly one process of interest was expected, but found {len(counts_sim_this)}'
                raise Exception(msg)
            counts_sim_this = sum(counts_sim_this)
            counts_sim_other = sum(counts_sim_other)
            errors_sim_this = np.sqrt(sum([np.square(e) for e in errors_sim_this]))
            errors_sim_other = (np.sqrt(sum([np.square(e) for e in errors_sim_other]))
                                 if len(errors_sim_other)>0 else np.zeros(len(counts_data)))
            errors_sim_total = np.sqrt(np.square(errors_sim_this) + np.square(errors_sim_other))
            # original: just do data / total sim
            numerator = np.clip(counts_data, a_min=0, a_max=None)
            counts_sim_this += counts_sim_other
            denominator = np.clip(counts_sim_this, a_min=1, a_max=None)
            # alternative: do (data - other sim) / (this sim)
            #numerator = np.clip(counts_data - counts_sim_other, a_min=0, a_max=None)
            #denominator = np.clip(counts_sim_this, a_min=1, a_max=None)
            # make ratio
            ratio = np.ones(len(counts_data))
            ratio = np.where(counts_sim_this > 0, np.divide(numerator, denominator), 1)
            ratios[region_name][variable.name] = ratio
            # propagate statistical uncertainties onto the ratio
            # (bins without simulation statistics get an infinite uncertainty,
            #  i.e. they carry no information and get zero weight when smoothing)
            safe_numerator = np.clip(numerator, a_min=1, a_max=None)
            rel_error_num = errors_data / safe_numerator
            rel_error_denom = errors_sim_total / denominator
            ratio_error = ratio * np.sqrt(np.square(rel_error_num) + np.square(rel_error_denom))
            ratio_error = np.where(counts_sim_this > 0, ratio_error, np.inf)
            ratio_errors[region_name][variable.name] = ratio_error

    # write output file
    output_struct = {}
    for region_name in tag_selections.keys():
        output_struct[region_name] = {}
        for variable in variables:
            values = ratios[region_name][variable.name]
            errors = ratio_errors[region_name][variable.name]
            bins = variable.bins
            output_struct[region_name][variable.name] = {
                "bins": list(bins),
                "values": list(values),
                "errors": [float(e) for e in errors]
            }
    outputfile = os.path.join(args.outputdir, 'output.json')
    with open(outputfile, 'w') as f:
        json.dump(output_struct, f, indent=2)

    # make a grid figure of the scale factor curves
    axtitledict = {variable.name: variable.axtitle for variable in variables}
    scale_factor_outputfile = os.path.join(args.outputdir, 'scale_factors.png')
    make_scale_factor_plot(output_struct, scale_factor_outputfile,
        region_labeldict=region_labeldict, region_colordict=region_colordict, axtitledict=axtitledict,
        smooth_window=args.smooth_window)
