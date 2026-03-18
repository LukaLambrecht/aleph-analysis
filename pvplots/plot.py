# Plot analysis results

import os
import sys
import json
import uproot
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.variabletools import read_variables
from tools.variabletools import HistogramVariable, DoubleHistogramVariable
from tools.samplelisttools import read_samplelist, read_sampledict, find_files
from tools.lumitools import get_lumidict, get_sqrtsdict
from tools.plottools import merge_events, merge_sampledict
from tools.processinfo import ProcessInfoCollection, ProcessCollection
from analysis.plot import make_histograms
from analysis.objectselection import load_objectselection
from analysis.eventselection import load_eventselection, get_variable_names
from plotting.plot import plot

# global pyplot settings
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def plot_hists(hists_combined, variables, outputdir,
      regions=None, datatag=None,
      colordict=None, labeldict=None, styledict=None, stacklist=None,
      shapes=False, normalizesim=False, dolog=False,
      extracmstext=None, lumiheader=None, event_selection_name=None, select_processes=None):
    '''
    Plotting loop
    '''

    # make a list of all simulated processes
    dummykey = list(hists_combined['sim'].keys())[0]
    sim_processes = list(hists_combined['sim'][dummykey].keys())

    # make color dict
    if colordict is None:
        colordict = {}
        colordict['qqb'] = 'grey'
        colordict['light'] = 'grey'
        colordict['uudd'] = 'paleturquoise'
        colordict['ss'] = 'dodgerblue'
        colordict['cc'] = 'slateblue'
        colordict['bb'] = 'darkorchid'

    # make label dict
    if labeldict is None:
        labeldict = {}
        for p in sim_processes:
            labeldict[p] = p
        labeldict['bb'] = r'$b\overline{b}$'
        labeldict['cc'] = r'$c\overline{c}$'
        labeldict['ss'] = r'$s\overline{s}$'
        labeldict['uudd'] = r'$u\overline{u}$, $d\overline{d}$'

    # set histogram styles
    if styledict is None:
        styledict = {}
        for p in sim_processes: styledict[p] = 'fill'
        if shapes:
            for p in sim_processes: styledict[p] = 'step'

    # set histogram stacking
    if stacklist is None:    
        stacklist = [p for p in sim_processes]
        normalize = False
        if shapes:
            stacklist = []
            normalize = True

    # loop over regions and variables
    if regions is None: regions = {'baseline': None}
    for region_name, mask_name in regions.items():
        for variable in variables:
            print(f'Plotting selection {region_name}, variable {variable.name}...')
            region_variable_key = f'{region_name}_{variable.name}'

            # get nominal histograms for simulation
            hists_sim_nominal = {}
            for process_key in hists_combined['sim'][region_variable_key].keys():
                hists_sim_nominal[process_key] = hists_combined['sim'][region_variable_key][process_key]['nominal']

            # get histograms for data
            hists_data = None
            if datatag is not None:
                hists_data = {}
                for process_key in hists_combined['data'][region_variable_key].keys():
                    hists_data[process_key] = hists_combined['data'][region_variable_key][process_key]['nominal']

            # fit a function and calculate the width per process
            bins = variable.bins
            bincenters = (bins[:-1] + bins[1:]) / 2
            def gauss(x, a, mu, sigma):
                return a * np.exp(-0.5*np.square((x-mu)/sigma))
            def studentt(x, a, mu, sigma, nu):
                return a * scipy.stats.t.pdf(x, nu, mu, sigma)
            process_widths = {}
            fitted_functions = {}
            for process_key, hist in hists_sim_nominal.items():
                counts = hist[0]
                a_init = np.amax(counts)
                mu_init = bincenters[np.argmax(counts)]
                sigma_init = mu_init - bincenters[np.nonzero(counts > a_init/2)[0][0]]

                # gaussian fit
                #fitresult = curve_fit(gauss, bincenters, counts, p0=[a_init, mu_init, sigma_init])
                #a, mu, sigma = fitresult[0]
                #process_widths[process_key] = sigma
                #fitted_functions[process_key] = gauss(bincenters, a, mu, sigma)

                # student t fit
                nu_init = 10
                fitresult = curve_fit(studentt, bincenters, counts, p0=[a_init, mu_init, sigma_init, nu_init])
                a, mu, sigma, nu = fitresult[0]
                process_widths[process_key] = sigma
                fitted_functions[process_key] = studentt(bincenters, a, mu, sigma, nu)

            # concatenate all histograms in a single array (for later use)
            histarray = [h[0] for h in hists_sim_nominal.values()]
            if hists_data is not None:
                histarray += [h[0] for h in hists_data.values()]
            histarray = np.array(histarray)

            # split off data hist
            data = None
            if hists_data is not None: data = {datatag: hists_data[datatag]}

            # make a ProcessCollection
            hists_sim = {}
            for process_key in hists_combined['sim'][region_variable_key].keys():
                for systematic_key, hist in hists_combined['sim'][region_variable_key][process_key].items():
                    histname = f'{process_key}_{region_variable_key}_{systematic_key}'
                    hists_sim[histname] = hist
            pic = ProcessInfoCollection.fromhistlist(list(hists_sim.keys()), region_variable_key)
            pc = ProcessCollection(pic, hists_sim)
            print(pic)

            # extract the systematic uncertainties (per process)
            systematics = {}
            for process_key in hists_combined['sim'][region_variable_key].keys():
                systematic = pc.get_systematics_rss(processes=[process_key])[0]
                systematics[process_key] = (hists_sim_nominal[process_key][0], systematic)

            # define ratios to plot
            ratios = []
            ratio_yaxtitles = []
            if datatag is not None:
                ratios.append([datatag, stacklist])
                ratio_yaxtitles.append('Data / MC')

            # modify label dict to include the yield per process
            this_labeldict = labeldict.copy()
            print_yield = False # maybe later add as argument
            if print_yield:
                for process_key, hist in hists_sim_nominal.items():
                    old_label = labeldict.get(process_key, None)
                    if old_label is None: continue
                    process_yield = np.sum(hist[0])
                    new_label = old_label + ' ({:.2e})'.format(process_yield)
                    this_labeldict[process_key] = new_label

            # modify label dict to include the width per process
            print_width = True # maybe later add as argument
            if print_width:
                for process_key, hist in hists_sim_nominal.items():
                    old_label = this_labeldict.get(process_key, None)
                    if old_label is None: continue
                    process_width = process_widths[process_key]
                    widthtxt = '{:.0f}'.format(process_width*1e4) + r' $\mu m$'
                    new_label = old_label + r' ($\sigma =$ {})'.format(widthtxt)
                    this_labeldict[process_key] = new_label

            # set y-axis title
            yaxtitle = 'Events'
            if variable.variable.startswith('Jets_'): yaxtitle = 'Jets'
            include_binwidth = True # maybe later add as argument
            if include_binwidth:
                if variable.unit is not None and len(variable.unit)>0:
                    bins = variable.bins
                    binwidths = bins[1:] - bins[:-1]
                    unique_binwidths = list(set(binwidths))
                    unique_binwidths = ([unique_binwidths[0]]
                        + [el for el in unique_binwidths[1:] if abs(el-unique_binwidths[0])/unique_binwidths[0] > 1e-6])
                    if len(unique_binwidths)==1:
                        binwidth = unique_binwidths[0]
                        # specific hack for this specific case
                        binwidth = int(round(1e4*binwidth))
                        variable_yax_unit = r'$\mu m$'
                        # continue generic approach
                        binwidthtxt = '{:.2f}'.format(binwidth)
                        if binwidth.is_integer(): binwidthtxt = str(int(binwidth))
                        yaxtitle += f' / {binwidthtxt} {variable_yax_unit}'
                    else: yaxtitle += ' / Bin'
                else: yaxtitle += ' / Bin'
            if normalize: yaxtitle += ' (normalized)'

            # do plotting
            fig, axs = plot(bkg=hists_sim_nominal,
                       data=data,
                       systematics=systematics,
                       variable=variable,
                       stacklist=stacklist,
                       colordict=colordict,
                       labeldict=this_labeldict,
                       styledict=styledict,
                       multdict=None,
                       normalize=normalize,
                       normalizesim=normalizesim,
                       extracmstext=extracmstext,
                       lumiheader=lumiheader,
                       yaxtitle=yaxtitle,
                       dolegend=False,
                       ratios=ratios,
                       ratio_yaxtitles=ratio_yaxtitles)

            # optional for checking: add fitted function
            showfit = False
            if showfit:
                for process_key, fitted_function in fitted_functions.items():
                    if normalize:
                        integral = np.sum(np.multiply(fitted_function, binwidths))
                        fitted_function /= integral
                    axs[0].plot(bincenters, fitted_function, color='red', linestyle=':')

            # some more plot aesthetics
            axs[0].set_ylim((0, axs[0].get_ylim()[1]*1.4))
            axs[0].legend(loc='upper right', fontsize=17, ncols=1)
            #if len(regions.keys())>1:
            #    axs[0].text(0.05, 0.9, region_name, ha='left', va='top', fontsize=12,
            #        transform=axs[0].transAxes)
            #if event_selection_name is not None:
            #    label = event_selection_name
            #    if select_processes is not None and len(select_processes)>0:
            #        label += ' (for {})'.format(', '.join(select_processes))
            #    axs[0].text(0.05, 0.85, label, ha='left', va='top', fontsize=12,
            #      transform=axs[0].transAxes)
            if normalizesim:
                axs[0].text(0.05, 0.8, 'Simulation normalized to data', ha='left', va='top', fontsize=15,
                  transform=axs[0].transAxes)
            # data ratio pad
            #if datatag is not None: axs[1].set_ylim((0, 2))

            # save the figure
            fig.tight_layout()
            figname = region_name + '_' + variable.name + '.png'
            figname = os.path.join(outputdir, figname)
            if not os.path.exists(outputdir): os.makedirs(outputdir)
            fig.savefig(figname)
            fig.savefig(figname.replace('.png', '.pdf'))
            plt.close(fig)
            print(f'Figure saved to {figname}.')
            del axs
            del fig

            # same with log scale
            if dolog:
                fig, axs = plot(bkg=hists_sim_nominal,
                       data=data,
                       systematics=systematics,
                       variable=variable,
                       stacklist=stacklist,
                       colordict=colordict,
                       labeldict=this_labeldict,
                       styledict=styledict,
                       logscale=True,
                       multdict=None,
                       normalize=normalize,
                       normalizesim=normalizesim,
                       extracmstext=extracmstext,
                       lumiheader=lumiheader,
                       yaxtitle=yaxtitle,
                       dolegend=False,
                       ratios=ratios,
                       ratio_yaxtitles=ratio_yaxtitles)

                # optional for checking: add fitted function
                if showfit:
                    for process_key, fitted_function in fitted_functions.items():
                        if normalize:
                            integral = np.sum(np.multiply(fitted_function, binwidths))
                            fitted_function /= integral
                        axs[0].plot(bincenters, fitted_function, color='red', linestyle=':')

                # some more plot aesthetics
                if np.any(histarray > 0):
                    if not normalize: ymin = np.min(histarray[np.nonzero(histarray)])
                    else: ymin = axs[0].get_ylim()[0]
                    axs[0].set_ylim((ymin, axs[0].get_ylim()[1]**1.4))
                axs[0].legend(loc='upper right', fontsize=17, ncols=1)
                #if len(regions.keys())>1:
                #    axs[0].text(0.05, 0.9, region_name, ha='left', va='top', fontsize=12,
                #        transform=axs[0].transAxes)
                #if event_selection_name is not None:
                #    label = event_selection_name
                #    if select_processes is not None and len(select_processes)>0:
                #        label += ' (for {})'.format(', '.join(select_processes))
                #    axs[0].text(0.05, 0.85, label, ha='left', va='top', fontsize=12,
                #      transform=axs[0].transAxes)
                if normalizesim:
                    axs[0].text(0.05, 0.8, 'Simulation normalized to data', ha='left', va='top', fontsize=15,
                      transform=axs[0].transAxes)
                # data ratio pad
                #if datatag is not None: axs[1].set_ylim((0, 2))

                # save the figure
                fig.tight_layout()
                figname = region_name + '_' + variable.name + '_log.png'
                figname = os.path.join(outputdir, figname)
                fig.savefig(figname)
                fig.savefig(figname.replace('.png', '.pdf'))
                plt.close(fig)
                print(f'Figure saved to {figname}.')
                del axs
                del fig


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True, nargs='+')
    parser.add_argument('-d', '--data', default=None, nargs='+')
    parser.add_argument('-v', '--variables', required=True, nargs='+')
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--objectselection', default=None)
    parser.add_argument('--eventselection', default=None)
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--sqrts', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--normalizesim', default=False, action='store_true')
    parser.add_argument('--shapes', default=False, action='store_true')
    parser.add_argument('--dolog', default=False, action='store_true')
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
        event_selection_name = list(eventselection.keys())[0]
        eventselection = eventselection[event_selection_name]

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

    # read variables
    variables = sum([read_variables(f) for f in args.variables], [])
    variablelist = []
    for variable in variables:
        if isinstance(variable, DoubleHistogramVariable):
            variablelist.append(variable.primary.variable)
            variablelist.append(variable.secondary.variable)
        else:
            variablelist.append(variable.variable)
    variablelist = sum([get_variable_names(v) for v in variablelist], [])
    variablelist = list(set(variablelist))

    # get luminosity and center-of-mass energy from year
    luminosity = args.luminosity
    sqrts = args.sqrts
    if args.year is not None:
        lumi_from_year = get_lumidict()[args.year]
        sqrts_from_year = get_sqrtsdict()[args.year]
        if args.luminosity is None or args.luminosity < 0:
            luminosity = lumi_from_year
        elif luminosity!=lumi_from_year:
            msg = f'WARNING: found inconsistency between provided luminosity ({luminosity})'
            msg += f' and the one corresponding to the provided year ({args.year}: {lumi_from_year}).'
            print(msg)
        if args.sqrts is None or args.sqrts < 0:
            sqrts = sqrts_from_year
        elif sqrts!=sqrts_from_year:
            msg = f'WARNING: found inconsistency between provided sqrt(s) ({sqrts})'
            msg += f' and the one corresponding to the provided year ({args.year}: {sqrts_from_year}).'
            print(msg)
    if luminosity < 0: luminosity = None
    if sqrts < 0: sqrts = None

    # define variables to read
    branches_to_read = []
    # add selection
    if objectselection is not None:
        branches_to_read += get_variable_names(objectselection[0])
    if eventselection is not None:
        branches_to_read += get_variable_names(eventselection)
    # add variables to plot
    branches_to_read += variablelist[:]
    # add variables needed for splitting
    if splitdict is not None:
        for splitkey, this_splitdict in splitdict.items():
            for selection_string in this_splitdict.values():
                branches_to_read += get_variable_names(selection_string)
    # remove potential duplicates
    branches_to_read = list(set(branches_to_read))
    print('Found following branches to read:')
    print(branches_to_read)

    # make histograms
    dtypedict = {'sim': sampledict_sim, 'data': sampledict_data}
    hists_combined = make_histograms(dtypedict, variables,
                       branches_to_read = branches_to_read,
                       objectselection = objectselection,
                       eventselection = eventselection,
                       splitdict = splitdict,
                       weight_variations = weight_variations,
                       lumi = luminosity,
                       xsections = xsections)

    # check number of data categories
    # (only one is supported for now)
    datatag = None
    if sampledict_data is not None:
        keys = list(sampledict_data.keys())
        if len(keys)==1: datatag = keys[0]
        else:
            msg = f'Found unexpected number of data categories: {keys}'
            raise Exception(msg)

    # plot aesthetics settings
    extracmstext = 'Archived Data'
    if args.data is None: extracmstext = 'Archived Sim.'
    lumiheaderparts = []
    if args.year is not None:
        lumiheaderparts.append(args.year)
    if luminosity is not None:
        lumiheaderparts.append('{:.1f}'.format(luminosity) + ' pb$^{-1}$')
    if sqrts is not None:
        lumiheaderparts.append('{:.1f}'.format(sqrts) + ' GeV')
    lumiheader = ', '.join(lumiheaderparts)

    # make output directory
    if not os.path.exists(args.outputdir): os.makedirs(args.outputdir)

    # plotting loop
    plot_hists(hists_combined, variables, args.outputdir,
      datatag=datatag,
      shapes=args.shapes, normalizesim=args.normalizesim, dolog=args.dolog,
      extracmstext=extracmstext, lumiheader=lumiheader,
      event_selection_name=event_selection_name)
