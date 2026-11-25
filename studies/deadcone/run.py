# Make histograms (and default data vs. simulation plots) for the dead cone study

# This is a thin wrapper around the histogramming and plotting of analysis/plot.py,
# with the same command line arguments (except regions), plus:
#   --derived: groups of derived variables to compute on the fly (see lib/derived.py),
#   --settings: json file with settings for the derived variables (see configs/),
#   --noplots: only make and save the histograms,
#   --samples: sample list json of the form {"sim": [patterns], "data": [patterns]} (instead of -s/-d),
#   -r/--runmode: 'local' or 'condor' (submit this exact command as a condor job).
# The histograms are saved in <outputdir>/hists.pkl (with the settings used),
# for further processing with plot_results.py.

import os
import sys
import json
import pickle
import glob
import argparse
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)
sys.path.append(os.path.join(thisdir, 'lib'))

from tools.variabletools import read_variables
from tools.samplelisttools import find_files
from tools.lumitools import get_lumidict, get_sqrtsdict
from tools.plottools import merge_sampledict
import tools.condortools as ct
from analysis.eventselection import load_eventselection, get_variable_names
from analysis.objectselection import load_objectselection
from analysis.plot import make_histograms, plot_hists_default
from derived import get_derived_variables, GROUPS

plt.rcdefaults()


def find_sampledict(sampledirs, maxfiles=None):
    # as in analysis/plot.py: use 'files.json' if present, else all .root files in the directory
    sampledirs = [os.path.join(d, 'files.json') if os.path.exists(os.path.join(d, 'files.json')) else d
                  for d in sampledirs]
    sampledict = find_files(sampledirs, verbose=False)
    if maxfiles is not None:
        # keep only the first files (sorted by name) of the full list
        allfiles = sorted(f for files in sampledict.values() for f in files)[:maxfiles]
        sampledict = {k: [f for f in v if f in allfiles] for k, v in sampledict.items()}
        sampledict = {k: v for k, v in sampledict.items() if len(v) > 0}
    return sampledict


def read_samplelist(path):
    # sample list json of the form {"sim": [patterns], "data": [patterns]} (as in analysis/configs),
    # with the patterns (unix-style wildcards) expanded to the individual files
    with open(path, 'r') as f: samples = json.load(f)
    res = {}
    for dtype in ['sim', 'data']:
        if samples.get(dtype) is None: continue
        res[dtype] = sorted(set(sum([sorted(glob.glob(p)) for p in samples[dtype]], [])))
        if len(res[dtype]) == 0: raise Exception(f'No {dtype} files found for patterns {samples[dtype]}')
    return res


def load_json(path, description):
    if path is None: return None
    with open(path, 'r') as f: content = json.load(f)
    print(f'Found following {description}:')
    print(json.dumps(content, indent=2))
    return content


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Dead cone study: make histograms')
    parser.add_argument('-s', '--sim', default=None, nargs='+')
    parser.add_argument('-d', '--data', default=None, nargs='+')
    parser.add_argument('--samples', default=None,
      help='Sample list json with keys "sim" and "data" (alternative to -s and -d)')
    parser.add_argument('-v', '--variables', required=True, nargs='+')
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--derived', default=[], nargs='+', choices=list(GROUPS.keys()))
    parser.add_argument('--settings', default=os.path.join(thisdir, 'configs', 'settings_default.json'))
    parser.add_argument('--objectselection', default=None, nargs='+')
    parser.add_argument('--eventselection', default=None, nargs='+')
    parser.add_argument('--select_processes', default=[], nargs='+')
    parser.add_argument('--external_variables', default=None)
    parser.add_argument('--files_per_batch', default=None)
    parser.add_argument('--maxfiles_sim', default=None, type=int,
      help='Use only the first n simulation files (for quick tests)')
    parser.add_argument('--maxfiles_data', default=None, type=int,
      help='Use only the first n data files (for quick tests)')
    parser.add_argument('--year', default=None)
    parser.add_argument('--luminosity', default=-1, type=float)
    parser.add_argument('--sqrts', default=-1, type=float)
    parser.add_argument('--xsections', default=None)
    parser.add_argument('--merge', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--normalizesim', default=False, action='store_true')
    parser.add_argument('--shapes', default=False, action='store_true')
    parser.add_argument('--dolog', default=False, action='store_true')
    parser.add_argument('--noplots', default=False, action='store_true')
    parser.add_argument('-r', '--runmode', default='local', choices=['local', 'condor'])
    parser.add_argument('--condor_cpus', default=2, type=int)
    parser.add_argument('--condor_mem', default=4000, type=int, help='Memory for condor job (MB)')
    args = parser.parse_args()

    # if requested, submit this exact command (with runmode switched to 'local')
    # as a condor job instead of running locally
    if args.runmode == 'condor':
        env_script = os.path.join(topdir, 'setup.sh')
        # (the resubmission only recognizes the long form of the runmode flag)
        argv = ['--runmode' if a == '-r' else a for a in sys.argv]
        ct.submitScriptAsCondorJob('cjob_deadcone', argv, env_script=env_script,
          cpus=args.condor_cpus, mem=args.condor_mem)
        sys.exit()

    # input samples: either a sample list or sample directories
    if args.samples is not None:
        if args.sim is not None or args.data is not None:
            raise Exception('Use either --samples or -s/-d, not both.')
        samplelist = read_samplelist(args.samples)
        args.sim = samplelist.get('sim')
        args.data = samplelist.get('data')
    if args.sim is None: raise Exception('No simulation samples given (use -s or --samples).')

    # read selections
    objectselection = None
    if args.objectselection is not None:
        objectselection = [load_objectselection(f) for f in args.objectselection]
    eventselection = None
    select_processes = None
    event_selection_name = None
    if args.eventselection is not None:
        eventselection = load_eventselection(args.eventselection, nexpect=1)
        event_selection_name = list(eventselection.keys())[0]
        eventselection = eventselection[event_selection_name]
        print(f'Found following extra event selection to apply: {eventselection}')
        if len(args.select_processes) > 0: select_processes = args.select_processes

    # read settings, cross-sections and merging/splitting instructions
    settings = load_json(args.settings, 'settings for derived variables')
    xsections = load_json(args.xsections, 'cross-sections')
    mergedict = load_json(args.merge, 'instructions for merging samples')
    splitdict = load_json(args.split, 'instructions for splitting samples')

    # find samples
    sampledict_sim = find_sampledict(args.sim, maxfiles=args.maxfiles_sim)
    sampledict_data = find_sampledict(args.data, maxfiles=args.maxfiles_data) if args.data is not None else None
    if mergedict is not None:
        sampledict_sim = merge_sampledict(sampledict_sim, mergedict, verbose=False)
        if sampledict_data is not None:
            sampledict_data = merge_sampledict(sampledict_data, mergedict, verbose=False)
    for sampledict in [sampledict_sim, sampledict_data]:
        if sampledict is None: continue
        for key, val in sampledict.items(): print(f'  - {key}: {len(val)} files')

    # read variables (keeping only the first of variables with the same name)
    variables = {}
    for v in sum([read_variables(f) for f in args.variables], []): variables.setdefault(v.name, v)
    variables = list(variables.values())

    # get luminosity and center-of-mass energy from year
    luminosity = args.luminosity if args.luminosity >= 0 else None
    sqrts = args.sqrts if args.sqrts >= 0 else None
    if args.year is not None:
        if luminosity is None: luminosity = get_lumidict()[args.year]
        if sqrts is None: sqrts = get_sqrtsdict()[args.year]

    # derived variables
    derived_funcs, derived_names, derived_inputs = get_derived_variables(args.derived, settings)

    # define branches to read
    # (derived variables are calculated on the fly, so read their inputs instead)
    expressions = [v.variable for v in variables]
    if objectselection is not None: expressions += [sel for sel, _ in objectselection]
    if eventselection is not None: expressions.append(eventselection)
    if splitdict is not None:
        for this_splitdict in splitdict.values(): expressions += list(this_splitdict.values())
    branches_to_read = set(sum([get_variable_names(e) for e in expressions], []))
    branches_to_read = sorted((branches_to_read - set(derived_names)) | set(derived_inputs))
    print(f'Found following branches to read: {branches_to_read}')

    # make histograms
    dtypedict = {'sim': sampledict_sim, 'data': sampledict_data}
    hists_combined = make_histograms(dtypedict, variables,
                       branches_to_read = branches_to_read,
                       files_per_batch = args.files_per_batch,
                       objectselection = objectselection,
                       eventselection = eventselection,
                       select_processes = select_processes,
                       external_variables = args.external_variables,
                       derived_variables = derived_funcs,
                       splitdict = splitdict,
                       lumi = luminosity,
                       xsections = xsections)

    # save histograms
    os.makedirs(args.outputdir, exist_ok=True)
    histfile = os.path.join(args.outputdir, 'hists.pkl')
    with open(histfile, 'wb') as f:
        pickle.dump({'hists': hists_combined, 'settings': settings, 'args': vars(args),
                     'variables': [v.to_dict() for v in variables]}, f)
    print(f'Saved histograms to {histfile}')
    if args.noplots: sys.exit()

    # check number of data categories (only one is supported, as in analysis/plot.py)
    datatag = None
    if sampledict_data is not None:
        keys = list(sampledict_data.keys())
        if len(keys) != 1: raise Exception(f'Found unexpected number of data categories: {keys}')
        datatag = keys[0]

    # default data vs. simulation plots
    lumiheaderparts = []
    if args.year is not None: lumiheaderparts.append(args.year)
    if luminosity is not None: lumiheaderparts.append('{:.1f}'.format(luminosity) + ' pb$^{-1}$')
    if sqrts is not None: lumiheaderparts.append('{:.1f}'.format(sqrts) + ' GeV')
    plot_hists_default(hists_combined, variables, os.path.join(args.outputdir, 'plots'),
      datatag=datatag,
      shapes=args.shapes, normalizesim=args.normalizesim, dolog=args.dolog,
      extracmstext='Archived Data', lumiheader=', '.join(lumiheaderparts),
      event_selection_name=event_selection_name, select_processes=select_processes,
      axtitlesize=15)
