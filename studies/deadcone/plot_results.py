# Flavour comparisons for the dead cone study

# Reads the histograms made by run.py and makes the following plots:
#   - xi_charged.png, xi_prompt.png: xi_p spectra of uds, c and b jets and their ratios to uds,
#     for all charged constituents (compared to OPAL, Eur. Phys. J. C7 (1999) 369) and without displaced tracks;
#   - xi_double_ratio.png: effect of removing the displaced tracks: b / uds with and without them,
#     the fraction of tracks kept per flavour, and the double ratio of b / uds without and with them;
#   - lund_ratios.png: ln(1/theta) distributions of b and uds jets and their ratios R(theta) = b / uds
#     (and c / uds in simulation), per radiator energy bin, with the dead cone angles m_Q/E,
#     for one or more input files;
#   - lund_variations.png: R(theta) in simulation for several splitting cuts;
#   - lund_scaling.png: R in simulation for all radiator energy bins overlaid, as a function of ln(1/theta),
#     of the scaled angle ln(E/(m_Q theta)) (where a dead cone gives the same curve for all energies),
#     and of ln(kt).
#
# Momentum spectra are normalized per jet (and multiplied by 2 for the per-event xi_p spectra).
# Declustering spectra are normalized per radiator, i.e. per jet with at least one splitting
# in the radiator energy bin (if these counters are available, else per jet).
# In simulation they are obtained from the true event flavour.
# In data they are obtained from the hemisphere-opposite tagged samples,
# corrected bin by bin with simulation:
#   D_b(data) = D_btag(data) * D_b(sim) / D_btag(sim), and similarly for uds with the light tag.

import os
import sys
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(thisdir, 'lib'))

from variables import ERAD_BINS, KTMIN, LUND_CUT_VARIATIONS, LUND_CUT_LABELS

plt.rcdefaults()

FLAVOURS = {'b': ['bb'], 'c': ['cc'], 'uds': ['uudd', 'ss']}
TAGGED = {'b': 'btag', 'uds': 'lighttag'}
MASSES = {'b': 4.8, 'c': 1.5}
COLORS = {'b': 'darkorchid', 'c': 'slateblue', 'uds': 'darkturquoise'}
LABELS = {'b': 'b', 'c': 'c', 'uds': 'uds'}


class Spectra(object):
    '''Per-jet normalized spectra from the histograms saved by run.py.'''

    def __init__(self, histfile):
        with open(histfile, 'rb') as f: content = pickle.load(f)
        self.hists = content['hists']
        self.settings = content['settings']
        self.bins = {v['name']: np.linspace(v['xlow'], v['xhigh'], v['nbins']+1) for v in content['variables']}
        self.has_data = 'data' in self.hists and len(self.hists['data']) > 0

    def _sum(self, dtype, var, processes=None):
        hists = self.hists[dtype][f'baseline_{var}']
        if processes is None: processes = list(hists.keys())
        counts = sum(hists[p]['nominal'][0] for p in processes)
        errors = np.sqrt(sum(hists[p]['nominal'][1]**2 for p in processes))
        return counts, errors

    def normalization(self, var):
        ### name of the counter variable to normalize a variable to (with {sample} placeholder)
        if var.startswith(('lund', 'lnkt')):
            ebin = var.split('_')[-1]
            if f'nrad_incl_{ebin}' in self.bins: return 'nrad_{sample}_' + ebin
        return 'njets_{sample}'

    def spectrum(self, dtype, var, sample, processes=None):
        ### (1/N) dN/dx, with statistical errors (N = number of jets or radiators)
        counts, errors = self._sum(dtype, f'{var.format(sample=sample)}', processes)
        njets = self._sum(dtype, self.normalization(var).format(sample=sample), processes)[0].sum()
        width = np.diff(self.bins[var.format(sample=sample)])
        return counts / njets / width, errors / njets / width

    def truth(self, var, flavour):
        ### flavour spectrum in simulation (true event flavour, all jets)
        return self.spectrum('sim', var, 'incl', FLAVOURS[flavour])

    def corrected_data(self, var, flavour):
        ### flavour spectrum in data from the tagged sample, corrected bin by bin with simulation
        tag = TAGGED[flavour]
        dtag, dtag_err = self.spectrum('data', var, tag)
        stag, stag_err = self.spectrum('sim', var, tag)
        struth, struth_err = self.truth(var, flavour)
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = struth / stag
            value = dtag * factor
            relerr = np.sqrt(_rel(dtag, dtag_err)**2 + _rel(stag, stag_err)**2 + _rel(struth, struth_err)**2)
        return value, value * relerr


def _rel(value, error):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(value > 0, error / value, 0)


def _ratio(num, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        value = num[0] / den[0]
        return value, value * np.sqrt(_rel(*num)**2 + _rel(*den)**2)


def _step(ax, bins, values, errors=None, **kwargs):
    ax.stairs(values, bins, **kwargs)
    if errors is not None:
        with np.errstate(invalid='ignore'):
            ax.stairs(values + errors, bins, baseline=values - errors, fill=True, alpha=0.25,
                      color=kwargs.get('color'))


def _points(ax, bins, values, errors, **kwargs):
    centers = (bins[1:] + bins[:-1]) / 2
    mask = np.isfinite(values) & (values > 0)
    kwargs.setdefault('marker', 'o')
    ax.errorbar(centers[mask], values[mask], yerr=errors[mask], linestyle='none', markersize=4, **kwargs)


def load_opal(path):
    with open(path) as f: ref = json.load(f)
    bins = np.array([b[0] for b in ref['bins']] + [ref['bins'][-1][1]])
    res = {}
    for flavour in ['uds', 'b']:
        value = np.array(ref[flavour]['value'])
        error = np.sqrt(np.array(ref[flavour]['stat'])**2 + np.array(ref[flavour]['syst'])**2)
        res[flavour] = (value, error)
    return bins, res


def plot_xi(spectra, particles, outputfile, opal=None):
    ### xi_p spectra per event (= 2 x per jet) for b and uds, and the ratio b / uds
    var = '{sample}' + f'_{particles}'
    var = 'xi_' + var
    bins = spectra.bins[var.format(sample='incl')]
    fig, axs = plt.subplots(nrows=2, figsize=(7, 7), height_ratios=[2, 1], sharex=True)
    curves = {}
    for flavour in ['uds', 'c', 'b']:
        value, error = spectra.truth(var, flavour)
        curves[('sim', flavour)] = (2*value, 2*error)
        _step(axs[0], bins, 2*value, 2*error, color=COLORS[flavour], label=f'Simulation, {LABELS[flavour]}')
    if spectra.has_data:
        for flavour in ['uds', 'b']:
            value, error = spectra.corrected_data(var, flavour)
            curves[('data', flavour)] = (2*value, 2*error)
            _points(axs[0], bins, 2*value, 2*error, color=COLORS[flavour], label=f'Data (tag-corrected), {LABELS[flavour]}')
    if opal is not None:
        obins, ovals = opal
        for flavour in ['uds', 'b']:
            _points(axs[0], obins, *ovals[flavour], color=COLORS[flavour], marker='s', mfc='none',
                    label=f'OPAL (hadron level), {LABELS[flavour]}')
    axs[0].set_ylabel(r'$1/N_{ev}\, dN_{ch}/d\xi_p$', fontsize=15)
    axs[0].legend(fontsize=9, loc='upper left')
    title = 'all charged constituents' if particles == 'charged' else 'charged, displaced tracks removed'
    axs[0].set_title(f'Momentum spectra ({title})', fontsize=13)
    # ratios
    for flavour in ['c', 'b']:
        _step(axs[1], bins, *_ratio(curves[('sim', flavour)], curves[('sim', 'uds')]), color=COLORS[flavour],
              label=f'Simulation, {LABELS[flavour]} / uds')
    if spectra.has_data:
        _points(axs[1], bins, *_ratio(curves[('data', 'b')], curves[('data', 'uds')]), color='black',
                label='Data (tag-corrected), b / uds')
    if opal is not None:
        _points(axs[1], obins, *_ratio(ovals['b'], ovals['uds']), color='grey', marker='s', mfc='none',
                label='OPAL, b / uds')
    axs[1].axhline(1, color='grey', linestyle='--')
    axs[1].set_ylim(0, 2)
    axs[1].set_ylabel('Ratio to uds', fontsize=15)
    axs[1].set_xlabel(r'$\xi_p = \ln(1/x_p)$', fontsize=15)
    axs[1].legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(outputfile)
    plt.close(fig)
    print(f'Saved {outputfile}')
    return curves


def plot_xi_double_ratio(spectra, outputfile):
    ### effect of removing the displaced tracks on the xi_p spectra:
    ###   top: b / uds for all charged constituents and without displaced tracks;
    ###   middle: fraction of tracks kept (without displaced / all charged) per flavour;
    ###   bottom: double ratio (b / uds without displaced) / (b / uds all charged),
    ###           i.e. the kept fraction in b relative to uds.
    ### (Uncertainties are propagated as if the two spectra were independent,
    ### which overestimates them since one is a subset of the other.)
    bins = spectra.bins['xi_incl_charged']
    var = {p: 'xi_{sample}_' + p for p in ['charged', 'prompt']}
    sim = {(p, f): spectra.truth(var[p], f) for p in var for f in ['uds', 'c', 'b']}
    data = {}
    if spectra.has_data:
        data = {(p, f): spectra.corrected_data(var[p], f) for p in var for f in ['uds', 'b']}
    fig, axs = plt.subplots(nrows=3, figsize=(7, 9), height_ratios=[1, 1, 1], sharex=True)
    styles = {'charged': ('-', 'o', 'all charged'), 'prompt': ('--', 's', 'displaced removed')}
    # b / uds for both particle selections
    for p, (ls, marker, label) in styles.items():
        _step(axs[0], bins, *_ratio(sim[(p, 'b')], sim[(p, 'uds')]), color=COLORS['b'], linestyle=ls,
              label=f'Simulation, {label}')
        if data:
            _points(axs[0], bins, *_ratio(data[(p, 'b')], data[(p, 'uds')]), color='black', marker=marker,
                    mfc='black' if p == 'charged' else 'none', label=f'Data (tag-corrected), {label}')
    axs[0].set_ylabel('b / uds', fontsize=15)
    axs[0].set_ylim(0, 2)
    # fraction of tracks kept per flavour
    for f in ['uds', 'c', 'b']:
        _step(axs[1], bins, *_ratio(sim[('prompt', f)], sim[('charged', f)]), color=COLORS[f],
              label=f'Simulation, {LABELS[f]}')
    for f in ['uds', 'b'] if data else []:
        _points(axs[1], bins, *_ratio(data[('prompt', f)], data[('charged', f)]), color=COLORS[f],
                label=f'Data (tag-corrected), {LABELS[f]}')
    axs[1].set_ylabel('Fraction kept', fontsize=15)
    axs[1].set_ylim(0.4, 1.1)
    # double ratio
    for f in ['c', 'b']:
        num = _ratio(sim[('prompt', f)], sim[('prompt', 'uds')])
        den = _ratio(sim[('charged', f)], sim[('charged', 'uds')])
        _step(axs[2], bins, *_ratio(num, den), color=COLORS[f], label=f'Simulation, {LABELS[f]}')
    if data:
        num = _ratio(data[('prompt', 'b')], data[('prompt', 'uds')])
        den = _ratio(data[('charged', 'b')], data[('charged', 'uds')])
        _points(axs[2], bins, *_ratio(num, den), color='black', label='Data (tag-corrected), b')
    axs[2].set_ylabel('Double ratio', fontsize=15)
    axs[2].set_ylim(0.4, 1.1)
    axs[2].set_xlabel(r'$\xi_p = \ln(1/x_p)$', fontsize=15)
    for ax in axs:
        ax.axhline(1, color='grey', linestyle='--')
        ax.legend(fontsize=8, loc='lower right')
    axs[0].set_title('Effect of removing displaced tracks', fontsize=13)
    fig.tight_layout()
    fig.savefig(outputfile)
    plt.close(fig)
    print(f'Saved {outputfile}')


def plot_lund(spectra_list, outputfile, labels):
    ### ratio R(theta) = b / uds (and c / uds in simulation) of ln(1/theta) per radiator energy bin
    nbins = len(ERAD_BINS)
    fig, axs = plt.subplots(nrows=2, ncols=nbins, figsize=(4*nbins, 7), height_ratios=[1, 1], sharex=True)
    styles = ['-', '--', ':']
    for col, (low, high) in enumerate(ERAD_BINS):
        var = 'lund_{sample}' + f'_E{low}-{high}'
        for idx, (spectra, label) in enumerate(zip(spectra_list, labels)):
            bins = spectra.bins[var.format(sample='incl')]
            ls = styles[idx % len(styles)]
            curves = {f: spectra.truth(var, f) for f in ['uds', 'c', 'b']}
            for f in ['uds', 'b']:
                _step(axs[0, col], bins, *curves[f], color=COLORS[f], linestyle=ls,
                      label=f'Sim. {LABELS[f]} ({label})')
            for f in ['c', 'b']:
                _step(axs[1, col], bins, *_ratio(curves[f], curves['uds']), color=COLORS[f], linestyle=ls,
                      label=f'Sim. {LABELS[f]}/uds ({label})')
            if spectra.has_data:
                data = {f: spectra.corrected_data(var, f) for f in ['uds', 'b']}
                _points(axs[1, col], bins, *_ratio(data['b'], data['uds']), color='black',
                        marker='o' if idx == 0 else 's', mfc='black' if idx == 0 else 'none',
                        label=f'Data b/uds ({label})')
        # dead cone angles at the bin centre energy
        ecenter = (low + high) / 2
        for f in ['b', 'c']:
            if MASSES[f] / ecenter < 1:
                for ax in axs[:, col]:
                    ax.axvline(np.log(ecenter / MASSES[f]), color=COLORS[f], linestyle='-.', alpha=0.7)
        axs[0, col].set_title(f'{low} < $E_{{rad}}$ < {high} GeV', fontsize=13)
        axs[1, col].axhline(1, color='grey', linestyle='--')
        axs[1, col].set_ylim(0, 2)
        axs[1, col].set_xlabel(r'$\ln(1/\theta)$', fontsize=14)
    axs[0, 0].set_ylabel(r'$1/N\, dN/d\ln(1/\theta)$', fontsize=14)
    axs[1, 0].set_ylabel(r'$R(\theta)$ = ratio to uds', fontsize=14)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.suptitle(f'Iterative C/A declustering (charged constituents, $k_T$ > {KTMIN} GeV);'
                 + ' dash-dotted lines: $\\ln(E/m_Q)$ for b and c', fontsize=13)
    fig.tight_layout()
    fig.savefig(outputfile)
    plt.close(fig)
    print(f'Saved {outputfile}')


def plot_lund_variations(spectra, outputfile):
    ### R(theta) = b / uds and c / uds in simulation for several splitting cuts, per radiator energy bin
    cuts = {'': LUND_CUT_LABELS['']}
    cuts.update({c: LUND_CUT_LABELS[c] for c in LUND_CUT_VARIATIONS
                 if f'lund{c}_incl_E{ERAD_BINS[0][0]}-{ERAD_BINS[0][1]}' in spectra.bins})
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(cuts)))
    nbins = len(ERAD_BINS)
    fig, axs = plt.subplots(nrows=2, ncols=nbins, figsize=(4*nbins, 7), sharex=True)
    for col, (low, high) in enumerate(ERAD_BINS):
        for (cut, label), color in zip(cuts.items(), colors):
            var = f'lund{cut}' + '_{sample}' + f'_E{low}-{high}'
            bins = spectra.bins[var.format(sample='incl')]
            curves = {f: spectra.truth(var, f) for f in ['uds', 'c', 'b']}
            for row, f in enumerate(['b', 'c']):
                _step(axs[row, col], bins, *_ratio(curves[f], curves['uds']), color=color, label=label)
        ecenter = (low + high) / 2
        for row, f in enumerate(['b', 'c']):
            ax = axs[row, col]
            if MASSES[f] / ecenter < 1: ax.axvline(np.log(ecenter / MASSES[f]), color='grey', linestyle='-.')
            ax.axhline(1, color='grey', linestyle='--')
            ax.set_ylim(0, 2)
        axs[0, col].set_title(f'{low} < $E_{{rad}}$ < {high} GeV', fontsize=13)
        axs[1, col].set_xlabel(r'$\ln(1/\theta)$', fontsize=14)
    axs[0, 0].set_ylabel('Simulation, b / uds', fontsize=14)
    axs[1, 0].set_ylabel('Simulation, c / uds', fontsize=14)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('Splitting cut variations; dash-dotted lines: $\\ln(E/m_Q)$', fontsize=13)
    fig.tight_layout()
    fig.savefig(outputfile)
    plt.close(fig)
    print(f'Saved {outputfile}')


def plot_lund_scaling(spectra, outputfile, cut=''):
    ### R(theta) for all radiator energy bins overlaid, versus ln(1/theta), ln(E/(m_Q theta)) and ln(kt);
    ### for a dead cone, the curves for different energies coincide in ln(E/(m_Q theta))
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(ERAD_BINS)))
    has_kt = f'lnkt_incl_E{ERAD_BINS[0][0]}-{ERAD_BINS[0][1]}' in spectra.bins
    ncols = 3 if has_kt else 2
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(5*ncols, 7))
    for (low, high), color in zip(ERAD_BINS, colors):
        ecenter = (low + high) / 2
        label = f'{low} < $E_{{rad}}$ < {high} GeV'
        var = f'lund{cut}' + '_{sample}' + f'_E{low}-{high}'
        bins = spectra.bins[var.format(sample='incl')]
        curves = {f: spectra.truth(var, f) for f in ['uds', 'c', 'b']}
        for row, f in enumerate(['b', 'c']):
            ratio = _ratio(curves[f], curves['uds'])
            _step(axs[row, 0], bins, *ratio, color=color, label=label)
            _step(axs[row, 1], bins - np.log(ecenter / MASSES[f]), *ratio, color=color, label=label)
            if has_kt:
                ktvar = 'lnkt_{sample}' + f'_E{low}-{high}'
                ktcurves = {g: spectra.truth(ktvar, g) for g in [f, 'uds']}
                _step(axs[row, 2], spectra.bins[ktvar.format(sample='incl')],
                      *_ratio(ktcurves[f], ktcurves['uds']), color=color, label=label)
    for row, f in enumerate(['b', 'c']):
        axs[row, 0].set_xlabel(r'$\ln(1/\theta)$', fontsize=14)
        axs[row, 1].set_xlabel(r'$\ln(E/(m_' + f + r'\theta)) = \ln(\theta_0/\theta)$', fontsize=14)
        axs[row, 1].axvline(0, color='grey', linestyle='-.')
        if has_kt: axs[row, 2].set_xlabel(r'$\ln(k_T/\mathrm{GeV})$ (no cut)', fontsize=14)
        axs[row, 0].set_ylabel(f'Simulation, {f} / uds', fontsize=14)
        for ax in axs[row]:
            ax.axhline(1, color='grey', linestyle='--')
            ax.set_ylim(0, 2)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f'Energy scaling of R ({LUND_CUT_LABELS[cut]}); dead cone: curves coincide in the middle column',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(outputfile)
    plt.close(fig)
    print(f'Saved {outputfile}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dead cone study: flavour comparison plots')
    parser.add_argument('-i', '--inputfiles', required=True, nargs='+',
      help='hists.pkl file(s) from run.py. The first one is the reference, used for all plots;'
          +' the others are only compared to it in lund_ratios.png')
    parser.add_argument('-l', '--labels', default=None, nargs='+',
      help='Labels for the input files (default: their directory names)')
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('--opal', default=os.path.join(thisdir, 'references', 'opal1999_xi.json'))
    args = parser.parse_args()

    os.makedirs(args.outputdir, exist_ok=True)
    spectra_list = [Spectra(f) for f in args.inputfiles]
    labels = args.labels or [os.path.basename(os.path.dirname(os.path.abspath(f))) for f in args.inputfiles]
    ref = spectra_list[0]
    def has(spectra, prefix): return f'{prefix}_incl_E{ERAD_BINS[0][0]}-{ERAD_BINS[0][1]}' in spectra.bins

    # momentum spectra
    if 'xi_incl_charged' in ref.bins:
        opal = load_opal(args.opal)
        # (OPAL includes heavy-hadron decay products, so it is only compared to the spectra of all charged particles)
        plot_xi(ref, 'charged', os.path.join(args.outputdir, 'xi_charged.png'), opal=opal)
        plot_xi(ref, 'prompt', os.path.join(args.outputdir, 'xi_prompt.png'))
        plot_xi_double_ratio(ref, os.path.join(args.outputdir, 'xi_double_ratio.png'))

    # declustering
    if has(ref, 'lund'):
        compare = [(s, l) for s, l in zip(spectra_list, labels) if has(s, 'lund')]
        plot_lund([s for s, _ in compare], os.path.join(args.outputdir, 'lund_ratios.png'), [l for _, l in compare])
        plot_lund_scaling(ref, os.path.join(args.outputdir, 'lund_scaling.png'))
        if any(has(ref, f'lund{c}') for c in LUND_CUT_VARIATIONS):
            plot_lund_variations(ref, os.path.join(args.outputdir, 'lund_variations.png'))
