# Print a table of purity and efficiency

import os
import sys
import numpy as np
import pandas as pd


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
        label = key
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


if __name__=='__main__':

    # set input file(s)
    inputfiles = sys.argv[1:]

    # other settings (hard-coded for now)
    sig_effs = [0.2, 0.4, 0.6, 0.8]

    # initialize table
    table = {}
    table['sig_effs'] = sig_effs

    # loop over input files
    for inputfile in inputfiles:

        # determine which category is signal based on file name
        signal = None
        if inputfile.endswith('score_isB.csv'): signal = 'b'
        elif inputfile.endswith('score_isC.csv'): signal = 'c'
        elif inputfile.endswith('score_isS.csv'): signal = 's'
        else: raise Exception('Could not determine signal category.')

        # read file and extract useful information
        purity_col = f'purity_{signal}{signal}'
        efficiency_col = f'efficiency_{signal}{signal}'
        columns_to_read = [purity_col, efficiency_col]
        df = pd.read_csv(inputfile, usecols=columns_to_read)
        purity = df[purity_col].values
        efficiency = df[efficiency_col].values

        # find purity for given signal efficiency
        table_entry = []
        for sig_eff in sig_effs:
            idx = np.nonzero(efficiency[::-1] > sig_eff)[0][0]
            this_purity = purity[::-1][idx]
            # store contamination rather than purity (easier for high purity)
            this_purity = 1 - this_purity
            table_entry.append(this_purity)
        label = r'$\PQ' + signal + r'\PAQ' + signal + '$ events'
        table[label] = table_entry

    # print results
    print(format_table_txt(table))
    print(format_table_txt_latex(table))
