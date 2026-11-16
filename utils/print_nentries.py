# Print the number of entries in a tree for each of the provided files, and the total.
# Usage: python print_nentries.py -i <file(s)> [-t <treename>]
# If no tree name is given, the first TTree (or RNTuple) found in each file is used.


import argparse
import uproot


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Print number of entries per file')
    parser.add_argument('-i', '--inputfiles', nargs='+')
    parser.add_argument('-t', '--treename', default=None)
    args = parser.parse_args()

    total = 0
    for inputfile in args.inputfiles:
        with uproot.open(inputfile) as f:
            print(f.keys())
            treename = args.treename
            if treename is None:
                treename = f.keys(filter_classname=['TTree', 'ROOT::RNTuple'], cycle=False)[0]
            nentries = f[treename].num_entries
        print(f'{inputfile}: {nentries}')
        total += nentries
    print(f'Total: {total} entries in {len(args.inputfiles)} files')
