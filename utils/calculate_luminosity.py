import os
import sys
import uproot
import numpy as np


def check_data_consistency(run_numbers, lumi_numbers):
    if len(run_numbers) != len(lumi_numbers):
        raise Exception('Run numbers and lumi numbers have different lengths.')
    ids = np.where(run_numbers != np.roll(run_numbers, 1))[0]
    for startidx, stopidx in zip(ids[:-1], ids[1:]):
        batch_runs = run_numbers[startidx : stopidx]
        batch_lumi = lumi_numbers[startidx : stopidx]
        if len(np.unique(batch_runs))!=1:
            raise Exception('Something wrong in run selection')
        run = batch_runs[0]
        if startidx > 0:
            previous_run = run_numbers[startidx-1]
            if previous_run >= run:
                msg = f'WARNING: run at idx {startidx} ({run}) unexpected'
                msg += f' compared to {startidx-1} ({previous_run})'
                print(msg)
        if len(np.unique(batch_lumi))!=1:
            print(f'WARNING: lumi values for run {run} are not unique.')
        lumi = batch_lumi[0]
        if startidx > 0:
            previous_lumi = lumi_numbers[startidx-1]
            previous_run = run_numbers[startidx-1]
            if lumi > 1e-6 and abs(lumi - previous_lumi) < 1e-6:
                msg = 'WARNING: lumi values seem to be suspiciously close for different runs:'
                msg += f' {lumi} (for run {run})'
                msg += f' vs {previous_lumi} (for run {previous_run})'
                print(msg)


def check_lumi_consistency(lcals, sicals):
    for lcal, sical in zip(lcals, sicals):
        if abs(lcal - sical) > 1e-6:
            print('WARNING: LCAL and SICAL values do not agree.')


if __name__=='__main__':

    input_files = sorted(sys.argv[1:])

    # initializations
    lumidict_lcal = {}
    lumidict_sical = {}
    lumi_lcal = 0.
    lumi_sical = 0.

    # loop over input files
    for idx, input_file in enumerate(input_files):
        print(f'Reading file {idx+1} / {len(input_files)}: {input_file}')
        with uproot.open(input_file+':events') as f:
            
            # get luminosity from RunInformation array
            # (see more info here: https://aleph-new.docs.cern.ch/eos/1994-data/)
            runinfo = f['RunInformation'].array()
            this_lumi_lcal = runinfo[:,5].to_numpy()
            this_lumi_sical = runinfo[:,6].to_numpy()
            # note: at this point, the luminosity is an array of shape (nevents),
            # with for each event the luminosity of the run the event belongs to (?);
            # so we also need to get the run numbers to calculate the total luminosity.

            # get run numbers
            run_numbers = f['EventHeader.runNumber'].array().to_numpy()
            run_numbers = np.squeeze(run_numbers)

            # find indices where a new run starts
            ids = np.where(run_numbers != np.roll(run_numbers, 1))[0]

            # printouts for debugging
            #print('  Idx -> run number')
            #for idx in ids:
            #    print(f'  - {idx} -> {run_numbers[idx]}')
            
            # do a full check if all information seems to be consistent
            # (usually disabled for speed, enable for debugging)
            check_data_consistency(run_numbers, this_lumi_lcal)
            check_data_consistency(run_numbers, this_lumi_sical)

            # select lumi per run instead of per event
            this_lumi_lcal = this_lumi_lcal[ids]
            this_lumi_sical = this_lumi_sical[ids]

            # check consistency between lcal and sical
            # (usually disabled for speed, enable for debugging)
            #check_lumi_consistency(this_lumi_lcal, this_lumi_sical)

            # add the per-run values and add to total
            this_lumi_lcal = np.sum(this_lumi_lcal)
            this_lumi_sical = np.sum(this_lumi_sical)
            lumidict_lcal[input_file] = this_lumi_lcal
            lumidict_sical[input_file] = this_lumi_sical
            lumi_lcal += this_lumi_lcal
            lumi_sical += this_lumi_sical

    print()
    print('Luminosity per input file:')
    for input_file in input_files:
        lcal = lumidict_lcal[input_file]
        sical =  lumidict_sical[input_file]
        print(f'  - {input_file}: {lcal} (LCAL) / {sical} (SICAL)')
    print(f'Total luminosity: {lumi_lcal} (LCAL) / {lumi_sical} (SICAL)')
