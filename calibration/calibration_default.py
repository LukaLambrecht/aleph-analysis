# Run some standard calibration commands for default case


import os
import sys
import json
import argparse

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

import tools.condortools as ct
import tools.slurmtools as st


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim', required=True)
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-o', '--outputdir', required=True)
    parser.add_argument('-e', '--external_variables', required=True)
    parser.add_argument('-b', '--calibration_branch', required=True)
    parser.add_argument('-r', '--runmode', default='local', choices=['local', 'condor'])
    args = parser.parse_args()

    # initialize commands per job
    cmds = []

    # make calibration
    cmd = 'python calibration.py'
    cmd += f' -s {args.sim}'
    cmd += f' -d {args.data}'
    cmd += f' -o {args.outputdir}'
    cmd += f' --external_variables {args.external_variables}'
    cmd += f' --objectselection ../analysis/selections/selection_jets.json'
    cmd += f' --eventselection ../analysis/selections/selection.json'
    cmd += f' --year 1994 --xsections ../analysis/cross-sections/cross_sections.json'
    cmd += f' --merge ../analysis/merging/merging.json --split ../analysis/merging/splitting.json'
    cmds.append(cmd)

    # apply calibration
    cmd = 'python apply_calibration.py'
    cmd += f' -s {args.sim}'
    cmd += f' -d {args.data}'
    cmd += f' -c {os.path.join(args.outputdir, "output.json")}'
    cmd += f' -b {args.calibration_branch}'
    cmd += f' -o {args.outputdir}'
    cmd += f' --external_variables {args.external_variables}'
    cmd += f' --objectselection ../analysis/selections/selection_jets.json'
    cmd += f' --eventselection ../analysis/selections/selection.json'
    cmd += f' --regions ../analysis/selections/regions_flavour_enriched.json --recalculate_regions'
    cmd += f' --year 1994 --xsections ../analysis/cross-sections/cross_sections.json'
    cmd += f' --merge ../analysis/merging/merging.json --split ../analysis/merging/splitting.json'
    cmd += f' -v ../analysis/variables/variables_scores_withstrange.json ../analysis/variables/variables_event.json'
    cmd += f' --dolog'
    cmds.append(cmd)

    # run or submit commands
    if args.runmode=='local':
        for cmd in cmds:
            print(cmd)
            os.system(cmd)
    elif args.runmode=='condor':
        env_script = os.path.abspath('../../../setup.sh')
        env_cmd = f'source {env_script}'
        ct.submitCommandsAsCondorJob('cjob_analysis', cmds,
            jobflavour='workday', conda_activate=env_cmd)
