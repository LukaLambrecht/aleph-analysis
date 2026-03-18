import os
import sys
import glob

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../../'))
sys.path.append(topdir)

import tools.condortools as ct
import tools.slurmtools as st


if __name__=='__main__':

    # settings
    modeltag = '20251222_fitpv_withbs'
    ntupletag = 'fitpv-withbs'
    model = os.path.abspath(f'../models/output_{modeltag}/model.onnx')
    preprocess = model.replace('model.onnx', 'preprocess.json')
    outputdir = f'output_scores_model_{modeltag}'
    runmode = 'condor'
    resubmit = False
    ntuplename = f'ntuples-{ntupletag}' if ntupletag is not None else 'ntuples'
    files = [
      f'/eos/user/l/llambrec/aleph-data/{ntuplename}/jetlevel/mc/output_qqb_*_test.root',
    ]

    # find files
    inputfiles = []
    for pattern in files:
        inputfiles += glob.glob(pattern)
    print(f'Found {len(inputfiles)} files matching patterns.')

    # filter resubmission
    if resubmit:
        resubmit_files = []
        for inputfile in inputfiles:
            outputfile = inputfile.replace('/','').replace('.root', '.pkl')
            outputfile = os.path.join(outputdir, outputfile)
            if not os.path.exists(outputfile):
                resubmit_files.append(inputfile)
        inputfiles = resubmit_files
        print(f'Found {len(inputfiles)} files for resubmission.')

    # loop over input files
    cmds = []
    for f in inputfiles:

        # make the command
        cmd = 'python inference_jetlevel.py'
        cmd += f' -s {f}'
        cmd += f' -m {model}'
        cmd += f' -p {preprocess}'
        cmd += f' -o {outputdir}'
        cmd += ' -t tree'
        cmds.append(cmd)

    # run commands
    if runmode=='local':
        for cmd in cmds:
            print(cmd)
            os.system(cmd)
    elif runmode=='condor':
        env_script = os.path.abspath('../../../../setup.sh')
        env_cmd = f'source {env_script}'
        ct.submitCommandsAsCondorCluster('cjob_inference', cmds,
          jobflavour='workday', conda_activate=env_cmd)
    elif runmode=='slurm':
        env_cmds = ([
          'source /blue/avery/llambre1.brown/miniforge3/bin/activate',
          'conda activate weaver',
          f'cd {thisdir}'
        ])
        slurmscript = 'sjob_inference.sh'
        job_name = os.path.splitext(slurmscript)[0]
        st.submitCommandsAsSlurmJobs(cmds, script=slurmscript,
                job_name=job_name, env_cmds=env_cmds,
                memory='16G', time='05:00:00', constraint='el9')
