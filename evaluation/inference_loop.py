import os
import sys
import six
import glob

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

import tools.condortools as ct
import tools.slurmtools as st


if __name__=='__main__':

    # settings
    #modeltag = '20260305_withnewks_withdedx_masked_standardized'
    #ntupletag = 'withnewks'
    #model = os.path.abspath(f'models/output_{modeltag}/model.onnx')
    #preprocess = model.replace('model.onnx', 'preprocess.json')
    #outputdir = f'/eos/user/l/llambrec/aleph-data/model_output_scores/output_scores_model_{modeltag}'
    #ntuplename = f'ntuples-{ntupletag}' if ntupletag is not None else 'ntuples'
    #files = [
    #  f'/eos/user/l/llambrec/aleph-data/{ntuplename}/eventlevel/mc/output_qqb_*.root',
    #  f'/eos/user/l/llambrec/aleph-data/{ntuplename}/eventlevel/data/output_data_*.root',
    #]
    #translation = 'translations/translations.json'
    # settings for the models trained on the vtag01 ntuples,
    # applied on the vtag01 version of the data and of the official ALEPH simulation
    # (the vtag01 ntuples are translated to our naming convention while reading, see tools/schematools.py)
    modeltag = '20261013_vtag01_nepochs_480'
    model = '/eos/user/l/llambrec/aleph-weaver-output/output_test_20261013_vtag01_nepochs_480_nsteps_300_batch_1024/model.onnx'
    preprocess = model.replace('model.onnx', 'preprocess.json')
    outputdir = f'/eos/user/l/llambrec/aleph-data/model_output_scores/output_scores_model_{modeltag}'
    files = [
      '/eos/experiment/fcc/ee/analyses/case-studies/aleph/processedMC/1994/zqq/stage1/prefix/VTAG-01-tagged/*/QQB/*.root',
      '/eos/experiment/fcc/ee/analyses/case-studies/aleph/processedData/1994/stage1/prefix/VTAG-01-data/1994/*.root',
    ]
    translation = 'translations/translations_vtag01.json'
    runmode = 'condor'
    resubmit = True

    # set test mode
    test = False

    # find files
    inputfiles = []
    for pattern in files:
        inputfiles += glob.glob(pattern)
    print(f'Found {len(inputfiles)} files matching patterns.')

    # find model
    if not os.path.exists(model):
        raise Exception(f'Model {model} does not exist.')

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
        print(f'Continue? (y/n)')
        go = six.moves.input()
        if go!='y': sys.exit()

    # check number of files to (re)submit
    if len(inputfiles)==0:
        msg = 'Found 0 input files, exiting...'
        print(msg)
        sys.exit()

    # loop over input files
    cmds = []
    for f in inputfiles:

        # set object selection
        # update: do not do object (i.e. jet) selection anymore,
        #         as we want to store the scores for individual jets as well
        #         rather than per-event combined scores only;
        #         but then the jet selection needs to be applied at a later stage
        #         when making the per-event combined scores!
        #objectselection = 'selections/selection_jets.json'
        objectselection = None

        # make the command
        cmd = 'python inference.py'
        cmd += f' -s {f}'
        cmd += f' -m {model}'
        cmd += f' -p {preprocess}'
        cmd += f' -o {outputdir}'
        cmd += ' -t events'
        if objectselection is not None: cmd += f' --objectselection {objectselection}'
        cmd += f' --translation {translation}'
        # (limit the memory usage of the model inference to about 1 GB)
        cmd += ' --inference_batch_size 100'
        cmds.append(cmd)

    # test mode
    if test:
        print('WARNING: test mode is set to True.')
        cmds = [cmds[0]]
        runmode = 'local'

    # run commands
    if runmode=='local':
        for cmd in cmds:
            print(cmd)
            os.system(cmd)
    elif runmode=='condor':
        conda_activate = 'export PATH=/eos/user/l/llambrec/miniforge3/envs/weaver/bin:$PATH'
        ct.submitCommandsAsCondorCluster('cjob_inference', cmds,
          jobflavour='workday', conda_activate=conda_activate)
    elif runmode=='slurm':
        env_cmds = ([
          'export PATH=/blue/avery/llambre1.brown/miniforge3/envs/weaver/bin:$PATH',
          f'cd {thisdir}'
        ])
        slurmscript = 'sjob_inference.sh'
        job_name = os.path.splitext(slurmscript)[0]
        st.submitCommandsAsSlurmJobs(cmds, script=slurmscript,
                job_name=job_name, env_cmds=env_cmds,
                memory='16G', time='05:00:00', constraint='el9')
