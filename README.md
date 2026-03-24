# Simple analysis of ALEPH ntuples with new jet flavour classifier

The intention is mainly to serve as a proof of concept on how to work with the data and simulation,
including the following items:
- Simple object and event selection, and making simulation-vs-data distributions.
- Evaluating previously trained classifiers using ONNX and integrating them into the analysis.

### Input files

This analysis code is intended to run on the ntuples produced with [aleph-ntuplizer](https://github.com/LukaLambrecht/aleph-ntuplizer).
They are currently stored in `/eos/user/l/llambrec/aleph-data` (on `lxplus`), let me know if you need access.
They can also be copied over to `Oscar` / `BRUX` if requested.

### How to set up?

This code should run in a standard FCCAnalyses environment (the same that was used to produce the ntuples).
Activate it with `source /cvmfs/sw.hsf.org/key4hep/setup.sh`.
Alternatively, you can run `source setup.sh` (which is just a wrapper around the former command).

Todo: were other packages installed on top of this at some point?
To figure out and update documentation if this is the case.

Todo: this works on lxplus, but not sure if available on BRUX, to figure out.

### How to run?

The main analysis code is in `analysis/plot.py`. A small example can be run as follows:
```
cd analysis
python plot_config.py -s configs/samples_ksloose_eos.json -c configs/config_test.json -r local
```

Some remarks:
- The file `configs/samples_ksloose_eos.json` contains the path to the latest version of the ntuples (at the time of writing).
Replace it if needed with a similar file containing the path to wherever you can access the samples.
- The file `configs/config_test.json` contains a small test configuration.
In this context, configuration means a particular combination of object selection (e.g. jets above a certain momentum threshold),
event selection (e.g. hadronic Z candidates), variables to plot (e.g. dijet invariant mass), and potentially some other settings.
- Examples of object and event selections are given in the `selections` subfolder,
and examples of variables are given in the `variables` subfolder.
You can copy, modify, and extend them as needed.
- Instead of grouping these in a config file, you can also run `python plot.py` and specify all arguments directly on the command linei.
Run with `python plot.py -h` to see a list of all available options.
- The option `-r local` means running interactivaly in the terminal.
It can take a few minutes to run, as there are about 1M simulated events and about 10M data events that need to be loaded into memory.
Alternatively, use `-r condor` to run it as a condor job instead.
