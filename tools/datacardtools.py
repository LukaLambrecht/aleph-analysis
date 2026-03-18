############################################################
# tools for automatically generating datacards for combine #
############################################################

import os
import sys
import re
import ROOT

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

import tools.roothisttools as ht


def makecolumn( channel, process, pid, pyield, systematicsdict, systematicslist ):
    ### get list of strings representing one column in the datacard.
    # input arguments:
    # - channel: name of the channel (= first row) (called 'bin' in combine)
    # - process: name of the process (= second row)
    # - pid and pyield: id number and expected yield of this process
    # - systematicsdict: dict mapping systematics for this process to either '1' or '-'
    # - systematicslist: list of systematics
    #   - note: all columns (also the first one listing the names of the systematics)
    #     must be made using the same systematicslist, also same order,
    #     else the marks ('1' or '-') will not be in the correct row
    #   - note: all names in systematicslist are expected to be in systematicsdict,
    #     either with '1' (str or float) or '-' as value
    #   - this is for shape systematics only, for flat ones, use makerow instead of makecolumn
    # output:
    # - list of strings (in correct order) corresponding to the rows of the column
    column = []
    column.append(str(channel))
    column.append(str(process))
    column.append(str(pid))
    column.append(str(pyield))
    for systematic in systematicslist:
        column.append(str(systematicsdict[systematic]))
    return column

def makerateparamrow( process ):
    ### return a row for a datacard adding a rateParam for the given process
    # note: maybe extend functionality later, for now just adding a line
    #       rate_<process> rateParam * <process> 1.0 [0.0,3.0]
    # warning: it should be checked in the calling function if process is a valid name!
    rpstr = 'rate_{} rateParam * {} 1.0 [0.0,3.0]'.format(process,process)
    return rpstr

def getseparator( length=20, endline=True ):
    ### return a string usable as horizontal separator in the card
    # optionally include a newline char at the end
    sep = '-'*length
    if endline: sep += '\n'
    return sep

def makealigned(stringlist):
    ### append spaces to all strings in stringlist until they have the same length
    # get maximum length of strings in list
    maxlen = 0
    for s in stringlist: 
        if len(s)>maxlen: maxlen=len(s)
    maxlen += 2
    # replace each string by string of fixed length
    for i,s in enumerate(stringlist):
        stringlist[i] = str('{:<'+str(maxlen)+'}').format(s)

def writedatacard( datacarddir, channelname, processinfo,
		   histfile, variable, dummydata=False,
		   shapesyslist=[], smoothsyslist=[], lnnsyslist=[],
		   rateparamlist=[], ratio=[],
		   automcstats=10,
		   writeobs=False,
           writeselectedhists=True,
           writeallhists=False,
           autof=False ):
    ### write a datacard corresponding to a single channel ('bin' in combine terminology)
    # input arguments:
    # - datacarddir: directory where the card and histograms will go
    # - channelname: name of the channel/bin
    # - processinfo: object of type ProcessInfoCollection
    # - histfile: path to root file containing the histograms
    #   note: the file will be copied to the datacard directory
    # - variable: name of the variable for which the histograms will be used
    # - dummydata: use some random histogram for data rather than the one from datatag
    #   (useful when no data is available, but still need correct datacard syntax)
    # - shapesyslist: list of shape systematics to consider (default: none)
    #	(must be a subset of the ones included in processinfo)
    # - smoothsyslist: list of shape systematics to smooth
    #   (must be a subset of shapesyslist)
    # - lnnsyslist: list of normalization systematics (type lnN) to consider (default: none)
    #   (must be a subset of the ones included in processinfo)
    # - rateparamlist: a list of process names for which to add a rate param
    #   note: maybe extend functionality later, for now just adding a line
    #         rate_<process> rateParam * <process> 1.0 [0.0,3.0]
    # - ratio: list of 2 process names, a rate param will be added linking both
    #   (so the signal strength for the first of both corresponds to the ratio)
    #   note: the first element in ratio must be the (unique) signal process
    #   note: can also contain only 1 element with a wildcard in it,
    #         in this case the rateParam will be added if there is a matching process,
    #         else ignored
    # - automcstats: threshold for combine's autoMCStats functionality
    # - writeobs: whether to write for each process the observed yield
    #   (if False, simply write -1 to extract the yield at runtime,
    #    can be useful but not very clean when recycling datacards for different variables
    #    with slightly different yields (depending on the binning) 
    #    without remaking the processinfo)
    # - writeselectedhists: whether to copy the required histograms to a new root file
    #   and make the datacard refer to that new file.
    #   (update: selected histograms are also clipped,
    #    for in case this was not done in the input file.)
    # - writeallhists: whether to copy the entire histfile 
    #   (which might contain more histograms than needed for the datacard / processinfo).
    #   (warning: file is copied as is, histograms are not clipped.)
    # - autof: boolean whether to overwrite existing card without explicitly asking

    # make path to datacard
    datacardname = 'datacard_'+channelname+'.txt'
    datacardpath = os.path.join(datacarddir,datacardname) 
    # check if datacard file already exists
    if( os.path.exists(datacardpath) and not autof ):
        print('WARNING in writedatacard: requested file already exists. Overwrite? (y/n)')
        go = raw_input()
        if not go=='y': return
    # check the datacard directory and make if needed
    if not os.path.exists(datacarddir):
      os.makedirs(datacarddir)
    # check if provided processinfo has data
    hasdata = True
    datahistname = processinfo.datahistname
    if processinfo.datahistname is None:
      hasdata = False
      msg = 'WARNING in datacardtools.py / writedatacard:'
      msg += ' provided ProcessInfoCollection does not contain data.'
      print(msg)
    if dummydata:
      hasdata = True
      dummyprocess = processinfo.plist[0]
      datahistname = processinfo.pinfos[dummyprocess].histname
      msg = 'WARNING in datacardtools.py / writedatacard:'
      msg += ' will use dummy data (for correct syntax without actual data).'
      print(msg)
    # copy root file to location if requested
    newhistname = histfile
    newhistpath = histfile
    if( writeselectedhists or writeallhists ): 
      newhistname = 'histograms_'+channelname+'.root'
      newhistpath = os.path.join(datacarddir,newhistname)
      if not os.path.exists(histfile):
        raise Exception('ERROR in writedatacard: input histogram file {} '
                        +'does not seem to exist'.format(histfile))
      if writeallhists: os.system('cp '+histfile+' '+newhistpath)
      else:
        histnames = processinfo.allhistnames()
        hists = ht.loadhistogramlist(histfile, histnames)
        # histogram processing: clipping
        ht.cliphistograms(hists)
        # histogram processing: smoothing
        for hist in hists:
            smooth = False
            systematic_tag = None
            for systematic in smoothsyslist:
                if systematic in hist.GetName():
                    smooth = True
                    systematic_tag = systematic
            if smooth:
                print('Smoothing histogram {}'.format(hist.GetName()))
                requestnominal = hist.GetName().split(systematic_tag)[0]+'nominal'
                nominal = ht.findhistogram(hists, requestnominal)
                ht.smoothvariation(hist, nominal)
        f = ROOT.TFile.Open(newhistpath,'recreate')
        for hist in hists: hist.Write()
        f.Close()
    # open (recreate) datacard file
    datacard = open(datacardpath,'w')
    # write nchannels, nprocesses and nparameters
    datacard.write('imax\t1'+'\n') # only 1 channel, combine later
    datacard.write('jmax\t'+str(processinfo.nprocesses()-1)+'\n')
    datacard.write('kmax\t'+'*\n')
    datacard.write(getseparator())
    # write file info
    for p in processinfo.plist:
        nominal_histname = processinfo.pinfos[p].histname
        sys_histname = nominal_histname.replace('_nominal','_$SYSTEMATIC')
        datacard.write('shapes '+p+' '+channelname+' '+newhistname
                        +' '+nominal_histname
                        +' '+sys_histname+'\n')
    if hasdata: datacard.write('shapes data_obs '+channelname+' '+newhistname
                    +' '+datahistname+'\n')
    datacard.write(getseparator())
    # write bin info
    datacard.write('bin\t\t'+channelname+'\n')
    datacard.write('observation\t-1\n')
    datacard.write(getseparator())
    # make first and second column
    c1 = ['bin','process','process','rate']
    c2 = ['','','','']
    for systematic in shapesyslist:
        c1.append(systematic)
        c2.append('shape')
    for systematic in lnnsyslist:
        c1.append(systematic)
        c2.append('lnN')
    # make rest of the columns
    columns = [c1,c2]
    for p in processinfo.plist:
        pyield = processinfo.pinfos[p].pyield
        if not writeobs: pyield = -1
        sysdict = {}
        for key in processinfo.pinfos[p].systematics.keys():
            sysdict[key] = processinfo.pinfos[p].get_datacard_impact(key)
        pcolumn = makecolumn(channelname,p,processinfo.pinfos[p].pid,
                    pyield,sysdict,
                    shapesyslist+lnnsyslist)
        columns.append(pcolumn)
    # format the columns
    for c in columns: 
        makealigned(c)
    # write all info row by row
    nrows = len(columns[0])
    for row in range(nrows):
        for col in range(len(columns)):
            datacard.write(columns[col][row]+' ')
        datacard.write('\n')
        if(row==3): datacard.write(getseparator())
        if(row==3+len(shapesyslist)): datacard.write(getseparator())
    datacard.write(getseparator())
    # add rate parameters
    if len(rateparamlist)>0:
        for p in rateparamlist:
            if p not in processinfo.plist:
                raise Exception('ERROR in writedatacard: rateParam requested for '
                                +'{}, but not in list of processes'.format(p))
            datacard.write( makerateparamrow(p)+'\n' )
        datacard.write(getseparator())
    # add rate parameters for ratio measurement
    if len(ratio)>0:
        if len(ratio)==1:
            if '*' not in ratio[0]:
                raise Exception('ERROR in writedatacard: list of ratio measurement has length 1,'
                                ' which is only supported if the process name contains a wildcard.')
            hasmatch = False
            for p in processinfo.plist:
                if re.match(ratio[0].replace('*','.*'),p): hasmatch = True
            if hasmatch:
                datacard.write( 'ratio_scale rateParam * '+ratio[0]+' 1.0\n' )
                datacard.write(getseparator())
        elif len(ratio)==2:
            if not ratio[0] in processinfo.plist:
                raise Exception('ERROR in writedatacard: process '+ratio[0]+' in list for ratio'
                            +' but not in list of recognized processes')
            if not ratio[1] in processinfo.plist:
                raise Exception('ERROR in writedatacard: process '+ratio[1]+' in list for ratio'
                            +' but not in list of recognized processes')
            if not processinfo.pinfos[ratio[0]].pid <= 0:
                raise Exception('ERROR in writedatacard: process '+ratio[0]+' is numerator for ratio'
                            +' but not defined as signal')
            datacard.write( 'ratio_scale rateParam * '+ratio[0]+' 1.0\n' )
            datacard.write( 'ratio_scale rateParam * '+ratio[1]+' 1.0\n' )
            datacard.write(getseparator())
        else:
            raise Exception('ERROR in writedatacard: list for ratio measurement has unexpected '
                            +' length: {}'.format(len(ratio)))
            
    # manage statistical uncertainties
    datacard.write(channelname+' autoMCStats '+str(automcstats))
    # close datacard
    datacard.close()

    # return the written files
    return (datacardpath, newhistpath)
