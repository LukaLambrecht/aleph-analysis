#################################################################
# Data structure for reading and managing analysis output files #
#################################################################

import sys
import os
import ROOT

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

import tools.roothisttools as ht


class ProcessInfo(object):
  ### dict-like data structure storing information for a single process
  
  def __init__( self, name,
                pid=None, pyield=None,
                histname=None, systematics={} ):
    ### initializer
    # input arguments:
    # - name: string representing the name of this process
    # - pid: integer representing the id number for combine
    # - pyield: float representing the total yield for this process
    # - histname: name of the nominal histogram in a root file
    # - systematics: dict mapping systematics names to impacts;
    #   the keys of this dict can have the following types:
    #     - str '-' (if the systematic is not applicable)
    #     - float (for flat systematics)
    #     - tuple of two names of TH1 in a root file (up,down).
    # note: this class does not contain the actual TH1 objects.
    #       only the info needed to retrieve them from a root file.
    self.name = name
    self.pid = pid
    self.pyield = pyield
    self.histname = histname
    self.systematics = systematics
    for key,val in systematics.items():
      if not self.check_systematic_val( val ):
        msg = 'ERROR in ProcessInfo.init:'
        msg += ' systematics argument contains unrecognized value:'
        msg += ' key "{}" has a value "{}" of type {}.'.format(key,val,type(val))
        raise Exception(msg)
    
  def allhistnames( self ):
    ### return all histogram names for this process
    histnames = [self.histname]
    for val in self.systematics.values():
      if( isinstance(val,tuple) ):
        histnames.append(val[0])
        histnames.append(val[1])
    return histnames

  def check_systematic_val( self, val ):
    ### internal helper function for checking validity of systematics argument
    if( isinstance(val,str) and val=='-' ): return True
    if( isinstance(val,float) ): return True
    if( isinstance(val,tuple) and len(val)==2
        and isinstance(val[0],str)
        and isinstance(val[1],str) ): return True
    return False

  def hassys( self, sysname ):
    return (sysname in self.systematics.keys())

  def addsys( self, sysname, mag ):
    ### add a systematic to this process
    # for allowed values of mag, see initializer
    # note: use this function only for adding a new systematic;
    #       to update an already present systematic, use enablesys
    if self.hassys(sysname):
      msg = 'ERROR in ProcessInfo.addsys:'
      msg += ' systematic "{}" already exists'.format(sysname)
      msg += ' for process "{}".'.format(self.name)
      raise Exception(msg)
    if not self.check_systematic_val( mag ): raise Exception('')
    self.systematics[sysname] = mag

  def removesys( self, sysname ):
    ### remove a systematic for this process
    if not self.hassys(sysname):
      msg = 'ERROR in ProcessInfo.removesys:'
      msg += ' systematic "{}" does not exist'.format(sysname)
      msg += ' for process "{}".'.format(self.name)
      raise Exception(msg)
    self.systematics.pop(sysname)

  def enablesys( self, sysname, mag ):
    ### enable a systematic for this process
    # for allowed valus of mag, see initializer
    # note: use this function only for modifying an existing systematic;
    #       to add a new systematic, use addsys
    if not self.hassys(sysname):
      msg = 'ERROR in ProcessInfo.enablesys:'
      msg += ' systematic "{}" not found for process "{}"'.format(sysname,self.name)
      raise Exception(msg)
    if not self.check_systematic_val( mag ): raise Exception('')
    self.systematics[sysname] = mag

  def disablesys( self, sysname ):
    ### disable a given systematic for a given set of processes
    self.enablesys( sysname, '-' )

  def considersys( self, sysname ):
    ### return whether a given systematic should be considered for this process
    if not self.hassys(sysname):
      msg = 'ERROR in ProcessInfo.considersys:'
      msg += ' systematic "{}" not found for process "{}"'.format(sysname,self.name)
      raise Exception(msg)
    val = self.systematics[sysname]
    if( isinstance(val,str) and val=='-' ): return False
    return True

  def get_datacard_impact( self, sysname ):
    ### return the entry for the datacard for a given systematic
    if not self.hassys(sysname):
      msg = 'ERROR in ProcessInfo.get_datacard_impact:'
      msg += ' systematic "{}" not found for process "{}"'.format(sysname,self.name)
      raise Exception(msg)
    val = self.systematics[sysname]
    if( isinstance(val,tuple) ): val = 1.
    return val

  def check_systematics( self, systematics ):
    ### internal helper function to check list of systematics
    for s in systematics:
      if not s in self.systematics.keys():
        raise Exception('ERROR in ProcessInfo.check_systematics:'
          +' systematic {} not found.'.format(s))

  def changename( self, newname ):
    ### change name of this ProcessInfo
    # note: names of systematics that may contain the process name,
    #     as well as hname, are unaffected!
    self.name = newname

  def __str__( self ):
    ### get printable string for this ProcessInfo
    res = 'ProcessInfo:\n'
    res += '  process: {}, pid: {}, yield: {}\n'.format(self.name, self.pid, self.pyield)
    res += '  nominal histogram: {}\n'.format(self.histname)
    res += '  systematics\n'
    for s,v in self.systematics.items(): res += '  {}: {}'.format(s,v)
    return res


class Process(object):
  ### extension of ProcessInfo containing the actual histograms
  
  def __init__( self, info, rootfile, doclip=False ):
    ### initializer
    # input arguments:
    # - info: an instance of type ProcessInfo
    # - rootfile: the path to a root file containing the required histograms
    self.info = info
    self.hist = None
    self.systhists = {}
    # open file
    f = ROOT.TFile.Open(rootfile,'read')
    keylist = [k.GetName() for k in f.GetListOfKeys()]
    # read nominal histogram
    if not self.info.histname in keylist:
      msg = 'ERROR in Process.init:'
      msg += ' nominal histogram "{}" not found in file "{}"'.format(
             self.info.histname, rootfile)
      raise Exception(msg)
    self.hist = f.Get(self.info.histname)
    self.hist.SetDirectory(0)
    # clip histogram if requested
    if doclip: ht.cliphistogram(self.hist)
    # set name and title of nominal histogram
    # (can diverge from the key name if only the key was changed for speed!)
    self.hist.SetName( self.info.histname )
    self.hist.SetTitle( self.info.name )
    # read systematic histograms
    for systematic,val in self.info.systematics.items():
      if( isinstance(val,str) and val=='-' ):
        # use nominal
        uphist = self.hist.Clone()
        uphist.SetDirectory(0)
        uphist.SetName(self.info.histname.replace('nominal',systematic+'Up'))
        uphist.SetTitle(self.info.name)
        downhist = self.hist.Clone()
        downhist.SetDirectory(0)
        downhist.SetName(self.info.histname.replace('nominal',systematic+'Down'))
        downhist.SetTitle(self.info.name)
        self.systhists[systematic] = (uphist,downhist)
      elif( isinstance(val,float) ):
        # use nominal scaled by a factor
        uphist = self.hist.Clone()
        uphist.SetDirectory(0)
        uphist.SetName(self.info.histname.replace('nominal',systematic+'Up'))
        uphist.SetTitle(self.info.name)
        uphist.Scale(val)
        downhist = self.hist.Clone()
        downhist.SetDirectory(0)
        downhist.SetName(self.info.histname.replace('nominal',systematic+'Down'))
        downhist.SetTitle(self.info.name)
        downhist.Scale(2-val)
        self.systhists[systematic] = (uphist,downhist)
      elif( isinstance(val,tuple) ):
        # read up-histogram
        if not val[0] in keylist:
          msg = 'ERROR in Process.init:'
          msg += ' histogram "{}" not found in file "{}"'.format(
                 val[0], rootfile)
          raise Exception(msg)
        uphist = f.Get(val[0])
        uphist.SetDirectory(0)
        if doclip: ht.cliphistogram(uphist)
        uphist.SetName(val[0])
        uphist.SetTitle(self.info.name)
        # read down-histogram
        if not val[1] in keylist:
          msg = 'ERROR in Process.init:'
          msg += ' histogram "{}" not found in file "{}"'.format(
                 val[1], rootfile)
          raise Exception(msg)
        downhist = f.Get(val[1])
        downhist.SetDirectory(0)
        if doclip: ht.cliphistogram(downhist)
        downhist.SetName(val[0])
        downhist.SetTitle(self.info.name)
        self.systhists[systematic] = (uphist,downhist)
    f.Close()

  def get_nominal( self ):
    ### get nominal histogram
    return self.hist

  def get_allhists( self ):
    ### get a list of all histograms
    histlist = [self.hist]
    for s in self.systhists.keys():
      histlist.append(self.systhists[s][0])
      histlist.append(self.systhists[s][1])
    return histlist

  def get_yield( self, systematic=None ):
    ### get the yield
    if systematic is None:
      return self.get_nominal().Integral()
    else:
      raise Exception('Process.get_yield for non-nominal histograms not yet implemented.')

  def get_systematic( self, systematic, idx, diff=False, absolute=False ):
    ### internal helper function
    # arguments:
    # - diff: subtract nominal from varied histogram
    # - absolute: take absolute value (only relevant if diff is True)
    if systematic not in self.systhists.keys():
      msg = 'ERROR in Process.get_systematic:'
      msg += ' systematic "{}" not in list of systematics.'.format(
             systematic)
      raise Exception(msg)
    hist = self.systhists[systematic][idx]
    if diff:
      hist = hist.Clone()
      hist.Add(self.hist,-1)
      if absolute: ht.absolute( hist )
    return hist

  def get_systematic_up( self, systematic ):
    ### get up-variation for a systematic
    return self.get_systematic( systematic, 0, diff=False )

  def get_systematic_down( self, systematic ):
    ### get down-variation for a systematic
    return self.get_systematic( systematic, 1, diff=False )

  def get_difference_up( self, systematic, absolute=False ):
    ### get up-variation for a systematic, nominal subtracted
    return self.get_systematic( systematic, 0, diff=True, absolute=absolute )
  
  def get_difference_down( self, systematic, absolute=False ):
    ### get down-variation for a systematic, nominal subtracted
    return self.get_systematic( systematic, 1, diff=True, absolute=absolute )

  def get_systematics_rss( self, systematics='all' ):
    ### get root-sum-square of relative systematics
    # arguments:
    # - systematics: list of systematics to include.
    #   use 'all' to include all systematics in the current Process.

    # default case of no systematics
    if( isinstance(systematics, list) and len(systematics)==0 ):
      hist = self.get_nominal().Clone()
      hist.Reset()
      return hist 
    if( isinstance(systematics,str) and systematics=='all' ): 
      systematics = self.systhists.keys()
    self.info.check_systematics(systematics)
    maxhistlist = []
    # loop over systematics
    for systematic in systematics:
      # find bin-per-bin maximum absolute variation w.r.t nominal
      uphist = self.get_systematic_up( systematic )
      downhist = self.get_systematic_down( systematic )
      maxhist = ht.binperbinmaxvar( [uphist,downhist], self.hist )
      maxhistlist.append(maxhist)
    # add resulting histograms in quadrature
    syshist = ht.rootsumsquare(maxhistlist)
    return syshist


class ProcessInfoCollection(object):
  ### collection of ProcessInfos with addtional info common to all of them

  def __init__( self ):
    ### empty initializer
    self.pinfos = {}
    self.plist = []
    self.slist = []
    self.minpid = None
    self.maxpid = None
    self.datahistname = None

  def nprocesses( self ):
    ### get current number of processes
    return len(self.plist)

  def allhistnames( self ):
    ### return all histogram names stored in this collection
    histnames = []
    for p in self.plist:
      for hname in self.pinfos[p].allhistnames(): 
        histnames.append(hname)
    if self.datahistname is not None:
      histnames.append(self.datahistname)
    return histnames

  def hassys( self, systematic ):
    ### check if a systematic is present in this collection
    return (systematic in self.slist)

  def check_processes( self, processes ):
    ### internal helper function to check list of processes
    for p in processes:
      if not p in self.plist:
        raise Exception('ERROR in ProcessInfoCollection.check_processes:'
          +' process {} not found.'.format(p))

  def check_systematics( self, systematics ):
    ### internal helper function to check list of systematics
    for s in systematics:
      if not s in self.slist:
        raise Exception('ERROR in ProcessInfoCollection.check_systematics:'
          +' systematic {} not found.'.format(s))

  def addprocess( self, processinfo ):
    ### add a process
    # note that the slist and already present processes will be updated
    # so each ProcessInfo in the collection contains the same systematics
    # input arguments:
    # - processinfo: object of type ProcessInfo
    if not isinstance(processinfo,ProcessInfo):
      raise Exception('ERROR in ProcessInfoCollection.addprocess:'
        +' type {} for argument is invalid.'.format(type(processinfo)))
    if processinfo.name in self.plist:
      raise Exception('ERROR in ProcessInfoCollection.addprocess:'
        +' a process with name {} already exists.'.format(processinfo.name))
    for p in self.plist:
      if self.pinfos[p].pid == processinfo.pid:
        raise Exception('ERROR in ProcessInfoCollection.addprocess:'
          +' a process with pid {} already exists.'.format(processinfo.pid))
    # add the systematics of this new process to all existing processes
    for sys in processinfo.systematics.keys():
      if not sys in self.slist: 
        self.slist.append(sys)
        for p in self.pinfos.values(): p.addsys( sys, '-' )
    # add the existing systematics to the new process
    for sys in self.slist:
      if not sys in processinfo.systematics.keys():
        processinfo.addsys( sys, '-' )
    # add the new process
    self.pinfos[processinfo.name] = processinfo
    self.plist.append(processinfo.name)
    if( self.maxpid is None or processinfo.pid > self.maxpid ): self.maxpid = processinfo.pid
    if( self.minpid is None or processinfo.pid < self.minpid ): self.minpid = processinfo.pid
    # re-sort lists
    self.sort()

  def adddata( self, datahistname ):
    ### set the data histogram name
    self.datahistname = datahistname

  def removeprocess( self, process ):
    ### remove a process
    # input arguments:
    # - process: name of the process to remove
    if process not in self.plist:
      raise Exception('ERROR in ProcessInfoCollection.removeprocess:'
        +' a process with name {} does not exists'.format(process))
    # remove the process
    self.plist.remove(process)
    self.pinfos.pop(process)
    # remove unused systematics
    for sys in self.slist:
      count = 0
      for p in self.plist:
        if self.pinfos[p].systematics[sys] != '-': count += 1
      if count==0:
        self.slist.remove(sys)
        for p in self.pinfos.values(): p.removesys(sys)
    # update minpid and maxpid
    self.maxpid = max([p.pid for p in self.pinfos.values()])
    self.minpid = min([p.pid for p in self.pinfos.values()])
    # re-sort lists
    self.sort()

  def removesystematic( self, systematic ):
    ### remove a systematic
    # input arguments:
    # - process: name of the systematic to remove
    if systematic not in self.slist:
      raise Exception('ERROR in ProcessInfoCollection.removesystematic:'
        +' a systematic with name {} does not exists'.format(systematic))
    # remove the systematic from all processes
    for p in self.pinfos.values(): p.removesys(systematic)
    self.slist.remove(systematic)
    # re-sort lists
    self.sort()

  def sort( self ):
    ### internal helper function to (re-)sort internal lists.
    # lists that will be sorted: 
    # - systematic list (alphabetically)
    # - process list (by ID)
    self.slist = sorted(self.slist)
    newplist = []
    for i in range(self.minpid,self.maxpid+1):
      for p in self.plist:
        if self.pinfos[p].pid==i: newplist.append(p)
    self.plist = newplist

  def addnormsys( self, sysname, impacts ):
    ### add a normalization uncertainty
    # input arguments:
    # - sysname: name of systematic uncertainty
    # - impacts: dict mapping process names to impacts (either float or equivalent string, or '-')
    # note: the keys of impacts must correspond exactly to this ProcessInfoCollection, 
    #       for safety of accidentally using wrong process names and hence lacking uncertainties
    if sorted(self.plist)!=sorted(impacts.keys()):
      raise Exception('ERROR in ProcessInfoCollection.addnormsys:'
        +' processes in info struct and impacts do not agree;'
        +' found\n{}\nand\n{}'.format(sorted(self.plist),sorted(impacts.keys())))
    if sysname in self.slist:
      raise Exception('ERROR in ProcessInfoCollection.addnormsys:'
        +' a systematic with name {} already exists.'.format(sysname))
    self.slist.append( sysname )
    for p in self.plist:
      self.pinfos[p].addsys( sysname, impacts[p] )

  def enablesys( self, sysname, processes, mag ):
    ### enable a given systematic for a given set of processes with given magnitude
    if sysname not in self.slist:
        raise Exception('ERROR in ProcessInfoCollection.enablesys:'
          +' systematic {} not found.'.format(sysname))
    for p in processes:
      if p not in self.plist:
        raise Exception('ERROR in ProcessInfoCollection.enablesys:'
          +' process {} not found.'.format(p))
      self.pinfos[p].enablesys( sysname, mag )

  def disablesys( self, sysname, processes ):
    ### disable a given systematic for a given set of processes
    self.enablesys( sysname, processes, '-' )

  def changename( self, oldpname, newpname ):
    ### change the name of a process from oldpname to newpname
    # note: all systematic uncertainties associated to this process,
    # as well as the histogram names are unaffected!
    if not oldpname in self.plist:
      raise Exception('ERROR in ProcessInfoCollection.changename:'
        +' old name {} not found.'.format(oldpname))
    if newpname in self.plist:
      raise Exception('ERROR in ProcessInfoCollection.changename:'
        +' new name {} already exists.'.format(newpname))
    self.pinfos[newpname] = self.pinfos.pop(oldpname)
    self.plist.append( newpname )
    self.plist.remove( oldpname )
    self.sort()

  def changebkgsig( self, pname, totype ):
    ### internal helper function for makebkg and makesig
    # arguments;
    # - totype must be either 'bkg' (from signal to background)
    #   or 'sig' (from background to signal)
    if not totype in ['bkg','sig']:
      raise Exception('ERROR in ProcessInfoCollection.changebkgsig:'
        +' totype {} not recognized.'.format(totype))
    if not pname in self.plist:
      raise Exception('ERROR in ProcessInfoCollection.changebkgsig:'
        +' process {} not in process info'.format(pname))
    if( totype=='bkg' and self.pinfos[pname].pid>0 ):
      print('WARNING in ProcessInfoCollection.changebkgsig:'
        +' process {} is already a background process'.format(pname))
      return
    if( totype=='sig' and self.pinfos[pname].pid<0 ):
      print('WARNING in ProcessInfoCollection.changebkgsig:'
        +' process {} is already a signal process'.format(pname))
      return
    # case to background
    if totype=='bkg':
      # redefine ids of unmodified processes
      for p in self.plist:
        # add 1 to all backgrounds
        if self.pinfos[p].pid>0: self.pinfos[p].pid += 1
        # add 1 to all signals with pid < original pid of given process
        if self.pinfos[p].pid<self.pinfos[pname].pid: self.pinfos[p].pid += 1
      # redefine given process
      self.pinfos[pname].pid = 1
      self.maxpid += 1
      self.minpid += 1
    # case to signal
    if totype=='sig':
      # redefine ids of unmodified processes
      for p in self.plist:
        # subtract 1 from all signals
        if self.pinfos[p].pid<=0: self.pinfos[p].pid -= 1
        # subtract 1 from all backgrounds with pid > original pid of given process
        if self.pinfos[p].pid>self.pinfos[pname].pid: self.pinfos[p].pid -= 1
      # redefine given process
      self.pinfos[pname].pid = 0
      self.maxpid -= 1
      self.minpid -= 1
    # resort
    self.sort()

  def makebkg( self, pname ):
    ### turn a given process from signal (id<=0) to background (id>0)
    self.changebkgsig( pname, totype='bkg' )

  def makesig( self, pname ):
    ### turn a given process from background (id>0) to signal (id<=0)
    self.changebkgsig( pname, totype='sig' )

  def __str__( self ):
    ### get a printable string of this collection
    pstring = 'processes'
    sstring = 'systematics'
    if len(self.plist)==1: pstring = 'process'
    if len(self.slist)==1: sstring = 'systematic'
    res = 'ProcessInfoCollection with {} {} and {} {}\n'.format(
          len(self.plist), pstring, len(self.slist), sstring)
    for n,p in self.pinfos.items():
      res += '  process: {}, pid: {}, yield: {}\n'.format(n,p.pid,p.pyield)
      for s in self.slist:
        res += '    {}: {}\n'.format(s,p.systematics[s])
    if self.datahistname is not None:
      res += '  data: {}\n'.format(self.datahistname)
    res = res.strip('\n')
    return res

  @staticmethod
  def fromhistlist( histnames, variable, signals=[],
                    includesystematics=None, excludesystematics=None,
                    datatag='data', adddata=False, nominaltag='_nominal' ):
    ### make a ProcessInfoCollection from a list of histogram names
    # note: this concerns a definition of processes and shape systematics;
    # to add normalization uncertainties (not stored as root histograms), 
    # use ProcessInfoCollection.addnormsys.
    # note: this function does not read all histograms, only the names (for speed).
    # input arguments:
    # - histfile: path to a root file containing all histograms
    #   the histograms are assumed to be named process_variable_systematic
    #   (with 'nominal' as systematic for the nominal histogram).
    #   all tags, i.e. process, variable and systematic, 
    #   are in principle allowed to contain underscores.
    # - variable is the name of the variable for which to extract the histograms
    # - signals is a list of process names that identify signal processes (opposed to background).
    #   signals are given an 'index' <= 0 to make combine define them as signal.
    # - includesystematics: list of systematics to include (default: all in file)
    # - excludesystematics: list of systematics to exclude (default: none)
    # - datatag is the process name of data histograms
    # - adddata: whether to add the data to the ProcessInfoCollection
    # - nominaltag: tag by which to recognize nominal histograms
    # output object: a ProcessInfoCollection object

    # initialization
    pinfo = {} # final output dict containing info for all processes
    plist = [] # list of process names
    slist = [] # list of systematics
    bkgcounter = 1 # id counter for backgrounds
    sigcounter = 0 # id counter for signals
    # select only histograms of the requested variable
    # and do not consider data
    selhistnames = ([el for el in histnames
                     if ('_'+variable+'_' in el
                         and datatag != el.split(variable)[0].rstrip('_'))])
    # make list of processes
    plist = [el.split(variable)[0].rstrip('_') for el in selhistnames]
    plist = list(set(plist))
    # loop over processes
    for process in plist:
      # determine whether process is signal or background
      if process in signals:
        idnumber = sigcounter
        sigcounter -= 1
      else:
        idnumber = bkgcounter
        bkgcounter += 1
      # subselect all histograms for this process
      thishistnames = ([el for el in selhistnames
                        if el.split(variable)[0].rstrip('_')==process])
      # find nominal histogram
      nomhistname = '{}_{}{}'.format(process, variable, nominaltag)
      if not nomhistname in thishistnames:
        raise Exception('ERROR in ProcessInfoCollection.fromhistlist:'
          +' nominal histogram {} not found for process {}'.format(nomhistname,process))
      thishistnames.remove(nomhistname)
      # read nominal histogram and determine yield
      # to implement or to skip...
      # make the ProcessInfo
      pinfo[process] = ProcessInfo( process, pid=idnumber, pyield=0.,
                                    histname=nomhistname, systematics={} )
      # loop over all other histograms for this process
      for histname in thishistnames:
        # determine what systematic the current histogram belongs to
        systematic = histname.split(variable)[-1].strip('_')
        if(systematic[-2:]=='Up'): systematic = systematic[:-2]
        elif(systematic[-4:]=='Down'): continue
        else: continue
        # (consider only up as only the name of the systematic is needed)
        # check whether to consider this systematic
        if( includesystematics is not None and (systematic not in includesystematics) ): continue
        if( excludesystematics is not None and (systematic in excludesystematics) ): continue
        # check if down variation is also present
        downhistname = '{}_{}_{}Down'.format(process, variable, systematic)
        if not downhistname in thishistnames:
          raise Exception('ERROR in ProcessInfoCollection.fromhistlist:'
            +' down histogram {} not found'.format(downhistname)
            +' (corresponding to up histogram {}).'.format(histname))
        # set systematic impacts
        if not systematic in slist: slist.append(systematic)
        pinfo[process].addsys(systematic,(histname,downhistname))
    # add all processes to a collection
    if len(plist)==0:
      print('WARNING in ProcessInfoCollection.fromhistlist:'
            +' returning an empty ProcessInfoCollection;'
            +' check if the file contains the right histograms'
            +' and if they are read correctly.')
    if len(slist)==0:
      print('WARNING in ProcessInfoCollection.fromhistlist:'
            +' returning a ProcessInfoCollection with no systematics;'
            +' check if the file contains the right histograms'
            +' and if they are read correctly.')
    PIC = ProcessInfoCollection()
    for p in pinfo.values(): PIC.addprocess(p)
    # add the data histogram if requested
    if adddata:
      datahistname = ([el for el in histnames
                        if (variable in el
                        and datatag in el.split(variable)[0].rstrip('_'))])
      if len(datahistname)!=1:
        msg = 'ERROR in ProcessInfoCollection.fromhistlist:'
        msg += ' expected one data histogram but found {}:\n'.format(len(datahistname))
        for dhname in datahistname: msg += '  - {}\n'.format(dhname)
        msg = msg.strip('\n')
        raise Exception(msg)
      PIC.adddata( datahistname[0] )
    return PIC

  @staticmethod
  def fromhistfile( histfile, variable, **kwargs):
    ### read a ROOT file containing histograms and make a ProcessInfoCollection
    # see fromhistlist for more info

    # get all histogram names
    f = ROOT.TFile.Open(histfile)
    keylist = f.GetListOfKeys()
    histnames = [k.GetName() for k in keylist]
    f.Close()
    return parsehistlist(histnames, variable, **kwargs )

  @staticmethod
  def fromdatacard( datacard, adddata=False ):
    ### convert a datacard to a ProcessInfoCollection
    # note: for now only works on elementary datacards,
    #       i.e. corresponding to a single plot.
    with open(datacard, 'r') as f:
        lines = [line.strip(' \t\n') for line in f.readlines()]
    # group the lines in blocks divided by separators
    # (note: depends on conventional datacard creation with separators!)
    separator = '--------------------'
    blocks = []
    startidx = 0
    for i,line in enumerate(lines):
        if line==separator:
            block = [lines[j] for j in range(startidx,i)]
            blocks.append(block)
            startidx = i+1
    lastblock = [lines[j] for j in range(startidx,i+1)]
    blocks.append(lastblock)
    if(len(blocks)!=7 and len(blocks)!=8):
        # (standard 7 blocks,
        #  +1 in case of rate parameters)
        msg = 'ERROR: number of blocks is {}'.format(len(blocks))
        msg += ' while 7 or 8were expected.'
        msg += ' Check the datacard formatting.'
        raise Exception(msg)
    # get a list of processes from the 'process' line
    # (note: alternatively, could get it from the 'shapes' block,
    #        but not sure if the correct order is guaranteed there)
    pline = ''
    pidline = ''
    for i,line in enumerate(lines):
        if line.startswith('process'):
            pline = line
            pidline = lines[i+1]
            break
    processes = pline.split()[1:]
    pids = [int(el) for el in pidline.split()[1:]]
    # get histogram base names from the 'shapes' block
    hbase = {}
    for line in blocks[1]:
        elements = line.split()
        p = elements[1]
        if p=='data_obs': continue
        hbase[p] = elements[4].replace('nominal','')
    if sorted(hbase.keys())!=sorted(processes):
        msg = 'ERROR: could not determine histogram base name for all processes.'
        msg += ' Check the datacard formatting.'
        raise Exception(msg)
    # get systematics for each process
    psysdict = {}
    for p in processes: psysdict[p] = {}
    for line in blocks[4]+blocks[5]:
        elements = line.split()
        systematic = elements[0]
        stype = elements[1]
        impacts = elements[2:]
        if len(impacts)!=len(processes):
            msg = 'ERROR: number of columns does not agree'
            msg += ' for systematic {}.'.format(systematic)
        for i,p in enumerate(processes):
            if impacts[i]=='-':
                psysdict[p][systematic] = '-'
            elif( stype=='shape' and float(impacts[i])==1. ):
                upname = hbase[p]+systematic+'Up'
                downname = hbase[p]+systematic+'Down'
                psysdict[p][systematic] = (upname, downname)
            elif( stype=='lnN' ):
                psysdict[p][systematic] = float(impacts[i])
            else:
                msg = 'ERROR: could not interpret systematic {}'.format(systematic)
                msg += ' for process {}'.format(p)
                print(msg)
    # make a ProcessInfoCollection
    PIC = ProcessInfoCollection()
    for i,p in enumerate(processes):
        PI = ProcessInfo(p, pid=pids[i], histname=hbase[p]+'nominal', systematics=psysdict[p])
        PIC.addprocess(PI)
    # add data if requested
    if adddata:
        dhname = ''
        for line in blocks[1]:
            elements = line.split()
            p = elements[1]
            if p=='data_obs':
                dhname = elements[4]
                break
        if len(dhname)==0:
            raise Exception('ERROR: could not find data histogram.')
        PIC.adddata( dhname )
    return PIC


class ProcessCollection(object):
  ### collection of Process instances with common info and functions

  def __init__( self, info, rootfile, doclip=False ):
    ### initializer from a ProcessInfoCollection and a root file containing the histograms
    if not isinstance(info, ProcessInfoCollection):
      raise Exception('ERROR in ProcessCollection.init:'
        +' unrecognized type for info argument: {}'.format(type(info))
        +' (expected ProcessInfoCollection)')
    self.info = info
    self.plist = self.info.plist # (shortcut)
    self.slist = self.info.slist # (shortcut)
    self.processes = {}
    for pname,pinfo in self.info.pinfos.items():
      self.processes[pname] = Process( pinfo, rootfile, doclip=doclip )
    self.datahist = None
    if self.info.datahistname is not None:
      f = ROOT.TFile.Open(rootfile, 'read')
      self.datahist = f.Get(self.info.datahistname)
      self.datahist.SetDirectory(0)
      f.Close()

  # to extend this class according to arising needs,
  # e.g. sum of nominal processes, linear sum of systematics,
  #      quadratic sum of systematics, ...

  def get_allhists( self ):
    ### return a list of all histograms
    histlist = []
    for pname in self.plist:
      histlist += self.processes[pname].get_allhists()
    return histlist

  def get_hist_sum( self, histlist ):
    ### internal helper function
    if len(histlist)==0:
      raise Exception('ERROR in ProcessCollection.get_hist_sum:'
        +' received empty histogram list.')
    if len(histlist)==1: return histlist[0].Clone()
    sumhist = histlist[0].Clone()
    for hist in histlist[1:]: sumhist.Add(hist)
    return sumhist

  def get_nominal( self ):
    ### get the nominal histogram for the sum of all processes
    return self.get_hist_sum( [self.processes[p].hist for p in self.plist] )

  def get_yields( self, systematic=None ):
    ### return the yields (per process and total) in a dict
    res = {}
    if systematic is None:
      for pname, p in self.processes.items():
        res[pname] = p.get_yield()
      res['total'] = self.get_nominal().Integral()
      return res
    else:
      raise Exception('ERROR: ProcessCollection.get_yields for non-nominal histograms'
                     +' not yet implemented.')
    
  def get_systematic_up( self, systematic, processes='all' ):
    ### get up variation for a given systematic and process(es)
    # arguments:
    # - processes: list of processes to vary upwards
    #   (will use nominal for all other processes in the collection).
    #   use 'all' to vary all processes upwards.
    # note: the up variations for several processes are added linearly.
    histlist = []
    if( isinstance(processes,str) and processes=='all' ): processes = self.plist
    else: self.info.check_processes( processes )
    for p in self.plist:
      if p in processes: histlist.append( self.processes[p].get_systematic_up(systematic) )
      else: histlist.append( self.processes[p].get_nominal() )
    return self.get_hist_sum( histlist )

  def get_systematic_down( self, systematic, processes='all' ):
    ### get down variation for given systematic and process(es)
    histlist = []
    if( isinstance(processes,str) and processes=='all' ): processes = self.plist
    else: self.info.check_processes( processes )
    for p in self.plist:
      if p in processes: histlist.append( self.processes[p].get_systematic_down(systematic) )
      else: histlist.append( self.processes[p].get_nominal() )
    return self.get_hist_sum( histlist )

  def get_difference_up( self, systematic, processes='all' ):
    ### same as get_systematic_up but subtract nominal
    uphist = self.get_systematic_up( systematic, processes=processes )
    uphist.Add( self.get_nominal(), -1 )
    return uphist

  def get_difference_down( self, systematic, processes='all' ):
    ### same as get_systematic_down but subtract nominal
    downhist = self.get_systematic_down( systematic, processes=processes )
    downhist.Add( self.get_nominal(), -1 )
    return downhist

  def get_systematics_rss( self, systematics='all', processes='all',
                           correlate_processes=False ):
    ### get root-sum-square of relative systematics
    # arguments:
    # - systematics: list of systematics to include.
    #   use 'all' to include all systematics in the current Process.
    # - processes: list of processes to take into account.
    #   use 'all' to use all processes in the ProcessCollection.
    # - correlate_processes: if True, each systematic will be summed linearly over processes,
    #   and the resulting total variations are summed quadratically.
    #   if False, the quadratic sum is performed over both systematics and processes.
    if( isinstance(systematics,str) and systematics=='all' ): systematics = self.slist
    else: self.info.check_systematics( systematics )
    if( isinstance(processes,str) and processes=='all' ): processes = self.plist
    else: self.info.check_processes( processes )
    # special case: no systematics
    if( len(systematics)==0 ):
      hist = self.get_nominal().Clone()
      hist.Reset()
      return hist
    if not correlate_processes:
      # case of root sum square over both systematics and processes
      per_process_rss = []
      for p in processes: 
        this_process_rss = self.processes[p].get_systematics_rss(systematics=systematics)
        per_process_rss.append( this_process_rss )
      return ht.rootsumsquare( per_process_rss )
    else:
      # case of linear sum over processes, quadratic over systematics
      maxhistlist = []
      for s in systematics:
        uphist = self.get_systematic_up( s, processes=processes )
        downhist = self.get_systematic_down( s, processes=processes )
        nominalhist = self.get_nominal()
        maxhist = ht.binperbinmaxvar( [uphist,downhist], nominalhist )
        maxhistlist.append(maxhist)
      return ht.rootsumsquare( maxhistlist )
