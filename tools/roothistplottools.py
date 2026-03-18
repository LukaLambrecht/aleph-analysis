# Contains the main functionality for the 'typical' MC vs data plots


import os
import sys
import ROOT
import numpy as np
import json

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

import tools.rootplottools as pt
import tools.roothisttools as ht


# help functions

def orderhistograms(histlist, ascending=True, weightlist=None):
    # order a list of histograms
    # input arguments:
    # - ascending: set the order (ascending or descending)
    # - weightlist: list of same length as histlist, with property to use for sorting,
    #   if None, the sum of weights of each histogram is used.
    if weightlist is None:
        weightlist = []
        for hist in histlist:
            sumofweights = hist.GetSumOfWeights()
            weightlist.append(sumofweights)
    weightlist = np.array(weightlist)
    sorted_indices = np.argsort(weightlist)
    if ascending: return [histlist[i] for i in sorted_indices]
    else: return [histlist[i] for i in np.flip(sorted_indices)]

def findbytitle(histlist, title):
    # find a histogram by its title, return the index or -1 if not found
    index = -1
    for i,hist in enumerate(histlist):
        if hist.GetTitle()==title: index = i
    return index

def grouphistograms(histlist, groupdict):
    # group some histograms together
    # input arguments:
    # - groupdict: dictionary defining the grouping,
    #   formatted as follows: {"orig1": "groupedname", "orig2": "groupedname", ...}
    # note: this only affects the plotting of nominal histograms,
    #       not the uncertainties or the behaviour in the likelihood fit
    # note: histograms that are not in groupdict are kept unchanged.
    new_hists = {}
    #for hist in histlist: print(hist.GetTitle())
    for hist in histlist:
      title = hist.GetTitle()
      newtitle = title
      if title in groupdict.keys():
        newtitle = groupdict[title]
        hist.SetTitle(newtitle)
      if newtitle in new_hists.keys(): new_hists[newtitle].Add(hist.Clone())
      else: new_hists[newtitle] = hist.Clone()
    newhists = list(new_hists.values())
    #print('---')
    #for hist in newhists: print(hist.GetTitle())
    return newhists

def stackcol(hist,color):
    # set color and line properties of histogram in stack
    hist.SetFillColor(color)
    hist.SetLineWidth(1)
    hist.SetLineColor(ROOT.kBlack)

def getminmax(datahist, mchist, yaxlog=False, margin=True):
    # get suitable minimum and maximum values for plotting a given data hist and summed mc hist
    # find maximum:
    histmax = (mchist.GetBinContent(mchist.GetMaximumBin())
                +mchist.GetBinErrorUp(mchist.GetMaximumBin()))
    histmax = max(histmax,datahist.GetBinContent(datahist.GetMaximumBin())
                            +datahist.GetBinErrorUp(datahist.GetMaximumBin()))
    if not yaxlog:
        if margin: histmax *= 1.5
        return (0, histmax)
    # find minimum (manually to avoid zero and small dummy values)
    histmin = histmax
    for i in range(1,mchist.GetNbinsX()+1):
        if( (not mchist.GetBinContent(i)<1e-3) and mchist.GetBinContent(i)<histmin):
            histmin = mchist.GetBinContent(i)
    if margin:
      histmin = histmin/5.
      histmax = histmax*np.power(histmax/histmin,0.6)
    return (histmin, histmax)

def getminmaxfromlist(hists, yaxlog=False, margin=True):
    # same as getminmax but from a list of histograms instead of data and total sim.
    histmax = 0
    for hist in hists:
        this_hist_max = ( hist.GetBinContent(hist.GetMaximumBin())
                          + hist.GetBinErrorUp(hist.GetMaximumBin()) )
        if this_hist_max > histmax: histmax = this_hist_max
    if not yaxlog:
        if margin: histmax *= 1.5
        return (0, histmax)
    # find minimum (manually to avoid zero and small dummy values)
    histmin = histmax
    for hist in hists:
        for i in range(1, hist.GetNbinsX()+1):
            if( hist.GetBinContent(i)>1e-3 and hist.GetBinContent(i)<histmin):
                histmin = hist.GetBinContent(i)
    if margin:
      histmin = histmin/5.
      histmax = histmax*np.power(histmax/histmin,0.6)
    return (histmin, histmax)

def drawbincontent(mchistlist,mchisterror,tag):
    text = ROOT.TLatex()
    text.SetTextAlign(21)
    text.SetTextFont(5);
    text.SetTextSize(22);
    taghist = mchistlist[findbytitle(mchistlist,tag)]
    for i in range(1, taghist.GetNbinsX()+1):
        xcoord  = taghist.GetXaxis().GetBinCenter(i)
        ycoord  = mchisterror.GetBinContent(i)+mchisterror.GetBinErrorUp(i)
        printvalue = taghist.GetBinContent(i)
        text.DrawLatex(xcoord, ycoord+0.3, '{:.2f}'.format(printvalue))
    return None

def plotdatavsmc(outfile, datahist, mchistlist,
	mcsysthist=None, mcstathist=None, dostat=True,
    groupdict=None,
	signals=None, datalabel=None, labelmap=None, colormap=None,
	xaxtitle=None, xaxinteger=False,
	yaxtitle=None, yaxlog=False, yaxrange=None,
	p2yaxtitle=None, p2yaxrange=None,
	p1legendncols=None, p1legendbox=None,
	extracmstext='', lumi=None,
	extrainfos=[], infosize=None, infoleft=None, infotop=None,
    binlabels=None, labelsize=None, labelangle=None,
    canvaswidth=None, canvasheight=None):
    ### make a (stacked) simulation vs. data plot
    # arguments:
    # - outfile is the output file where the figure will be saved
    # - datahist is the data histogram
    # - mchistlist is a list of simulated histograms
    #   (note that the title of each histogram will be used to determine its label!)
    #   (note that the title of each histogram will be used to determine its color!)
    # - mcsyshist is a histogram containing the total systematic variation on the simulation
    #   (note: the bin content of mcsyshist must be the absolute difference wrt nominal!)
    #   (note: if dostat is False, mcsysthist must be the total uncertainty, not only systematic!
    #          if dostat is True, mcsysthist is syst only;
    #	       stat error will be taken from mcstathist if it is defined or from mchistlist otherwise!)
    # - mcstathist is a histogram containing the total statistical variation on the simulation
    #   (note: see above!)
    # - signals = list of signal processes (will be put at the top of the plot)
    # - datalabel is a string that will be used as legend entry for datahist (default: Data)
    # - labelmap is a dict mapping strings (histogram titles) to legend entries
    # - colormap is a dict mapping strings (histogram titles) to colors
    # - xaxtitle is the x-axis title (default: no title)
    # - xaxinteger is a boolean whether to use only integer axis labels and ticks for the x-axis
    # - yaxtitle: y-axis title (default: no title)
    # - yaxlog is whether to make the y-axis log scale
    # - yaxrange: tuple of (min,max) for plotting range (default: automatic range)
    # - p2yaxtitle:  y-axis title of bottom pad (default: datalabel/Pred.)
    # - p2yaxrange: tuple of (min,max) for plotting range of bottom pad (default: (0,2))
    # - lumi is the luminosity value (float, in pb-1) that will be displayed
    # - extrainfos is a list of strings with extra info to display
    # - infosize: font size of extra info
    # - infoleft: left border of extra info text (default leftmargin + 0.05)
    # - infotop: top border of extra info text (default 1 - p1topmargin - 0.1)
    # - binlabels: list of alphanumeric x-axis labels (for categorical histograms)
    #   extension: this argument toggles single vs double variable plotting;
    #   in the latter case, binlabels must be a tuple of two lists: 
    #   (primary variable labels, secondary variable labels)
    
    pt.setTDRstyle()
    ROOT.gROOT.SetBatch(ROOT.kTRUE)

    ### regroup mc histograms if requested
    # note: this needs to be done first in order for automatic labels and colors
    #       to work properly (see definition of labels and colors below)
    if groupdict is not None: mchistlist = grouphistograms(mchistlist, groupdict)

    ### parse arguments
    if datalabel is None: datalabel = "Data"
    # if no labelmap, use histogram titles
    if labelmap is None: 
        labelmap = {}
        for hist in mchistlist: labelmap[hist.GetTitle()] = hist.GetTitle()
    # if no colormap, use default colors
    if colormap is None:
        clist = ([ROOT.kRed-7,ROOT.kOrange,ROOT.kCyan-7,ROOT.kYellow+1,ROOT.kBlue-10,
		    ROOT.kBlue-6,ROOT.kTeal-5,ROOT.kMagenta-7])
        colormap = {}
        for i,hist in enumerate(mchistlist):
            colormap[hist.GetTitle()] = clist[ i%len(clist) ]
    # y-axis title and range for bottom pad
    if p2yaxtitle is None: p2yaxtitle = 'D/P'
    if p2yaxrange is None: p2yaxrange = (0.,1.999)
    
    ### define global parameters for size and positioning
    if canvasheight is None: canvasheight = 600
    if canvaswidth is None: canvaswidth= 600
    # fonts and sizes:
    labelfont = 4
    if labelsize is None: labelsize = 22
    axtitlefont = 4; axtitlesize = 27
    infofont = 4
    if infosize is None: infosize = 20
    # title offset
    ytitleoffset = 1.5
    # temporary modification for an annoying comment on one figure:
    xtitleoffset = 1
    # margins:
    p1topmargin = 0.07
    p2bottommargin = 0.4
    leftmargin = 0.15
    rightmargin = 0.05
    # legend box
    if p1legendbox is None: 
        p1legendbox = [leftmargin+0.35,1-p1topmargin-0.3,1-rightmargin-0.03,1-p1topmargin-0.03]
    if p1legendncols is None:
        p1legendncols = 2
    p2legendbox = [leftmargin+0.03,0.84,1-rightmargin-0.03,0.97]
    # extra info box parameters
    if infoleft is None: infoleft = leftmargin+0.05
    if infotop is None: infotop = 1-p1topmargin-0.1
    # marker properties for data
    markerstyle = 20
    markercolor = 1
    markersize = 0.75
    # variable
    varmode = 'single'
    nprimarybins = None
    secondarybinlabels = None
    if( binlabels is not None and isinstance(binlabels,tuple) ): 
        varmode = 'double'
        nprimarybins = len(binlabels[0])
        secondarybinlabels = binlabels[1]
        binlabels = binlabels[0]*len(secondarybinlabels)

    ### order mc histograms
    # order by sumofweights
    mchistlist = orderhistograms(mchistlist)
    # get signals to the back (i.e. on top of the plot)
    #if signals is not None:
    #    for signal in signals[::-1]:
    #        sindex = findbytitle(mchistlist,signal)
    #        if(sindex>-1):
    #            indices = list(range(len(mchistlist)))
    #            indices.remove(sindex)
    #            indices = indices+[sindex]
    #            mchistlist = [mchistlist[i] for i in indices]

    ### operations on mc histograms
    mchistsum = mchistlist[0].Clone()
    mchistsum.Reset()
    mchiststack = ROOT.THStack("mchiststack","")
    for hist in mchistlist:
        stackcol( hist, colormap.get(hist.GetTitle(), ROOT.kBlack) )
        ht.cliphistogram(hist)
        mchistsum.Add(hist)
        mchiststack.Add(hist)

    ### calculate total mc error and set its histogram properties
    mcerror = mchistsum.Clone()
    mcstaterror = mchistsum.Clone()
    # in case of dostat=False: mcsysthist represents total variation to plot
    if not dostat:
        if mcsysthist is None:
            raise Exception('ERROR in histplotter.py: option dostat = False'
		    +' is not compatible with mcsysthist = None')
        else:
            for i in range(1,mchistsum.GetNbinsX()+1):
                mcerror.SetBinError(i,mcsysthist.GetBinContent(i))
    # in case of dostat=True: add stat errors and syst errors quadratically
    else:
        if mcstathist is not None:
            # take statistical variations from externally provided histograms
            for i in range(1,mchistsum.GetNbinsX()+1):
                mcstaterror.SetBinError(i,mcstathist.GetBinContent(i))
                mcerror.SetBinError(i,mcstathist.GetBinContent(i))
        if mcsysthist is not None:
            for i in range(1,mchistsum.GetNbinsX()+1):
                staterror = mcstaterror.GetBinError(i)
                systerror = mcsysthist.GetBinContent(i)
                mcerror.SetBinError(i,np.sqrt(np.power(staterror,2)+np.power(systerror,2)))
    mcerror.SetLineWidth(0)
    mcerror.SetMarkerStyle(0)
    mcerror.SetFillStyle(3254)
    mcerror.SetFillColor(ROOT.kBlack)

    ### calculate total and statistical mc error (scaled)
    scstaterror = mcerror.Clone()
    scerror = mcerror.Clone()
    for i in range(1,mcerror.GetNbinsX()+1):
        scstaterror.SetBinContent(i,1.)
        scerror.SetBinContent(i,1.)
        if not mcerror.GetBinContent(i)==0:
            scstaterror.SetBinError(i,mcstaterror.GetBinError(i)/mcstaterror.GetBinContent(i))
            scerror.SetBinError(i,mcerror.GetBinError(i)/mcerror.GetBinContent(i))
        else:
            scstaterror.SetBinError(i,0.)
            scerror.SetBinError(i,0.)
    # in case of dostat=False: plot only total scaled variation, same style as unscaled
    if not dostat:
        scstaterror.Reset()
        scerror.SetLineWidth(0)
        scerror.SetMarkerStyle(0)
        scerror.SetFillStyle(3254)
        scerror.SetFillColor(ROOT.kBlack)
    # in case of dostat=True: plot total and stat only scaled variation
    else:
        scstaterror.SetFillStyle(1001)
        scstaterror.SetFillColor(ROOT.kGray+1)
        scstaterror.SetMarkerStyle(0)
        scerror.SetFillStyle(3254)
        scerror.SetFillColor(ROOT.kBlack)
        scerror.SetMarkerStyle(0)

    ### operations on data histogram
    datahist.SetMarkerStyle(markerstyle)
    datahist.SetMarkerColor(markercolor)
    datahist.SetMarkerSize(markersize)
    datahist.SetLineColor(markercolor)

    ### calculate data to mc ratio
    ratiograph = ROOT.TGraphAsymmErrors(datahist)
    for i in range(1,datahist.GetNbinsX()+1):
        if not mchistsum.GetBinContent(i)==0:
            ratiograph.GetY()[i-1] *= 1./mchistsum.GetBinContent(i)
            ratiograph.SetPointError(i-1,0,0,datahist.GetBinErrorLow(i)/mchistsum.GetBinContent(i),
                                        datahist.GetBinErrorUp(i)/mchistsum.GetBinContent(i))
        # avoid drawing empty mc or data bins
        else: ratiograph.GetY()[i-1] = 1e6
        if(datahist.GetBinContent(i)<=0): ratiograph.GetY()[i-1] += 1e6

    ### calculate summed histograms for signal and background separately
    sighistsum = mchistlist[0].Clone()
    sighistsum.Reset()
    bkghistsum = mchistlist[0].Clone()
    bkghistsum.Reset()
    for hist in mchistlist:
        ht.cliphistogram(hist)
        if signals is not None and hist.GetTitle() in signals: sighistsum.Add(hist)
        else: bkghistsum.Add(hist)

    ### calculate signal to background ratio
    # note: the 
    sbratiohist = sighistsum.Clone()
    for i in range(1,sighistsum.GetNbinsX()+1):
        if not bkghistsum.GetBinContent(i)==0:
            sbratiohist.SetBinContent(i, sighistsum.GetBinContent(i) / bkghistsum.GetBinContent(i))
            sbratiohist.SetBinError(i, sighistsum.GetBinError(i) / bkghistsum.GetBinContent(i))
        else:
            sbratiohist.SetBinContent(i, 0)
            sbratiohist.SetBinError(i, 0)

    ### style settings for signal to background ratio
    sbratiohist.SetLineWidth(2)
    sbratiohist.SetLineColor(ROOT.kRed)
    sbratiohist.SetFillStyle(0)

    ### make legend for upper plot and add all histograms
    legend = ROOT.TLegend(p1legendbox[0],p1legendbox[1],p1legendbox[2],p1legendbox[3])
    legend.SetNColumns(p1legendncols)
    legend.SetFillStyle(0)
    legend.AddEntry(datahist,datalabel,"pe1")
    for hist in mchistlist[::-1]:
        legend.AddEntry(hist,labelmap.get(hist.GetTitle(), hist.GetTitle()),"f")
    legend.AddEntry(mcerror,"Uncertainty","f")

    ### make legend for lower plot and add all histograms
    # (note: will not be drawn if dostat = False)
    legend2 = ROOT.TLegend(p2legendbox[0],p2legendbox[1],p2legendbox[2],p2legendbox[3])
    legend2.SetNColumns(3) 
    legend2.SetFillStyle(0)
    legend2.AddEntry(scstaterror, "Stat. uncertainty", "f")

    ### make legend for signal to background ratio pad and add all histograms
    # todo

    ### make canvas and pads
    c1 = ROOT.TCanvas("c1","c1")
    c1.SetCanvasSize(canvaswidth, canvasheight)
    pad1 = ROOT.TPad("pad1","",0.,0.4,1.,1.)
    pad1.SetTopMargin(p1topmargin)
    pad1.SetBottomMargin(0.03)
    pad1.SetLeftMargin(leftmargin)
    pad1.SetRightMargin(rightmargin)
    pad1.SetFrameLineWidth(2)
    pad1.Draw()
    pad2 = ROOT.TPad("pad2","",0.,0.25,1.,0.4)
    pad2.SetTopMargin(0.01)
    pad2.SetBottomMargin(0.1)
    pad2.SetLeftMargin(leftmargin)
    pad2.SetRightMargin(rightmargin)
    pad2.SetFrameLineWidth(2)
    pad2.Draw()
    pad3 = ROOT.TPad("pad3","",0.,0.,1.,0.25)
    pad3.SetTopMargin(0.01)
    pad3.SetBottomMargin(p2bottommargin)
    pad3.SetLeftMargin(leftmargin)
    pad3.SetRightMargin(rightmargin)
    pad3.SetFrameLineWidth(2)
    pad3.Draw()

    ### make upper part of the plot
    pad1.cd()
    # determine range of pad
    if(yaxlog): pad1.SetLogy()
    #(histmin, histmax) = getminmax(datahist, mcerror, yaxlog=yaxlog, margin=False)
    #(rangemin, rangemax) = getminmax(datahist, mcerror, yaxlog=yaxlog)
    (histmin, histmax) = getminmaxfromlist([datahist]+mchistlist, yaxlog=yaxlog, margin=False)
    (rangemin,rangemax) = getminmaxfromlist([datahist]+mchistlist, yaxlog=yaxlog)
    if yaxrange is not None: (rangemin,rangemax) = yaxrange
    # temporary modification for an annoying comment on one figure:
    #rangemax = rangemax*1.2
    mcerror.SetMinimum(rangemin)
    mcerror.SetMaximum(rangemax)

    # X-axis layout
    xax = mcerror.GetXaxis()
    xax.SetNdivisions(10,5,0,ROOT.kTRUE)
    xax.SetLabelSize(0)
    # Y-axis layout
    yax = mcerror.GetYaxis()
    yax.SetMaxDigits(4)
    yax.SetNdivisions(10,5,0,ROOT.kTRUE)
    yax.SetLabelFont(10*labelfont+3)
    yax.SetLabelSize(labelsize)
    if yaxtitle is not None: yax.SetTitle(yaxtitle)
    yax.SetTitleFont(10*axtitlefont+3)
    yax.SetTitleSize(axtitlesize)
    yax.SetTitleOffset(ytitleoffset)

    # draw mcerror first to get range correct
    mcerror.Draw("e2")
    # now draw in correct order
    mchiststack.Draw("hist same")
    mcerror.Draw("e2 same")
    datahist.Draw("pe e1 x0 same") 
    # (note: e1 draws error bars, x0 suppresses horizontal error bars)
    legend.Draw("same")
    # draw some extra info if needed
    ROOT.gPad.RedrawAxis()

    # draw header
    lumistr = ''
    if lumi is not None:
        lumistr = '{0:.3g}'.format(lumi/1000.)+' fb^{-1} (13 TeV)'
    pt.drawLumi(pad1,extratext=extracmstext,lumitext=lumistr)

    # draw extra info
    tinfo = ROOT.TLatex()
    tinfo.SetTextFont(10*infofont+3)
    tinfo.SetTextSize(infosize)
    for i,info in enumerate(extrainfos):
        vspace = 0.05*(float(infosize)/20)
        tinfo.DrawLatexNDC(infoleft,infotop-(i+1)*vspace, info)

    # draw secondary bin labels and vertical lines
    if varmode=='double':
        tsecondarybinlabels = ROOT.TLatex()
        tsecondarybinlabels.SetTextFont(10*labelfont+3)
        tsecondarybinlabels.SetTextSize(labelsize)
        tsecondarybinlabels.SetTextAlign(21)
        vlines = []
        for i,label in enumerate(secondarybinlabels):
            labelxpos = 0.5 + i*nprimarybins + nprimarybins/2.
            labelxposndc = (labelxpos-ROOT.gPad.GetX1())/(ROOT.gPad.GetX2()-ROOT.gPad.GetX1())
            labelyposndc = 0.65
            tsecondarybinlabels.DrawLatexNDC(labelxposndc, labelyposndc, label)
            if i!=0:
                linexpos = 0.5+i*nprimarybins
                vline = ROOT.TLine(linexpos, mcerror.GetMinimum(), linexpos, histmax)
                vline.SetLineColor(ROOT.kBlack)
                vline.SetLineStyle(ROOT.kDashed)
                vline.Draw()
                vlines.append(vline)

    ### make the lower part of the plot
    pad2.cd()
    # X-axis layout
    xax = mcerror.GetXaxis()
    xax.SetNdivisions(10,5,0,ROOT.kTRUE)
    xax.SetLabelSize(0)

    # Y-axis layout
    yax = scerror.GetYaxis()
    yax.SetRangeUser(p2yaxrange[0],p2yaxrange[1]);
    yax.SetTitle(p2yaxtitle)
    yax.SetMaxDigits(3)
    yax.SetNdivisions(4,5,0)
    yax.SetLabelFont(10*labelfont+3)
    yax.SetLabelSize(labelsize)
    yax.SetTitleFont(10*axtitlefont+3)
    yax.SetTitleSize(axtitlesize)
    yax.SetTitleOffset(ytitleoffset)

    # draw objects
    scerror.Draw("e2")
    if dostat: scstaterror.Draw("e2 same")
    ratiograph.Draw("p e1 same")
    if dostat: legend2.Draw("same")
    ROOT.gPad.RedrawAxis()

    # make and draw unit ratio line
    xmax = datahist.GetXaxis().GetBinUpEdge(datahist.GetNbinsX())
    xmin = datahist.GetXaxis().GetBinLowEdge(1)
    line = ROOT.TLine(xmin,1,xmax,1)
    line.SetLineStyle(2)
    line.Draw("same")

    # draw vertical lines for secondary variables
    if varmode=='double':
        p2vlines = []
        for i,label in enumerate(secondarybinlabels):
            if i!=0:
                linexpos = 0.5+i*nprimarybins
                vline = ROOT.TLine(linexpos, p2yaxrange[0], linexpos, p2yaxrange[1])
                vline.SetLineColor(ROOT.kBlack)
                vline.SetLineStyle(ROOT.kDashed)
                vline.Draw()
                p2vlines.append(vline)

    ### make the second ratio pad
    pad3.cd()
    # X-axis layout
    xax = sbratiohist.GetXaxis()
    if xaxinteger:
    # option 1:
    #xax.SetOption("I") 
    # does not work properly since it is not defined for TAxis, only TGAxis (?)
    # option 2:
        xax.SetNdivisions(xax.GetNbins()*2,0,0,ROOT.kFALSE)
        for i in range(0,xax.GetNbins()+1):
            xax.ChangeLabel(2*i+1,-1,-1,-1,-1,-1," ")
    # works, but half-integer ticks are still there...
    # option 3: 
    #xax.SetNdivisions(xax.GetNbins(),0,0,ROOT.kFALSE)
    #xax.CenterLabels()
    # does not work properly since labels are still half-integer but simply shifted
    elif binlabels is not None:
        # set bin labels for x-axis
        if len(binlabels)!=scerror.GetNbinsX():
            msg = 'ERROR in histplotter.plotdatavsmc:'
            msg += ' binlabels has invalid length of {}'.format(len(binlabels))
            msg += ' while number of bins is {};'.format(scerror.GetNbinsX())
            msg += ' will ignore provided bin labels.'
            print(msg)
        else:
            for i in range(1,scerror.GetNbinsX()+1):
                label = str(binlabels[i-1])
                xax.SetBinLabel(i, label)
    else: xax.SetNdivisions(10,5,0,ROOT.kTRUE)
    xax.SetLabelSize(labelsize)
    xax.SetLabelFont(10*labelfont+3)
    if xaxtitle is not None: xax.SetTitle(xaxtitle)
    xax.SetTitleFont(10*axtitlefont+3)
    xax.SetTitleSize(axtitlesize)
    xax.SetTitleOffset(xtitleoffset)
    # label rotation
    # does not seem to work with xax.ChangeLabel (only for numeric labels...),
    # so use other option for now
    if labelangle is not None:
        scerror.LabelsOption('v')

    # Y-axis layout
    yax = sbratiohist.GetYaxis()
    if yaxlog: pad3.SetLogy()
    yax.SetRangeUser(0.005, 0.05)
    yax.SetTitle('S/B')
    yax.SetMaxDigits(3)
    yax.SetNdivisions(4,5,0)
    yax.SetLabelFont(10*labelfont+3)
    yax.SetLabelSize(labelsize)
    yax.SetTitleFont(10*axtitlefont+3)
    yax.SetTitleSize(axtitlesize)
    yax.SetTitleOffset(ytitleoffset)

    # draw objects
    sbratiohist.Draw("hist")
    #sbratiohist.Draw("same e2")
    ROOT.gPad.RedrawAxis()

    ### save the plot
    c1.SaveAs(outfile+'.png')
    #c1.SaveAs(outfile+'.eps')
    #c1.SaveAs(outfile+'.pdf')
