import os
import sys
import math
import numpy as np


def cliphistogram(hist):
    counts, errors = hist[0], hist[1]
    errors = np.where(counts < 0, 0, errors)
    counts = np.where(counts < 0, 0, counts)
    return (counts, errors)

def clonehistogram(hist):
    return (np.copy(hist[0]), np.copy(hist[1]))

def scalehistogram(hist, factor):
    scaled = clonehistogram(hist)
    scaled = (scaled[0]*factor, scaled[1]*abs(factor))
    return scaled

def addhistograms(hist1, hist2, factor=1):
    scaled = scalehistogram(hist2, factor)
    counts = hist1[0] + scaled[0]
    errors = np.sqrt(np.square(hist1[1]) + np.square(scaled[1]))
    return (counts, errors)

def binperbinmaxvar(histlist, refhist):
    counts = np.array([hist[0] for hist in histlist])
    refcounts = refhist[0]
    diff = np.abs(counts - refcounts[np.newaxis, :])
    maxdiff = np.amax(diff, axis=0)
    return (maxdiff, np.zeros(counts.shape[1]))

def rootsumsquare(histlist):
    counts = np.array([hist[0] for hist in histlist])
    rss = np.sqrt(np.sum(np.square(counts), axis=0))
    return (rss, np.zeros(counts.shape[1]))
