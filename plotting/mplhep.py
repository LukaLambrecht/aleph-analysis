import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def histplot(histograms,
      stack = True,
      bins = None,
      histtype = 'fill',
      color = None,
      edgecolor = None,
      linewidth = None,
      alpha = None,
      label = None,
      ax = None):

    # handle case where only one instance was provided
    if not isinstance(histograms, list):
        histograms = [histograms]
        color = [color]
        edgecolor = [edgecolor]
        label = [label]
    if not isinstance(alpha, list): alpha = [alpha]*len(histograms)
    if not isinstance(linewidth, list): linewidth = [linewidth]*len(histograms)

    for idx in range(len(histograms)):

        # set baseline
        line = histograms[idx]
        baseline = np.zeros(len(line))
        if stack:
            baseline = sum(histograms[:idx])
            line = baseline + histograms[idx]
            

        # make plot
        ax.stairs(line,
                  baseline = baseline,
                  edges = bins,
                  fill = (histtype=='fill'),
                  linewidth = linewidth[idx],
                  color = color[idx],
                  label = label[idx],
                  alpha = alpha[idx])

    return ax
