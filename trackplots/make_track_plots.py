import os
import sys
import json
import uproot
import argparse
import numpy as np
import awkward as ak
from fnmatch import fnmatch
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(topdir)

from tools.samplelisttools import read_sampledict


def get_track_curve(d0, z0, phi0, omega, tanlambda, s):
    
    # parameter parsing
    x0 = -d0 * np.sin(phi0)
    y0 = d0 * np.cos(phi0)

    # make parametrized coordinates
    x = x0 + 1/omega * (np.sin(phi0 + omega*s) - np.sin(phi0))
    y = y0 - 1/omega * (np.cos(phi0 + omega*s) - np.cos(phi0))
    z = z0 + s * tanlambda

    # also get direction vector at the reference point
    dxds = np.cos(phi0)
    dyds = np.sin(phi0)
    dzds = tanlambda
    direction = np.array([dxds, dyds, dzds])
    direction = direction / np.sqrt(np.sum(np.square(direction)))

    return ( (x, y, z), (x0, y0, z0), direction )


if __name__=='__main__':

    # settings
    inputfile = sys.argv[1]
    eventidx = int(sys.argv[2])
    sposrange = 1
    snegrange = 1
    #xlim = (-0.07, 0.07)
    #ylim = (-0.07, 0.07)
    xlim = (-0.1, 0.1)
    ylim = (-0.1, 0.1)
    zlim = None
    #zlim = (-0.4, -0.2)
    trackn = None
    #trackn = 5
    do_pv = True
    do_genpv = False
    do_refpoint = True
    do_pcapoint = True
    do_jetaxis = True
    do_tracktojet_pcapoint = False
    do_jettotrack_pcapoint = False
    do_linetopv_pcapoint = True

    # set branches to read
    branches_to_read = [
      'JetsConstituents_d0_wrt0',
      'JetsConstituents_z0_wrt0',
      'JetsConstituents_phi0_wrt0',
      'JetsConstituents_omega_wrt0',
      'JetsConstituents_tanlambda_wrt0',
    ]
    if do_pv or do_jetaxis:
        branches_to_read += [
            'PV_x',
            'PV_y',
            'PV_z',
        ]
    if do_genpv:
        branches_to_read += [
            'GenPV_x',
            'GenPV_y',
            'GenPV_z',
        ]
    if do_pcapoint:
        branches_to_read += [
            'JetsConstituents_dxy',
            'JetsConstituents_dz',
            'JetsConstituents_phi',
        ]
    if do_jetaxis:
        branches_to_read += [
            'Jets_px',
            'Jets_py',
            'Jets_pz',
        ]
    if do_tracktojet_pcapoint:
        branches_to_read += [
            'JetsConstituents_trackPCAToJetAxis_x',
            'JetsConstituents_trackPCAToJetAxis_y',
            'JetsConstituents_trackPCAToJetAxis_z',
        ]
    if do_jettotrack_pcapoint:
        branches_to_read += [
            'JetsConstituents_jetAxisPCAToTrack_x',
            'JetsConstituents_jetAxisPCAToTrack_y',
            'JetsConstituents_jetAxisPCAToTrack_z',
        ]
    if do_linetopv_pcapoint:
        branches_to_read += [
            'JetsConstituents_linePCAToPrimaryVertex_x',
            'JetsConstituents_linePCAToPrimaryVertex_y',
            'JetsConstituents_linePCAToPrimaryVertex_z',
        ]


    # read one event from input file
    key = 'temp'
    sampledict = {key: [inputfile]}
    events = read_sampledict(sampledict, treename='events',
               branches=branches_to_read,
               entry_start=eventidx,
               entry_stop=eventidx+1,
               verbose=True)
    events = events[key]

    # get primary vertex
    if do_pv or do_jetaxis:
        PV_x = events['PV_x'][0]
        PV_y = events['PV_y'][0]
        PV_z = events['PV_z'][0]
    if do_genpv:
        GenPV_x = events['GenPV_x'][0]
        GenPV_y = events['GenPV_y'][0]
        GenPV_z = events['GenPV_z'][0]

    # get track parameters (flatten over jets)
    d0 = ak.flatten(events['JetsConstituents_d0_wrt0'][0]).to_numpy()
    z0 = ak.flatten(events['JetsConstituents_z0_wrt0'][0]).to_numpy()
    phi0 = ak.flatten(events['JetsConstituents_phi0_wrt0'][0]).to_numpy()
    omega = ak.flatten(events['JetsConstituents_omega_wrt0'][0]).to_numpy()
    tanlambda = ak.flatten(events['JetsConstituents_tanlambda_wrt0'][0]).to_numpy()

    # set colors per jet
    color_values = []
    njets = len(events['JetsConstituents_d0_wrt0'][0])
    for jet_idx in range(njets):
        color_value = jet_idx / (njets - 1)
        length = len(events['JetsConstituents_d0_wrt0'][0][jet_idx])
        color_values.append(np.ones(length)*color_value)
    color_values = np.concatenate(tuple(color_values))
    cmap = plt.get_cmap('winter')

    # make parametric curves for tracks
    track_data = []
    spos = np.linspace(0, sposrange, num=100)
    sneg = np.linspace(-snegrange, 0, num=100)
    for track_idx in range(len(d0)):

        # get parameters for this track
        this_d0 = d0[track_idx]
        this_z0 = z0[track_idx]
        this_phi0 = phi0[track_idx]
        this_omega = omega[track_idx]
        this_tanlambda = tanlambda[track_idx]

        # skip neutral constituents
        if np.abs(this_d0 + 9) < 1e-12: continue

        # get track curve
        res = get_track_curve(this_d0, this_z0, this_phi0, this_omega, this_tanlambda, spos)
        (track_coords, refpoint_coords, refpoint_direction) = res
        ext_coords = get_track_curve(this_d0, this_z0, this_phi0, this_omega, this_tanlambda, sneg)[0]

        # add results to list
        track_data.append({
            'track_idx': track_idx,
            'track_coords': track_coords,
            'ext_coords': ext_coords,
            'refpoint_coords': refpoint_coords,
            'refpoint_direction': refpoint_direction
        })

    # make parametric curves for jets
    jet_data = []
    if do_jetaxis:
        spos = np.linspace(0, sposrange/10, num=100)
        jet_px = events['Jets_px'][0].to_numpy()
        jet_py = events['Jets_py'][0].to_numpy()
        jet_pz = events['Jets_pz'][0].to_numpy()
        
        for jet_idx in range(len(jet_px)):

            # get parameters for this jet
            this_px = jet_px[jet_idx]
            this_py = jet_py[jet_idx]
            this_pz = jet_pz[jet_idx]

            # get jet line
            jet_coords = (
                PV_x + spos * this_px,
                PV_y + spos * this_py,
                PV_z + spos * this_pz
            )
        
            # add results to list
            jet_data.append({
                'jet_idx': jet_idx,
                'jet_coords': jet_coords,
            })

    # find PCA of track to primary vertex
    pca_coords = []
    if do_pcapoint:
        dxy = ak.flatten(events['JetsConstituents_dxy'][0]).to_numpy()
        dz = ak.flatten(events['JetsConstituents_dz'][0]).to_numpy()
        phi = ak.flatten(events['JetsConstituents_phi'][0]).to_numpy()
        
        for track_idx in range(len(d0)):

            # get parameters for this track
            this_dxy = dxy[track_idx]
            this_dz = dz[track_idx]
            this_phi = phi[track_idx]

            # skip neutral constituents
            if np.abs(this_dxy + 9) < 1e-12: continue

            # get PCA to primary vertex
            pca_point = (PV_x - this_dxy*np.sin(this_phi), PV_y + this_dxy*np.cos(this_phi), PV_z + this_dz)
            pca_coords.append(pca_point)

    # find PCA of track to jet axis
    tracktojet_pca_coords = []
    if do_tracktojet_pcapoint:
        x = ak.flatten(events['JetsConstituents_trackPCAToJetAxis_x'][0]).to_numpy()
        y = ak.flatten(events['JetsConstituents_trackPCAToJetAxis_y'][0]).to_numpy()
        z = ak.flatten(events['JetsConstituents_trackPCAToJetAxis_z'][0]).to_numpy()

        for track_idx in range(len(d0)):

            # get parameters for this track
            this_x = x[track_idx]
            this_y = y[track_idx]
            this_z = z[track_idx]

            # skip neutral constituents
            if np.abs(this_x) < 1e-12: continue

            # get track PCA to jet
            pca_point = (this_x, this_y, this_z)
            tracktojet_pca_coords.append(pca_point)

    # find PCA of jet axis to track
    jettotrack_pca_coords = []
    if do_jettotrack_pcapoint:
        x = ak.flatten(events['JetsConstituents_jetAxisPCAToTrack_x'][0]).to_numpy()
        y = ak.flatten(events['JetsConstituents_jetAxisPCAToTrack_y'][0]).to_numpy()
        z = ak.flatten(events['JetsConstituents_jetAxisPCAToTrack_z'][0]).to_numpy()

        for track_idx in range(len(d0)):

            # get parameters for this track
            this_x = x[track_idx]
            this_y = y[track_idx]
            this_z = z[track_idx]

            # skip neutral constituents
            if np.abs(this_x) < 1e-12: continue

            # get track PCA to jet
            pca_point = (this_x, this_y, this_z)
            jettotrack_pca_coords.append(pca_point)

    # find PCA of line to PV
    linetopv_pca_coords = []
    if do_linetopv_pcapoint:
        x = ak.flatten(events['JetsConstituents_linePCAToPrimaryVertex_x'][0]).to_numpy()
        y = ak.flatten(events['JetsConstituents_linePCAToPrimaryVertex_y'][0]).to_numpy()
        z = ak.flatten(events['JetsConstituents_linePCAToPrimaryVertex_z'][0]).to_numpy()

        for track_idx in range(len(d0)):

            # get parameters for this track
            this_x = x[track_idx]
            this_y = y[track_idx]
            this_z = z[track_idx]

            # skip neutral constituents
            if np.abs(this_x) < 1e-12: continue

            # get track PCA to jet
            pca_point = (this_x, this_y, this_z)
            linetopv_pca_coords.append(pca_point)

    # check for debugging: check whether the track reference point is the PCA to the origin or to the beamline
    print('Dot-product between reference-point-connector and track direction:')
    for idx in range(len(track_data)):
        refpoint = np.array(track_data[idx]['refpoint_coords'])
        connector = refpoint / np.sqrt(np.sum(np.square(refpoint)))
        direction = track_data[idx]['refpoint_direction']
        dot3d = np.sum(np.multiply(connector, direction))
        dot2d = np.sum(np.multiply(connector[:-1], direction[:-1]))
        print('3D: ', dot3d)
        print('2D: ', dot2d)
        print('---')

    # check for debugging: same as above but for PCA w.r.t. primary vertex
    if do_pcapoint and do_pv:
      print('Dot-product between pca-connector and track direction:')
      for idx in range(len(track_data)):
        pcapoint = np.array(pca_coords[idx])
        pv = np.array([PV_x, PV_y, PV_z])
        connector = pcapoint - pv
        connector = connector / np.sqrt(np.sum(np.square(connector)))
        direction = track_data[idx]['refpoint_direction']
        dot3d = np.sum(np.multiply(connector, direction))
        dot2d = np.sum(np.multiply(connector[:-1], direction[:-1]))
        print('3D: ', dot3d)
        print('2D: ', dot2d)
        print('---')

    # check for debugging: verify that the line connecting the line-PCA to the primary vertex
    # is orthogonal to the line direction.
    if do_pv and do_linetopv_pcapoint and do_tracktojet_pcapoint:
      print('Dot-product between connector and line direction')
      print('(for ALEPH-style PCA):')
      for idx in range(len(track_data)):
        linepca = np.array(linetopv_pca_coords[idx])
        trackpca = np.array(tracktojet_pca_coords[idx])
        pv = np.array([PV_x, PV_y, PV_z])
        connector = (linepca - pv)
        connector = connector / np.sqrt(np.sum(np.square(connector)))
        direction = (linepca - trackpca)
        direction = direction / np.sqrt(np.sum(np.square(direction)))
        dot = np.sum(np.multiply(connector, direction))
        print(dot)
        # alternative: use track direction
        direction = track_data[idx]['refpoint_direction']
        dot = np.sum(np.multiply(connector, direction))
        print(dot)
        print('---')

    # check for debugging: verify that the line connecting the (builtin) PCA
    # is orthogonal to the track direction.
    if do_pv and do_pcapoint and do_tracktojet_pcapoint and do_linetopv_pcapoint:
      print('Dot product between connector and track direction')
      print('(for FCCAnalyses-style PCA):')
      for idx in range(len(track_data)):
        pca = np.array(pca_coords[idx])
        pv = np.array([PV_x, PV_y, PV_z])
        connector = (pca - pv)
        connector = connector / np.sqrt(np.sum(np.square(connector)))
        direction = track_data[idx]['refpoint_direction']
        dot = np.sum(np.multiply(connector, direction))
        print(dot)
        # alternative: use line direction
        linepca = np.array(linetopv_pca_coords[idx])
        trackpca = np.array(tracktojet_pca_coords[idx])
        direction = (linepca - trackpca)
        direction = direction / np.sqrt(np.sum(np.square(direction)))
        dot = np.sum(np.multiply(connector, direction))
        print(dot)
        print('---')

    # make a plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axis('equal')

    # add the tracks
    for counter, track_params in enumerate(track_data):
        if trackn is not None and counter != trackn: continue

        track_idx = track_params['track_idx']
        color = cmap(color_values[track_idx])
        x, y, z = track_params['track_coords']
        ax.plot(x, y, z, color=color)
        x, y, z = track_params['ext_coords']
        ax.plot(x, y, z, color=color, linestyle='--')

        # add points
        if do_refpoint:
            x, y, z = track_params['refpoint_coords']
            ax.scatter(x, y, z, color=color, s=10)
        if do_pcapoint:
            x, y, z = pca_coords[counter]
            ax.scatter(x, y, z, color='firebrick', zorder=1000, s=10)
        if do_tracktojet_pcapoint:
            x, y, z = tracktojet_pca_coords[counter]
            ax.scatter(x, y, z, color='dodgerblue', zorder=1000, s=10)
        if do_jettotrack_pcapoint:
            x, y, z = jettotrack_pca_coords[counter]
            ax.scatter(x, y, z, color='firebrick', zorder=1000, s=10)
        if do_linetopv_pcapoint:
            x, y, z = linetopv_pca_coords[counter]
            ax.scatter(x, y, z, color='lightblue', zorder=1000, s=10)

    # add the jets
    for counter, jet_params in enumerate(jet_data):

        jet_idx = jet_params['jet_idx']
        x, y, z = jet_params['jet_coords']
        ax.plot(x, y, z, color='red')

    # add the origin and PV
    ax.scatter(0, 0, 0, color='black', s=10, zorder=1000, label='Origin')
    if do_pv:
        ax.scatter(PV_x, PV_y, PV_z, color='red', s=10, zorder=1000, label='PV')
    if do_genpv:
        ax.scatter(GenPV_x, GenPV_y, GenPV_z, color='green', s=10, zorder=1000, label='Gen PV')

    # add the z-axis / beamline
    zlims = ax.get_zlim()
    ax.plot([0, 0], [0, 0], [zlims[0], zlims[1]], color='black', linestyle='dashed')

    # plot aesthetics
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(xlim)
    if zlim is not None: ax.set_zlim(zlim)
    ax.set_xlabel('x [cm]', fontsize=12)
    ax.set_ylabel('y [cm]', fontsize=12)
    ax.set_zlabel('z [cm]', fontsize=12)
    ax.legend(fontsize=12)
  
    fig.tight_layout()
    fig.savefig('test.png', dpi=300)

    # define helper function for projection plots
    def make_projection_plot(axis):
    
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.axis('equal')

        coords_ids = None
        coords_label = None
        if axis=='x':
            coords_ids = [1, 2]
            coords_labels = ['y', 'z']
        elif axis=='y':
            coords_ids = [0, 2]
            coords_labels = ['x', 'z']
        elif axis=='z':
            coords_ids = [0, 1]
            coords_labels = ['x', 'y']

        # add the tracks
        for counter, track_params in enumerate(track_data):
            if trackn is not None and counter != trackn: continue

            track_idx = track_params['track_idx']
            color = cmap(color_values[track_idx])
            coords = track_params['track_coords']
            ax.plot(coords[coords_ids[0]], coords[coords_ids[1]], color=color)
            coords = track_params['ext_coords']
            ax.plot(coords[coords_ids[0]], coords[coords_ids[1]], color=color, linestyle='--')

            # add points
            if do_refpoint:
                coords = track_params['refpoint_coords']
                ax.scatter(coords[coords_ids[0]], coords[coords_ids[1]], color=color, s=10)
            if do_pcapoint:
                coords = pca_coords[counter]
                ax.scatter(coords[coords_ids[0]], coords[coords_ids[1]], color='firebrick', zorder=1000, s=10)
            if do_tracktojet_pcapoint:
                coords = tracktojet_pca_coords[counter]
                ax.scatter(coords[coords_ids[0]], coords[coords_ids[1]], color='dodgerblue', zorder=1000, s=10)
            if do_jettotrack_pcapoint:
                coords = jettotrack_pca_coords[counter]
                ax.scatter(coords[coords_ids[0]], coords[coords_ids[1]], color='firebrick', zorder=1000, s=10)
            if do_linetopv_pcapoint:
                coords = linetopv_pca_coords[counter]
                ax.scatter(coords[coords_ids[0]], coords[coords_ids[1]], color='lightblue', zorder=1000, s=10)

        # add the jets
        for counter, jet_params in enumerate(jet_data):
            jet_idx = jet_params['jet_idx']
            coords = jet_params['jet_coords']
            ax.plot(coords[coords_ids[0]], coords[coords_ids[1]], color='red')

        # add the origin and PV
        ax.scatter(0, 0, color='black', s=10, zorder=1000, label='Origin')
        if do_pv:
            coords = [PV_x, PV_y, PV_z]
            ax.scatter(coords[coords_ids[0]], coords[coords_ids[1]], color='red', s=10, zorder=1000, label='PV')
        if do_genpv:
            coords = [GenPV_x, GenPV_y, GenPV_z]
            ax.scatter(coords[coords_ids[0]], coords[coords_ids[1]], color='green', s=10, zorder=1000, label='Gen PV')

        # plot aesthetics
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        ax.set_xlabel(f'{coords_labels[0]} [cm]', fontsize=12)
        ax.set_ylabel(f'{coords_labels[1]} [cm]', fontsize=12)
        ax.legend(fontsize=12)

        fig.tight_layout()
        fig.savefig(f'test_proj{axis}.png', dpi=300)

    # make a 2D projections
    make_projection_plot('z')
    make_projection_plot('x')
    make_projection_plot('y')
