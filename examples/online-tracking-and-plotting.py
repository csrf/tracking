# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
import matplotlib
matplotlib.use('agg')

from matplotlib.pyplot import *
rcParams['figure.figsize'] = (16,9)

import numpy as np

# <codecell>

# The root filename for the video to load
ROOT = 'PVTRA301a04'
#ROOT = 'PVTRA101b06'
DATA_DIR = '/data/rjw57/PVTR/video'

# The maximum number of frames to load
MAX_FRAMES = 500

# <codecell>

import os
import h5py

# <markdowncell>

# Restructure the features into a list of frames. Each frame matrix is Nx(3+168) where the first three columns are x, y and scale values and the remainder are keypoint descriptors.

# <codecell>

features_file = h5py.File(os.path.join(DATA_DIR, ROOT) + '.h5')

# <codecell>

from tracker import Keypoint, Track, Tracking

# <codecell>

# Find states for a particular frame
def frame_states(tracks, frame_idx):
    scs = []
    for t in tracks:
        if t.associated_keypoints[0].frame_idx > frame_idx or t.associated_keypoints[-1].frame_idx < frame_idx:
            continue
        
        state = t.states[frame_idx - t.associated_keypoints[0].frame_idx]
        covariance = t.covariances[frame_idx - t.associated_keypoints[0].frame_idx]
        scs.append((t, state, covariance))
    return scs

# <codecell>

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

# <codecell>

from matplotlib.patches import Ellipse

# <codecell>

import cv2
cap = cv2.VideoCapture(os.path.join(DATA_DIR, ROOT) + '.mp4')
frame_idx = -1
tracks = None
frame_pair = (None, None)
tracking = Tracking()
while frame_idx < MAX_FRAMES:
    # Read in frame image
    rv, frame = cap.read()
    frame_idx += 1
    
    # If we failed to read in a frame, exit
    if not rv:
        break
        
    # Show progress
    if frame_idx % 100 == 0:
        print('Frame index: {0} => {1} tracks'.format(frame_idx, len(tracking.tracks)))
        
    # Convert to greyscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
    # Update frame pair
    frame_pair = (frame_pair[1], frame_gray)
    
    # Work out where in the keypoints file, keypoints start and end
    frame_kp_start, n_kps = features_file['frames'][frame_idx]
    
    # Find keypoint locations and descriptors
    kp_locs = features_file['keypoints'][frame_kp_start:(frame_kp_start+n_kps)]
    kp_descs = features_file['descriptors'][frame_kp_start:(frame_kp_start+n_kps)]
    
    # Convert locations to image space
    kp_im_locs = np.array(kp_locs, dtype=np.float32)
    h, w = frame_gray.shape
    kp_im_locs[:,0] += 0.5*w
    kp_im_locs[:,1] += 0.5*h

    # Construct a list of key points
    kps = list(Keypoint(frame_idx, loc[:2], loc[2], desc) for loc, desc in zip(kp_im_locs, kp_descs))
    
    tracking.add_frame(frame_pair[0], frame_pair[1], frame_idx, kps)
    
    def good_track_and_state(t, s, c):
        # Mandate localisation to within 4 pixels
        if np.diag(c).max() > 4*4:
            return False
        
        kps = list(kp for kp in t.associated_keypoints if kp.frame_idx <= frame_idx)
        if len(kps) < 3:
            return False
        
        return True
        
    scs = list((t, s, c) for t, s, c in frame_states(tracking.tracks, frame_idx) if good_track_and_state(t,s,c))
    states = np.array(list(s for _, s, _ in scs))
    covs = list(c for _, _, c in scs)
   
    # Plot this frame
    
    clf()
    imshow(frame)   
    axis('off')
    tight_layout()
    autoscale(False)
    
    if states.shape[0] > 0:
        quiver(states[:,0], states[:,1], states[:,2], states[:,3], color='g', angles='xy', scale_units='xy', scale=0.1)
        A = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])
        for s, c in zip(states, covs):
            centre = A.dot(s)
            blob = A.dot(c).dot(A.T)
            plot_cov_ellipse(blob, centre, facecolor='b', alpha=0.1)
        scatter(states[:,0], states[:,1])
        
    savefig('frame-{0:05d}.png'.format(frame_idx))

# <codecell>


