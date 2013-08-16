#!/usr/bin/env python

"""
Usage:
    keypoint_tracker.py [options] <video> <keypoints> <output>
    keypoint_tracker.py ( -h | --help )

Options:
    --no-video                  Do not show video in output.
    --max-position-sigma=SIGMA  Do not show positions with >SIGMA error in
                                location. [default: None]
    --max-velocity-sigma=SIGMA  Do not show positions with >SIGMA error in
                                velocity. [default: None]

Input video can be anything OpenCV can read. Output video is AVI. Keypoints are
read from an HDF5 file.
"""

import os

import cv2
import docopt
import h5py
import numpy as np
from sklearn.cluster import AffinityPropagation

from tracker import Keypoint, Track, Tracking

def main():
    options = docopt.docopt(__doc__)

    features_file = h5py.File(options['<keypoints>'])
    cap = cv2.VideoCapture(options['<video>'])

    frame_idx = -1
    tracks = None
    frame_pair = (None, None)
    tracking = Tracking()
    cluster_tracks = []
    video_writer = None

    while True:
        # Read in frame image
        rv, frame = cap.read()
        frame_idx += 1
        
        # If we failed to read in a frame, exit
        if not rv:
            break

        if video_writer is None:
            h, w = frame.shape[:2]
            video_writer = cv2.VideoWriter(options['<output>'], cv2.cv.FOURCC(*'MJPG'), 25, (w,h), )

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
        
        # Track this frame's keypoints
        tracking.add_frame(frame_pair[0], frame_pair[1], frame_idx, kps)
        
        # All states and covariances for this frame
        frame_states, frame_covars, frame_track_kps = [], [], []
        for t in tracking.tracks:
            if t.final_frame_idx < frame_idx or t.initial_frame_idx > frame_idx:
                continue
            frame_states.append(t.states[frame_idx - t.initial_frame_idx])
            frame_covars.append(t.covariances[frame_idx - t.initial_frame_idx])
            frame_track_kps.append(t.associated_keypoints[-1])
            
        # Convert states to an array
        frame_states = np.array(frame_states)
        
        if options['--no-video']:
            output_frame = np.zeros_like(frame)
        else:
            output_frame = np.copy(frame)

        # Draw 'o' over each frame state
        sc = 10
        filtered_states = []
        filtered_covs = []
        
        for kp, s, c in zip(frame_track_kps, frame_states, frame_covars):
            # Extract sigmas
            sigmas = np.diag(np.linalg.cholesky(c))

            # Only sufficiently 'good' features pass
            if options['--max-position-sigma'] is not None:
                if np.any(sigmas[:2] > float(options['--max-position-sigma'])):
                    continue

            if options['--max-velocity-sigma'] is not None:
                if np.any(sigmas[2:4] > float(options['--max-velocity-sigma'])):
                    continue

            # Only those with keypoints at this frame
            if kp.frame_idx != frame_idx:
                continue
                
            # Only those with minimum velocity
            #speed = np.sqrt(np.sum(s[2:4]*s[2:4]))
            #if speed < 0.5:
            #    continue
            
            filtered_states.append(s)
            filtered_covs.append(c)

            draw_cov(output_frame, c[:2,:2], s[:2], (255,0,0), lineType=cv2.CV_AA)
            cv2.line(output_frame, (int(s[0]), int(s[1])), (int(s[0]+sc*s[2]), int(s[1]+sc*s[3])), (0,200,0), lineType=cv2.CV_AA)
            draw_cov(output_frame, sc*sc*c[2:4,2:4], s[:2]+sc*s[2:4], (0,200,0), lineType=cv2.CV_AA)

        filtered_states = np.array(filtered_states)

        # Cluster unlabelled states
        cluster_states = np.copy(frame_states)
        cluster_covs = list(frame_covars)

        clustering = AffinityPropagation()
        labels = clustering.fit_predict(cluster_states)

        # Process labels
        for label in np.unique(labels):
            label_indices = np.nonzero(labels == label)[0]
            if label_indices.shape[0] < 3:
                continue

            label_states = cluster_states[label_indices, :]
            label_covs = list(cluster_covs[i] for i in label_indices)
            
            mu = np.mean(label_states, axis=0)
            sigma = np.cov(label_states.T)

            for c in label_covs:
                sigma += c

            draw_cov(output_frame, sigma[:2,:2], mu, (0,0,200), lineType=cv2.CV_AA)
        
        # Write output
        video_writer.write(output_frame)
        
    del video_writer

def draw_cov(im, cov, pos, color, nstd=2, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = nstd * np.sqrt(vals)
    cv2.ellipse(im, (int(pos[0]), int(pos[1])), (int(width), int(height)), theta, 0, 360, color, **kwargs)

if __name__ == '__main__':
    main()
