#!/usr/bin/env python

"""
Usage:
    keypoint_tracker.py [options] <video> <keypoints> <output>
    keypoint_tracker.py ( -h | --help )

Options:
    --no-video                  Do not show video in output.
    --max-position-sigma=SIGMA  Do not show positions with >SIGMA error in location.
    --max-velocity-sigma=SIGMA  Do not show velocity with >SIGMA error in velocity.
    --max-frames=COUNT          If specified, only process at most COUNT frames.
    --no-cluster                Do not attempt clustering.
    --no-show-states            Do not show tracked states.
    --show-kps                  Show keypoints per frame.
    --trail-length=FRAMES       Show track trails with length FRAMES.
                                [default: 0]

Input video can be anything OpenCV can read. Output video is AVI. Keypoints are
read from an HDF5 file.
"""

import os

import cv2
import docopt
import h5py
import numpy as np
from sklearn.cluster import AffinityPropagation

from tracker import Keypoint, Track, Tracking, Cluster

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
    clusters = []

    while options['--max-frames'] is None or frame_idx < int(options['--max-frames']):
        # Read in frame image
        rv, frame = cap.read()
        frame_idx += 1
        
        # If we failed to read in a frame, exit
        if not rv:
            break
        
        if options['--no-video']:
            output_frame = np.zeros_like(frame)
        else:
            output_frame = np.copy(frame)

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

        if options['--show-kps']:
            for kp in kps:
                x, y = kp.location
                cv2.circle(output_frame, (int(x), int(y)), 5, (0,0,200), lineType=cv2.CV_AA)
        
        # Track this frame's keypoints
        tracking.add_frame(frame_pair[0], frame_pair[1], frame_idx, kps)
        
        # All states and covariances for this frame
        frame_states, frame_covars, frame_track_kps = [], [], []
        for t in tracking.tracks:
            if t.final_frame_idx < frame_idx or t.initial_frame_idx > frame_idx:
                continue
            frame_states.append(t.states[frame_idx - t.initial_frame_idx])
            frame_covars.append(t.covariances[frame_idx - t.initial_frame_idx].copy())
            frame_track_kps.append(t.associated_keypoints[-1])

        # Draw trails if required
        trail_length = int(options['--trail-length'])
        if trail_length > 0:
            for t in tracking.tracks:
                if t.final_frame_idx <= frame_idx - trail_length or t.initial_frame_idx > frame_idx:
                    continue

                start_frame = frame_idx - trail_length + 1
                start_idx = start_frame - t.initial_frame_idx

                for s1, s2 in zip(t.states[start_idx:-1], t.states[start_idx+1:]):
                    cv2.line(output_frame, (int(s1[0]), int(s1[1])),
                            (int(s2[0]), int(s2[1])), (200,0,200),
                            lineType=cv2.CV_AA)
            
        # Convert states to an array
        frame_states = np.array(frame_states)

        if not options['--no-cluster']:
            # PDF of choosing kp uniformly from image
            h, w = frame.shape[:2]
            non_cluster_pdf = -30

            # Best existing cluster for each state and the associated PDF
            state_association = [(-1, non_cluster_pdf),] * frame_states.shape[0]

            # PDF of choosing states from each active cluster
            for c_idx, cluster in enumerate(clusters):
                # skip elderly clusters
                if cluster.last_update_frame_idx != frame_idx - 1:
                    continue

                cluster_mu, cluster_sigma = cluster.predict(frame_idx)

                for s_idx in xrange(len(state_association)):
                    s = frame_states[s_idx,:]
                    c = frame_covars[s_idx]
                    _, current_pdf = state_association[s_idx]

                    pdf = mv_gaussian_log_pdf(s, cluster_mu, cluster_sigma + c)[0]
                    if pdf > current_pdf:
                        state_association[s_idx] = (c_idx, pdf)

            # Go through associations
            unassigned_states, unassigned_covars = [], []
            cluster_states = [None,] * len(clusters)
            for s, c, assoc in zip(frame_states, frame_covars, state_association):
                c_idx = assoc[0]
                if c_idx < 0:
                    unassigned_states.append(s)
                    unassigned_covars.append(c)
                    continue
                
                if cluster_states[c_idx] is None:
                    cluster_states[c_idx] = [(s, c)]
                else:
                    cluster_states[c_idx].append((s, c))

            for cluster, assignment in zip(clusters, cluster_states):
                if assignment is None:
                    if cluster.final_frame_idx >= frame_idx - 3:
                        cluster.update(frame_idx)
                else:
                    states = np.array(list(s for s,c in assignment))

                    if states.shape[0] >= 2:
                        sigma = np.cov(states.T)
                    else:
                        sigma = cluster.covariances[-1].copy()

                    mu = np.mean(states, axis=0)
                    for _, cov in assignment:
                        sigma += cov

                    cluster.update(frame_idx, mu, sigma)

                    minx, maxx = states[:,0].min(), states[:,0].max()
                    miny, maxy = states[:,1].min(), states[:,1].max()

                    if maxx - minx > 300 or maxy - miny > 300:
                        continue

                    cv2.rectangle(output_frame,
                            (int(minx), int(miny)), (int(maxx), int(maxy)), (0,0,200), lineType=cv2.CV_AA)

                    state, cov = cluster.predict(frame_idx)
                    draw_cov(output_frame, cov[:2,:2], state[:2], (0,0,200), lineType=cv2.CV_AA)

        # Draw 'o' over each frame state
        sc = 10.0
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

            ## Only those with keypoints at this frame
            #if kp.frame_idx != frame_idx:
            #    continue
                
            # Only those with minimum velocity
            #speed = np.sqrt(np.sum(s[2:4]*s[2:4]))
            #if speed < 0.5:
            #    continue
            
            filtered_states.append(s)
            filtered_covs.append(c)

            if not options['--no-show-states']:
                draw_cov(output_frame, c[:2,:2], s[:2], (255,0,0), lineType=cv2.CV_AA)
                cv2.line(output_frame, (int(s[0]), int(s[1])), (int(s[0]+sc*s[2]),
                    int(s[1]+sc*s[3])), (0,200,0), lineType=cv2.CV_AA)
                draw_cov(output_frame, sc*sc*c[2:4,2:4], s[:2]+sc*s[2:4], (0,200,0), lineType=cv2.CV_AA)

        filtered_states = np.array(filtered_states)

        # Cluster unlabelled states
        if not options['--no-cluster'] and len(unassigned_states) > 4:
            cluster_states = np.copy(np.array(unassigned_states))
            cluster_covs = list(unassigned_covars)

            clustering = AffinityPropagation()
            labels = clustering.fit_predict(cluster_states)

            # Process labels
            for label in np.unique(labels):
                label_indices = np.nonzero(labels == label)[0]
                if label_indices.shape[0] < 2:
                    continue

                label_states = cluster_states[label_indices, :]
                label_covs = list(cluster_covs[i] for i in label_indices)
                
                mu = np.mean(label_states, axis=0)
                sigma = np.cov(label_states.T)

                for c in label_covs:
                    sigma += c

                new_cluster = Cluster()
                new_cluster.update(frame_idx, mu, sigma)
                clusters.append(new_cluster)

                draw_cov(output_frame, sigma[:2,:2], mu, (0,200,200), lineType=cv2.CV_AA)
        
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

    # Skip invalid covariance matrices
    if np.any(vals <= 0):
        return

    width, height = nstd * np.sqrt(vals)
    cv2.ellipse(im, (int(pos[0]), int(pos[1])), (int(width), int(height)), theta, 0, 360, color, **kwargs)

def mv_gaussian_log_pdf(X, mu, sigma):
    d = np.linalg.det(sigma)
    if d < 1e-8:
        return -np.inf * np.ones(X.shape[0])
    X = np.atleast_2d(X)
    sigma_inv = np.linalg.inv(sigma)
    k = mu.shape[0]
    norm = (-0.5) * (k*np.log(2*np.pi) + np.log(np.linalg.det(sigma)))
    rvs = list(norm + (-0.5)*(((x-mu).T).dot(sigma_inv.dot(x-mu))) for x in X)
    return np.array(rvs)

if __name__ == '__main__':
    main()
