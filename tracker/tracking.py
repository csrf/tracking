import cv2
import numpy as np
import pyflann

# covariances of state evolution and observation
evolution_cov = np.power(np.diag((1e0, 1e0, 1e-2, 1e-2)), 2)

# this is estimated from keypoint scale
# observation_cov = np.power(np.diag((1, 1)), 2)

initial_state = np.array([0, 0, 0, 0])
initial_covariance = np.power(np.diag((1e8, 1e8, 1e1, 1e1)), 2)

state_evolution_mat = np.array([
    [1, 0, 1, 0,],
    [0, 1, 0, 1,],
    [0, 0, 1, 0,],
    [0, 0, 0, 1,],
], dtype=np.float32)

state_observation_mat = np.array([
    [1, 0, 0, 0,],
    [0, 1, 0, 0,],
], dtype=np.float32)

class Keypoint(object):
    def __init__(self, frame_idx, location, scale, descriptor):
        """Construct a new keypoint structure. *frame_idx* is the index of the
        frame this keypoint is found in. *location* is a pair giving the x- and
        y-co-ordinate of the keypoint in image space. *scale* is the scale of
        they keypoint and *descriptor* is the 168-component descriptor for the
        keypoint.

        """

        self.frame_idx = frame_idx
        self._kp_array = np.hstack((location, scale, descriptor))
        if self._kp_array.shape[0] != 171:
            raise ValueError('Location, scale and descriptor must total 171 values.')

    @property
    def location(self):
        return self._kp_array[:2]

    @property
    def scale(self):
        return self._kp_array[2]

    @property
    def descriptor(self):
        return self._kp_array[2:]

class Track(object):
    def __init__(self, kps=None):
        """Construct a track, optionally with a series of keypoints.

        If kps_and_frames is not None, it is a sequence of keypoints.

        """
        # We start with no initial frame index and no associated keypoints
        self.associated_keypoints = []

        # One state per frame. Each state is [x, y, x', y']
        self.states = []
        self._a_priori_states = []
        
        # One state covariance per frame.
        self.covariances = []
        self._a_priori_covariances = []

        for kp in kps:
            while self.initial_frame_idx is not None and self.final_frame_idx < kp.frame_idx - 1:
                self.update(self.final_frame_idx + 1)
            self.update(kp.frame_idx, kp)

    @property
    def initial_frame_idx(self):
        if len(self.associated_keypoints) == 0:
            return None
        return self.associated_keypoints[0].frame_idx

    @property
    def final_frame_idx(self):
        if len(self.associated_keypoints) == 0:
            return None
        return self.associated_keypoints[0].frame_idx + len(self.states) - 1
        
    def update(self, frame_idx, kp=None, obs_covariance=None):
        if self.initial_frame_idx is not None and frame_idx != self.final_frame_idx + 1:
            raise IndexError('Must update track for terminal frame')
            
        if len(self.states) > 0:
            a_posteriori_state = self.states[-1]
            a_posteriori_covariance = self.covariances[-1]
        else:
            a_posteriori_state = initial_state
            a_posteriori_covariance = initial_covariance
        
        F = state_evolution_mat
        H = state_observation_mat
        
        W = evolution_cov
        V = obs_covariance
        if V is None:
            if kp is not None:
                V = 0.25*kp.scale*kp.scale*np.eye(2)
            else:
                V = np.eye(1) # Doesn't really matter since it isn't used
            
        # Predict step
        a_priori_state = F.dot(a_posteriori_state)
        a_priori_covariance = F.dot(a_posteriori_covariance).dot(F.T) + W

        # Store predictions
        self._a_priori_states.append(a_priori_state)
        self._a_priori_covariances.append(a_priori_covariance)

        # Update
        if kp is not None:
            residual_measuement = np.array(kp.location) - H.dot(a_priori_state)
            residual_covariance = H.dot(a_priori_covariance).dot(H.T) + V
            kalman_gain = a_posteriori_covariance.dot(H.T).dot(np.linalg.inv(residual_covariance))
        
            a_posteriori_state = a_priori_state + kalman_gain.dot(residual_measuement)
            a_posteriori_covariance = (np.eye(kalman_gain.shape[0]) - kalman_gain.dot(H)).dot(a_priori_covariance)
            
            self.associated_keypoints.append(kp)
        else:
            a_posteriori_state = a_priori_state
            a_posteriori_covariance = a_posteriori_covariance
            
        # Record
        self.states.append(a_posteriori_state)
        self.covariances.append(a_posteriori_covariance)
   
    def predict_observation(self, frame_idx):
        """Predict a location for this track at a particular time index.

        Return predicted observation and prediction covariance.
        
        """
        if frame_idx < self.initial_frame_idx:
            raise IndexError('Frame index before track start')

        if self.initial_frame_idx is None:
            raise ValueError('Cannot predict in track with no initial frame')
        
        # If we're asked for a state we've already estimated, just return it
        state_idx = frame_idx - self.initial_frame_idx
        if state_idx < len(self.states):
            return (
                state_observation_mat.dot(self.states[state_idx]),
                state_observation_mat.dot(self.covariances[state_idx]).dot(state_observation_mat.T)
            )
        
        # Otherwise evolve forward in time
        n_forward = 1 + state_idx - len(self.states)
        predicted_state = np.copy(self.states[-1])
        predicted_cov = np.copy(self.covariances[-1])
        for iteration in xrange(n_forward):
            predicted_state = state_evolution_mat.dot(predicted_state)
            predicted_cov += evolution_cov
                
        return (
            state_observation_mat.dot(predicted_state),
            state_observation_mat.dot(predicted_cov).dot(state_observation_mat.T)
        )

    def backwards_smooth(self):
        """Return states and covariances from backwards smoothing track."""

        if len(self.associated_keypoints) == 0:
            return [], []

        n_states = 1 + self.associated_keypoints[-1].frame_idx - self.initial_frame_idx
        if n_states == 0:
            return [], []

        # The output array
        smoothed_states = [None,] * n_states
        smoothed_covars = [None,] * n_states

        # Start with final state/covariance
        smoothed_states[-1] = self.states[n_states-1]
        smoothed_covars[-1] = self.covariances[n_states-1]

        # Working backwards...
        for idx in xrange(n_states-1, 0, -1):
            smoothed_state_kp1 = smoothed_states[idx]
            smoothed_covar_kp1 = smoothed_covars[idx]

            a_posteriori_state = self.states[idx-1]
            a_posteriori_covariance = self.covariances[idx-1]

            a_priori_state = self._a_priori_states[idx]
            a_priori_covariance = self._a_priori_covariances[idx]

            C = a_posteriori_covariance.dot(state_evolution_mat.T).dot(np.linalg.inv(a_priori_covariance))
            smoothed_state = a_posteriori_state + C.dot(smoothed_state_kp1 - a_priori_state)
            smoothed_covar = a_posteriori_covariance + C.dot(smoothed_covar_kp1 - a_priori_covariance).dot(C.T)

            smoothed_states[idx-1] = smoothed_state
            smoothed_covars[idx-1] = smoothed_covar

        return smoothed_states, smoothed_covars

# Maximum time between keypoint observations. (This must never be <1 for obvious reasons.)
MAX_NO_KP_DELTA_T = 3


def _match_flann(before_pts, after_pts):
    prev_flann = pyflann.FLANN()
    
    prev_flann.build_index(before_pts)
    next_in_prev = prev_flann.nn_index(after_pts)[0]
    
    a_set = set(list(
         (p, n)
         for n, p in zip(xrange(after_pts.shape[0]), next_in_prev)
    ))
    
    next_flann = pyflann.FLANN()
    
    next_flann.build_index(after_pts)
    prev_in_next = next_flann.nn_index(before_pts)[0]
  
    b_set = set(list(
         (p, n)
         for p, n in zip(xrange(before_pts.shape[0]), prev_in_next)
    ))
    
    return np.array(list(a_set & b_set))

class Tracking(object):
    def __init__(self):
        self.tracks = []

    def add_frame(self, previous_frame, current_frame, frame_idx, keypoints):
        """Add a new frame to the tracking. The previous frame (if present)
        should be specified so that optical flow may be used to more accurately
        predict keypoint locations. Note that keypoint locations must be in
        image space. (That is, that the x co-ordinate must be in the range (0,
        ``current_frame.shape[1]``] and the y co-ordinate must be in the range
        (0, ``current_frame.shape[0]``].

        :param previous_frame: the frame preceeding *current_frame* in the video sequence
        :param current_frame: the current frame in the video sequence
        :param frame_idx: the 0-indexed index of the current frame
        :param keypoints: a sequence of :py:class:`Keypoint` instances for the current frame

        """
        # If this is the first frame, construct one track per key point and that's it...
        if self.tracks is None or len(self.tracks) == 0:
            self.tracks = list(Track([kp]) for kp in keypoints)
            return
            
        # ... otherwise

        if np.any(list(kp.frame_idx != frame_idx for kp in keypoints)):
            raise ValueError('All keypoints must have a frame index which matches frame_idx.')

        # Form a pair of (previous, current) frames
        frame_pair = (previous_frame, current_frame)

        # Extract keypoint locations and descriptors into a numpy array
        kp_locs = np.array(list(kp.location for kp in keypoints))
        kp_descs = np.array(list(kp.descriptor for kp in keypoints))
        
        # Use optical flow to predict where this frame's keypoints were in the previous frame
        
        # Optical flow
        oflow_kp_locs, _, _ = cv2.calcOpticalFlowPyrLK(frame_pair[1], frame_pair[0], kp_locs)
        
        # Get a set of predicted track locations for the previous frame from all tracks with their last keypoint within
        # MAX_NO_KP_DELTA_T frames
        candidate_tracks = list(t for t in self.tracks if t.associated_keypoints[-1].frame_idx >= frame_idx - MAX_NO_KP_DELTA_T)
        
        # Predict a set of keypoint locations (and covariances) from the candidate tracks
        predictions = list(t.predict_observation(frame_idx-1) for t in candidate_tracks)
        predicted_locs = np.array(list(p[0] for p in predictions), dtype=oflow_kp_locs.dtype)
        prev_descs = np.array(list(t.associated_keypoints[-1].descriptor for t in candidate_tracks), dtype=oflow_kp_locs.dtype)
        
        # Stack locations and descriptors together. The locations should be
        # close to zero for matching keypoints and so for overlapping spatial
        # keypoints we'll favour descriptor matching.
        before_kp_loc_descs = np.hstack((predicted_locs, prev_descs))
        after_kp_loc_descs = np.hstack((oflow_kp_locs, kp_descs))

        #matches = _match_flann(np.hstack((predicted_locs, desc_scale*prev_descs)), np.hstack((oflow_kp_locs,desc_scale*kp_descs)))
        matches = _match_flann(before_kp_loc_descs, after_kp_loc_descs)
        
        unmatched_after = np.ones(oflow_kp_locs.shape[0])
        unmatched_before = np.ones(predicted_locs.shape[0])
        for before, after in matches:
            # Where did we find it and where did we expect it?
            pred_loc = before_kp_loc_descs[before,:]
            after_loc = after_kp_loc_descs[after,:]
            kp = keypoints[after]

            # How far from predicted was it? We only accept ones within 2 x scale.
            delta_sq = np.sum(np.power(pred_loc - after_loc, 2))
            if delta_sq > 4.0*kp.scale*kp.scale:
                continue

            t = candidate_tracks[before]
            t.update(frame_idx, kp)
            
            unmatched_after[after] = 0
            unmatched_before[before] = 0

        for idx in np.nonzero(unmatched_after)[0]:
            self.tracks.append(Track([keypoints[idx]]))
        
        for idx in np.nonzero(unmatched_before)[0]:
            candidate_tracks[idx].update(frame_idx)
            

# vim:sw=4:sts=4:et
