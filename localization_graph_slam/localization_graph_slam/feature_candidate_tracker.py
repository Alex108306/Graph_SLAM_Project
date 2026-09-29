import numpy as np
from math import cos, sin
from .utils import warp_angle


class FeatureCandidateTracker:
    """
    Holds tentative line features (in the world frame) that have only been seen
    a few times. A candidate is promoted to a real map landmark once it has been
    observed at least ``min_observations`` times. Candidates that are not
    re-observed for ``max_frames_unseen`` consecutive frames are dropped.

    The tracker only *gates* features by count — it deliberately does NOT fuse
    observations across sightings. Each sighting comes from a different robot
    pose (with drift), so the world-frame projections are correlated through
    the shared pose history; naive fusion produces an "average" landmark that
    matches no individual lidar scan and over-confidently pins the GTSAM
    landmark away from the latest measurement. The job of optimally combining
    the measurements belongs to ISAM2 once the landmark is in the graph.
    """

    def __init__(self,
                 min_observations: int = 3,
                 match_rho_threshold: float = 0.35,
                 match_alpha_threshold: float = 0.25,
                 max_frames_unseen: int = 8):
        self.min_observations = int(min_observations)
        self.match_rho_threshold = float(match_rho_threshold)
        self.match_alpha_threshold = float(match_alpha_threshold)
        self.max_frames_unseen = int(max_frames_unseen)

        # Each entry: dict with world_polar, world_cov, line_world, count,
        # frames_unseen, last_obs_robot, last_obs_cov_robot.
        self._candidates = []

    @staticmethod
    def _observation_to_world_polar(xk, zfi):
        x_r, y_r, theta_r = xk[0][0], xk[1][0], xk[2][0]
        rho_obs, alpha_obs = float(zfi[0]), float(zfi[1])
        alpha_world = warp_angle(alpha_obs + theta_r)
        rho_world = rho_obs + cos(alpha_world) * x_r + sin(alpha_world) * y_r
        return np.array([rho_world, alpha_world])

    @staticmethod
    def _observation_cov_to_world(xk, zfi, Rfi, Pk):
        x_r, y_r, theta_r = xk[0][0], xk[1][0], xk[2][0]
        alpha_obs = float(zfi[1])
        a = alpha_obs + theta_r
        J_pose = np.array([
            [cos(a), sin(a), -sin(a) * x_r + cos(a) * y_r],
            [0.0,    0.0,    1.0],
        ])
        J_obs = np.array([
            [1.0, -x_r * sin(a) + y_r * cos(a)],
            [0.0,  1.0],
        ])
        cov_world = J_pose @ Pk @ J_pose.T + J_obs @ Rfi @ J_obs.T
        cov_world = 0.5 * (cov_world + cov_world.T) + 1e-9 * np.eye(2)
        return cov_world

    @staticmethod
    def _transform_segment_to_world(line_segment, xk):
        x_r, y_r, theta_r = xk[0][0], xk[1][0], xk[2][0]
        c, s = cos(theta_r), sin(theta_r)
        p0 = line_segment[0]
        p1 = line_segment[1]
        wp0 = [c * p0[0] - s * p0[1] + x_r, s * p0[0] + c * p0[1] + y_r]
        wp1 = [c * p1[0] - s * p1[1] + x_r, s * p1[0] + c * p1[1] + y_r]
        return [wp0, wp1]

    def _match_score(self, candidate_polar, world_polar):
        rho_err = abs(candidate_polar[0] - world_polar[0])
        alpha_err = abs(warp_angle(candidate_polar[1] - world_polar[1]))
        if rho_err > self.match_rho_threshold:
            return None
        if alpha_err > self.match_alpha_threshold:
            return None
        return (rho_err / max(self.match_rho_threshold, 1e-9)
                + alpha_err / max(self.match_alpha_threshold, 1e-9))

    def update(self, xk, Pk, zf_robot, Rf_robot, line_segments_robot):
        """
        Feed the unassociated observations into the candidate pool and return
        the subset that has reached the persistency threshold.

        On a re-observation we *replace* the candidate's stored world estimate
        with the latest one (no fusion) — see the class docstring for why.

        Returns five parallel lists, populated only for candidates promoted
        this frame:
          - world_polar       : [rho, alpha] in world frame (latest sighting)
          - world_cov         : 2x2 covariance in world frame (latest sighting)
          - line_world        : [[x0,y0],[x1,y1]] in world frame (latest sighting)
          - last_obs_robot    : [rho, alpha] in robot frame (latest sighting)
          - last_obs_cov_robot: 2x2 covariance in robot frame (latest sighting)
        """
        for candidate in self._candidates:
            candidate['matched_this_frame'] = False

        for zfi, Rfi, seg in zip(zf_robot, Rf_robot, line_segments_robot):
            world_polar = self._observation_to_world_polar(xk, zfi)
            world_cov = self._observation_cov_to_world(xk, zfi, Rfi, Pk)
            world_seg = self._transform_segment_to_world(seg, xk)

            best_idx = None
            best_score = np.inf
            for idx, candidate in enumerate(self._candidates):
                if candidate['matched_this_frame']:
                    continue
                score = self._match_score(candidate['world_polar'], world_polar)
                if score is None:
                    continue
                if score < best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                self._candidates.append({
                    'world_polar': [float(world_polar[0]), float(world_polar[1])],
                    'world_cov': world_cov,
                    'line_world': world_seg,
                    'count': 1,
                    'frames_unseen': 0,
                    'matched_this_frame': True,
                    'last_obs_robot': np.array([float(zfi[0]), float(zfi[1])]),
                    'last_obs_cov_robot': np.asarray(Rfi, dtype=float).copy(),
                })
                continue

            # Re-observation: bump the count and refresh stored estimates with
            # the latest sighting. No averaging — the latest pose is the most
            # corrected one available.
            candidate = self._candidates[best_idx]
            candidate['world_polar'] = [float(world_polar[0]), float(world_polar[1])]
            candidate['world_cov'] = world_cov
            candidate['line_world'] = world_seg
            candidate['count'] += 1
            candidate['frames_unseen'] = 0
            candidate['matched_this_frame'] = True
            candidate['last_obs_robot'] = np.array([float(zfi[0]), float(zfi[1])])
            candidate['last_obs_cov_robot'] = np.asarray(Rfi, dtype=float).copy()

        promoted_world_polar = []
        promoted_world_cov = []
        promoted_line_world = []
        promoted_last_obs_robot = []
        promoted_last_obs_cov_robot = []

        survivors = []
        for candidate in self._candidates:
            if not candidate['matched_this_frame']:
                candidate['frames_unseen'] += 1
                if candidate['frames_unseen'] > self.max_frames_unseen:
                    continue  # Stale candidate — drop it.

            if candidate['count'] >= self.min_observations and candidate['matched_this_frame']:
                # Only promote on a frame where we *just* saw the candidate, so
                # the stored "latest obs" lines up with the current pose key.
                promoted_world_polar.append(candidate['world_polar'])
                promoted_world_cov.append(candidate['world_cov'])
                promoted_line_world.append(candidate['line_world'])
                promoted_last_obs_robot.append(candidate['last_obs_robot'])
                promoted_last_obs_cov_robot.append(candidate['last_obs_cov_robot'])
                continue  # Promoted — drop from candidate pool.

            survivors.append(candidate)

        self._candidates = survivors
        for candidate in self._candidates:
            candidate.pop('matched_this_frame', None)

        return (promoted_world_polar, promoted_world_cov, promoted_line_world,
                promoted_last_obs_robot, promoted_last_obs_cov_robot)

    def candidate_count(self):
        return len(self._candidates)
