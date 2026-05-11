import numpy as np
import scipy
from math import cos, sin, atan2
from .utils import warp_angle

class DataAssociation:

    def __init__(self, confidenve_level, map_feature, map_feature_cov, line_segments_map):

        self.confidenve_level = confidenve_level
        self.map_feature = map_feature
        self.map_feature_cov = map_feature_cov
        self.map_line_segments = line_segments_map  # Initialize an empty list for line segments
        self.nf = len(map_feature)
        self.zfi_dim = 2

    
    def SquaredMahalanobisDistance(self, hfj, Pfj, zfi, Rfi):
        """
        Computes the squared Mahalanobis distance between the expected feature observation :math:`hf_j` and the feature observation :math:`z_{f_i}`.

        :param hfj: expected feature observation
        :param Pfj: expected feature observation covariance
        :param zfi: feature observation
        :param Rfi: feature observation covariance
        :return: Squared Mahalanobis distance between the expected feature observation :math:`hf_j` and the feature observation :math:`z_{f_i}`
        """
        zfi = np.array(zfi).reshape(-1, 1)
        innovation = zfi - hfj
        innovation[1, 0] = warp_angle(innovation[1, 0])  # wrap angle component
        S = Pfj + Rfi
        D2 = innovation.T @ np.linalg.inv(S) @ innovation
        return D2

    def IndividualCompatibility(self, D2_ij, dof, alpha):
        """
        Computes the individual compatibility test for the squared Mahalanobis distance :math:`D^2_{ij}`. The test is performed using the Chi-Square distribution with :math:`dof` degrees of freedom and a significance level :math:`\\alpha`.

        :param D2_ij: squared Mahalanobis distance
        :param dof: number of degrees of freedom
        :param alpha: confidence level
        :return: bolean value indicating if the Mahalanobis distance is smaller than the threshold defined by the confidence level
        """

        isCompatible = False 
        if D2_ij <= scipy.stats.chi2.ppf(alpha, dof):
            isCompatible = True

        return isCompatible
    
    def ICNN(self, hf, Phf, zf, Rf, dim):
        """
        Individual Compatibility Nearest Neighbor (ICNN) data association algorithm. Given a set of expected feature
        observations :math:`h_f` and a set of feature observations :math:`z_f`, the algorithm returns a pairing hypothesis
        :math:`H` that associates each feature observation :math:`z_{f_i}` with the expected feature observation
        :math:`h_{f_j}` that minimizes the Mahalanobis distance :math:`D^2_{ij}`.

        :param hf: vector of expected feature observations
        :param Phf: Covariance matrix of the expected feature observations
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :param dim: feature dimensionality
        :return: The vector of asociation hypothesis H
        """

        H = []
        for i in range(len(zf)):
            zfi= zf[i]
            Rfi = Rf[i]
            nearest_feature = None
            D2_min = np.inf
            for j in range(len(hf)):
                hfi = hf[j]
                Phfi = Phf[j]
                D2_ij = self.SquaredMahalanobisDistance(hfi, Phfi, zfi, Rfi)
                if self.IndividualCompatibility(D2_ij, dim, self.confidenve_level) and D2_ij < D2_min:
                    nearest_feature = j
                    D2_min = D2_ij

            H.append(nearest_feature)

        return H
    
    def DataAssociation(self, xk, Pk, zf, Rf):
        """
        Data association algorithm. Given state vector (:math:`x_k` and :math:`P_k`) including the robot pose and a set of feature observations
        :math:`z_f` and its covariance matrices :math:`R_f`,  the algorithm  computes the expected feature
        observations :math:`h_f` and its covariance matrices :math:`P_f`. Then it calls an association algorithms like
        :meth:`ICNN` (JCBB, etc.) to build a pairing hypothesis associating the observed features :math:`z_f`
        with the expected features observations :math:`h_f`.

        The vector of association hypothesis :math:`H` is stored in the :attr:`H` attribute and its dimension is the
        number of observed features within :math:`z_f`. Given the :math:`j^{th}` feature observation :math:`z_{f_j}`, *self.H[j]=i*
        means that :math:`z_{f_j}` has been associated with the :math:`i^{th}` feature . If *self.H[j]=None* means that :math:`z_{f_j}`
        has not been associated either because it is a new observed feature or because it is an outlier.

        :param xk: mean state vector including the robot pose
        :param Pk: covariance matrix of the state vector
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :return: The vector of asociation hypothesis H
        """

        h_F = []
        P_F = []
        for i in range(0, self.nf):
            h_Fi = self.hfj(xk, i)
            Jx = self.Jhfjx(xk, i)
            Jf = self.Jhfjf(xk, i)
            Pf_map = self.map_feature_cov[i]

            P_Fi = Jx @ Pk @ Jx.T + Jf @ Pf_map @ Jf.T
            P_Fi = 0.5 * (P_Fi + P_Fi.T) + 1e-9 * np.eye(2)
            # P_Fi = self.Jhfjx(xk, i) @ Pk @ self.Jhfjx(xk ,i).T
            h_F.append(h_Fi)
            P_F.append(P_Fi)

        H = self.ICNN(h_F, P_F, zf, Rf, self.zfi_dim)

        return H
    
    def hfj(self, xk, j):
        """
        Computes the expected feature observation :math:`h_{f_j}` for the :math:`j^{th}` feature in the map given the state vector :math:`x_k`.
        :param xk: mean state vector including the robot pose
        :param j: index of the feature in the map
        :return: expected feature observation :math:`h_{f_j}` (In polar coordinates)
        """

        # Implement sensor model
        range_f = self.map_feature[j][0]
        theta_f = self.map_feature[j][1]
        x_robot, y_robot, theta_robot = xk[0][0], xk[1][0], xk[2][0]
        range_obs = range_f - cos(theta_f) * x_robot - sin(theta_f) * y_robot
        theta_obs = warp_angle(theta_f - theta_robot)
        return np.array([[range_obs], [theta_obs]])


    def Jhfjx(self, xk, j):
        """
        Computes the Jacobian matrix of the expected feature observation :math:`h_{f_j}` with respect to the state vector :math:`x_k` for the :math:`j^{th}` feature in the map.

        :param xk: mean state vector including the robot pose
        :param j: index of the feature in the map
        :return: Jacobian matrix of the expected feature observation :math:`h_{f_j}` with respect to the state vector :math:`x_k`
        """
        # Implement Jacobian of the sensor model
        range_f = self.map_feature[j][0]
        theta_f = self.map_feature[j][1]
        x_robot, y_robot, theta_robot = xk[0][0], xk[1][0], xk[2][0]
        Jhfjx = np.array([[-cos(theta_f), -sin(theta_f), 0],
                          [0, 0, -1]])

        return Jhfjx
    
    def Jhfjf(self, xk, j):
        rho_f = self.map_feature[j][0]
        alpha_f = self.map_feature[j][1]
        x_robot, y_robot, _ = xk[0][0], xk[1][0], xk[2][0]

        return np.array([
            [1.0, sin(alpha_f) * x_robot - cos(alpha_f) * y_robot],
            [0.0, 1.0]
        ])
    
    def TransformLineSegmentToWorldFrame(self, line_segment, xk):
        """
        Transforms a line segment from the robot local frame to the world frame given the state vector :math:`x_k`.

        :param line_segment: line segment in the robot local frame
        :param xk: mean state vector including the robot pose
        :return: line segment in the world frame
        """
        line_segment = np.array([[line_segment[0][0], line_segment[1][0]],
                                 [line_segment[0][1], line_segment[1][1]]])  # Reshape to 2x2 array
        x_robot, y_robot, theta_robot = xk[0][0], xk[1][0], xk[2][0]
        R = np.array([[cos(theta_robot), -sin(theta_robot)], [sin(theta_robot), cos(theta_robot)]])
        line_segment_world = R @ line_segment + np.array([[x_robot], [y_robot]])
        line_segment_world = [[line_segment_world[0, 0], line_segment_world[1, 0]], [line_segment_world[0, 1], line_segment_world[1, 1]]]  # Reshape back to list of tuples
        return line_segment_world  # Flatten to a 1D list and convert to Python list
    
    def AddNewFeature(self, xk, Pk, zfi, Rfi, line_segment):
        """
        Adds a new feature to the map given a feature observation :math:`z_{f_i}` and its covariance matrix :math:`R_{f_i}`. The new feature is added to the map if it has not been associated with any expected feature observation :math:`h_{f_j}`.

        :param xk: mean state vector including the robot pose
        :param Pk: covariance matrix of the state vector
        :param zfi: feature observation
        :param Rfi: covariance matrix of the feature observation
        :param line_segment: line segment corresponding to the feature observation
        """
        
        # Implement inverted sensor model
        x_robot, y_robot, theta_robot = xk[0][0], xk[1][0], xk[2][0]
        range_obs = zfi[0]
        theta_obs = zfi[1]
        range_f = range_obs + cos(theta_obs + theta_robot) * x_robot + sin(theta_obs + theta_robot) * y_robot
        theta_f = warp_angle(theta_obs + theta_robot)
        line_segment_world = self.TransformLineSegmentToWorldFrame(line_segment, xk)

        J1 = np.array([
            [cos(theta_obs + theta_robot), sin(theta_obs + theta_robot), -sin(theta_obs + theta_robot) * x_robot + cos(theta_obs + theta_robot) * y_robot],
            [0, 0, 1]
        ])
        J2 = np.array([
            [1, - x_robot * sin(theta_obs + theta_robot) + y_robot * cos(theta_obs + theta_robot)],
            [0, 1]
        ])

        cov_feature_map = J1 @ Pk @ J1.T + J2 @ Rfi @ J2.T
        self.map_feature.append([range_f, theta_f])
        self.map_feature_cov.append(cov_feature_map)
        self.map_line_segments.append(line_segment_world)
    
    def AddmultipleNewFeatures(self, xk, Pk, zf, Rf, new_line_segments):
        """
        Adds multiple new features to the map given a set of feature observations :math:`z_f` and their covariance matrices :math:`R_f`. The new features are added to the map if they have not been associated with any expected feature observation :math:`h_{f_j}`.

        :param xk: mean state vector including the robot pose
        :param Pk: covariance matrix of the state vector
        :param zf: vector of feature observations unassociated with any expected feature observation
        :param Rf: Covariance matrix of the feature observations unassociated with any expected feature observation
        :param new_line_segments: vector of line segments corresponding to the new features
        """
        
        for i in range(len(zf)):
            self.AddNewFeature(xk, Pk, zf[i], Rf[i], new_line_segments[i])
        
        return self.map_feature, self.map_feature_cov, self.map_line_segments

    def GetUnassociatedFeatures(self, line_segments, zf, Rf, H):
        """
        Returns the vector of feature observations :math:`z_f` that have not been associated with any expected feature observation :math:`h_{f_j}` given the vector of association hypothesis :math:`H`.

        :param line_segments: vector of line segments extracted from the LiDAR scan
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :param H: vector of association hypothesis
        :return: vector of feature observations unassociated with any expected feature observation, their covariance matrices, and the corresponding line segments
        """

        unassociated_features = []
        unassociated_features_cov = []
        unassociated_line_segments = []
        for i in range(len(zf)):
            if H[i] is None:
                unassociated_features.append(zf[i])
                unassociated_features_cov.append(Rf[i])
                unassociated_line_segments.append(line_segments[i])

        return unassociated_features, unassociated_features_cov, unassociated_line_segments
