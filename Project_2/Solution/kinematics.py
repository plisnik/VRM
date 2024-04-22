import numpy as np
from numpy import linalg
from math import pi
from math import cos
from math import sin
from math import atan2
from math import acos
from math import sqrt
from math import asin
from scipy.spatial.transform.rotation import Rotation as R
 
class fwSolver():
    DH_matrix_UR3 = np.matrix([[0, pi / 2.0, 0.1519],
                            [-0.24365, 0, 0],
                            [-0.21325, 0, 0],
                            [0, pi / 2.0, 0.11235],
                            [0, -pi / 2.0, 0.08535],
                            [0, 0, 0.0819]])
 
    def mat_transtorm_DH(self, DH_matrix, n, edges=np.matrix([[0], [0], [0], [0], [0], [0]])):
        n = n - 1
        t_z_theta = np.matrix([[cos(edges[n]), -sin(edges[n]), 0, 0],
                            [sin(edges[n]), cos(edges[n]), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], copy=False)
        t_zd = np.matrix(np.identity(4), copy=False)
        t_zd[2, 3] = DH_matrix[n, 2]
        t_xa = np.matrix(np.identity(4), copy=False)
        t_xa[0, 3] = DH_matrix[n, 0]
        t_x_alpha = np.matrix([[1, 0, 0, 0],
                            [0, cos(DH_matrix[n, 1]), -sin(DH_matrix[n, 1]), 0],
                            [0, sin(DH_matrix[n, 1]), cos(DH_matrix[n, 1]), 0],
                            [0, 0, 0, 1]], copy=False)
        transform = t_z_theta * t_zd * t_xa * t_x_alpha
        return transform
 
 
    def forward_kinematic_solution(self, DH_matrix, edges=np.matrix([[0], [0], [0], [0], [0], [0]])):
        t01 = self.mat_transtorm_DH(DH_matrix, 1, edges)
        t12 = self.mat_transtorm_DH(DH_matrix, 2, edges)
        t23 = self.mat_transtorm_DH(DH_matrix, 3, edges)
        t34 = self.mat_transtorm_DH(DH_matrix, 4, edges)
        t45 = self.mat_transtorm_DH(DH_matrix, 5, edges)
        t56 = self.mat_transtorm_DH(DH_matrix, 6, edges)
        answer = t01 * t12 * t23 * t34 * t45 * t56
        return answer
 
class ikSolver():
    def __init__(self, a, d, alpha):
        # Initialize the DH parameters of the robot
        self.a = a
        self.d = d
        self.alpha = alpha
 
    def DHLink(self, alpha, a, d, angle):
        T = np.array([[np.cos(angle),                 -np.sin(angle),                0,              a],
                     [np.sin(angle) * np.cos(alpha), np.cos(angle) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                     [np.sin(angle) * np.sin(alpha), np.cos(angle) * np.sin(alpha), np.cos(alpha),  np.cos(alpha)*d],
                     [0,                             0,                             0,              1]])
        return T
    
    def create_Transformation_Matrix(self, position, orientation):

        """
 
        :param position: 3x1 numpy array of the position, xyz

        :param orientation: 3x1 numpy array of the orientation in euler angles xyz

        :return: homogenous transformation matrix

        """

        T = np.eye(4)

        rot = R.from_euler('xyz', orientation)

        T[0:3, 0:3] = rot.as_matrix()

        T[0:3, 3] = position

        return T
 
    def nearestQ(self, q_list, last_q):
        weights = np.array([6, 5, 4, 3, 2, 1])
        best_q = np.zeros(6)
        bestConfDist = np.inf
        for q in q_list:
            confDist = np.sum(((q - last_q) * weights)**2)
            if confDist < bestConfDist:
                bestConfDist = confDist
                best_q = q
        return np.asarray(best_q)
 
    def solveIK(self, T06, *last_q):
        theta = np.zeros([8,6])
 
        # ---------- Theta 1 ----------
        P05 = (T06 @ np.array([0,0,-self.d[5], 1]))[0:3]
        phi1 = np.arctan2(P05[1],P05[0])
        phi2 = np.array([np.arccos(self.d[3]/np.linalg.norm(P05[0:2])), -np.arccos(self.d[3]/np.linalg.norm(P05[0:2]))])
 
        for i in range(4):
            theta[i,0] = phi1 + phi2[0] + np.pi/2
            theta[i+4,0] = phi1 + phi2[1] + np.pi/2
 
        for i in range(8):
            if theta[i,0] <= np.pi:
                theta[i,0] += 2*np.pi
            if theta[i,0] > np.pi:
                theta[i,0] -= 2*np.pi
        
        # ---------- Theta 5 ----------
        P06 = T06[0:3,3]
        for i in range(8):
            theta[i,4] = np.arccos((P06[0]*np.sin(theta[i,0])-P06[1]*np.cos(theta[i,0])-self.d[3])/self.d[5])
            if np.isin(i, [2,3,6,7]):
                theta[i,4] = -theta[i,4]
 
        # ---------- Theta 6 ----------
        T60 = np.linalg.inv(T06)
        X60 = T60[0:3,0]
        Y60 = T60[0:3,1]
 
        for i in range(8):
            theta[i,5] = np.arctan2((-X60[1]*np.sin(theta[i,0])+Y60[1]*np.cos(theta[i,0]))/np.sin(theta[i,4]),
                                    ( X60[0]*np.sin(theta[i,0])-Y60[0]*np.cos(theta[i,0]))/np.sin(theta[i,4]))
 
        # ------- Theta 3 and 2 -------
 
        for i in range(8):
            T01 = self.DHLink(self.alpha[0],self.a[0],self.d[0], theta[i,0])
            T45 = self.DHLink(self.alpha[4],self.a[4],self.d[4], theta[i,4])
            T56 = self.DHLink(self.alpha[5],self.a[5],self.d[5], theta[i,5])
 
            T14 = np.linalg.inv(T01)@T06@np.linalg.inv(T45@T56)
            P14xz = np.array([T14[0,3], T14[2,3]])
 
            theta[i,2] = np.arccos((np.linalg.norm(P14xz)**2-self.a[1]**2-self.a[2]**2)/(2*self.a[1]*self.a[2]))
 
            if i % 2 != 0:
                theta[i,2] = -theta[i,2]
 
            theta[i,1] = np.arctan2(-P14xz[1], -P14xz[0]) - np.arcsin(-self.a[2]*np.sin(theta[i,2])/np.linalg.norm(P14xz))
 
        # ---------- Theta 4 ----------
 
        for i in range(8):
            T01 = self.DHLink(self.alpha[0],self.a[0],self.d[0], theta[i,0])
            T12 = self.DHLink(self.alpha[1],self.a[1],self.d[1], theta[i,1])
            T23 = self.DHLink(self.alpha[2],self.a[2],self.d[2], theta[i,2])
            T45 = self.DHLink(self.alpha[4],self.a[4],self.d[4], theta[i,4])
            T56 = self.DHLink(self.alpha[5],self.a[5],self.d[5], theta[i,5])
 
            T34 = np.linalg.inv(T01@T12@T23)@T06@np.linalg.inv(T45@T56)
 
            theta[i,3] = np.arctan2(T34[1,0], T34[0,0])
 
        if last_q:
            q = self.nearestQ(theta, last_q)
            return q, theta
        else:
            return theta