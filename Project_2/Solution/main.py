import sys
import cv2
import pyrealsense2 as rs
import numpy as np
import math
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import json
import time
import rtde_control
import rtde_receive
import UR_Commands
from camera_intel import Camera
from kinematics import ikSolver, fwSolver

"""
Notes:
    The command to run the script.
    $ ../> python3.8 main.py

    UR-RTDE Lib.
    https://sdurobotics.gitlab.io/ur_rtde/
"""

# ip_address = '192.168.0.10'

def main():
    ip_address = '192.168.0.10'
    cam = Camera()

    a = np.array([0, -0.24365, -0.21325, 0, 0, 0])
    d = np.array([0.1519, 0, 0, 0.11235, 0.08533, 0.0819])
    alpha = np.array([0, np.deg2rad(90), 0, 0, np.deg2rad(90), np.deg2rad(-90)]) 
    ik = ikSolver(a, d, alpha)
    fw = fwSolver()

    degrees = np.array([95.6, -77.08, 66.97, -100.55, -97.51, 5.88]) 
    middle_fly = degrees * (np.pi / 180) 

    degrees_pos = np.array([-8.75, -87.97, 93.6, -95.32, -91.98, 174.81]) 
    place_pos = degrees_pos * (np.pi / 180) 

    ctrl_cls = rtde_control.RTDEControlInterface(ip_address)
    UR_Commands.RG2_Control(ip_address, 100.0)
    del ctrl_cls
    ctrl_cls = rtde_control.RTDEControlInterface(ip_address)

    ctrl_cls.moveJ(place_pos, 0.4, 0.2)

    color_image = cam.get_and_save_pic()
    filtered_contours, center_coordinates, areas = cam.obj_detection(color_image)
    # seřazení podle velikostí
    zipped = list(zip(areas, filtered_contours, center_coordinates))
    zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
    areas, filtered_contours, center_coordinates = zip(*zipped)
    cons = cam.get_cons()
    contours_back, angle, new_center = cam.optimize_position(filtered_contours, center_coordinates, cons)
    transform_matrices, Ts, Rs = cam.get_coordinates(new_center,angle)
    transform_matrices = np.round(transform_matrices, 3)
    # print(Ts[0])
    # print(Rs[0])
    # Vykreslení kontur
    image_contours_new = color_image.copy()
    cv2.drawContours(image_contours_new, contours_back, -1, (0, 0, 255), 2)
    cv2.imwrite('detekce.jpg', image_contours_new)
    # cv2.imshow('Detekce', image_contours_new)
    # cv2.waitKey(0)
    
    for transformation in transform_matrices:

        qs = ik.solveIK(transformation)
        q = ik.nearestQ(qs, middle_fly)
        transform = fw.forward_kinematic_solution(fw.DH_matrix_UR3, q) 
        transform = np.round(transform, 3)
        print(transformation)

        ctrl_cls.moveJ(q, 0.4, 0.2)
        trans_2 = np.copy(transformation)
        trans_2[2,3] -= 0.055
        print(trans_2)

        qs = ik.solveIK(trans_2)
        q = ik.nearestQ(qs, middle_fly)
        ctrl_cls.moveJ(q, 0.1, 0.05)
        UR_Commands.RG2_Control(ip_address, 0.0)
        del ctrl_cls
        ctrl_cls = rtde_control.RTDEControlInterface(ip_address)

        trans_2 = np.copy(transformation)
        trans_2[2,3] += 0.070
        qs = ik.solveIK(trans_2)
        q = ik.nearestQ(qs, middle_fly)
        ctrl_cls.moveJ(q, 0.1, 0.05)

        ctrl_cls.moveJ(place_pos, 0.4, 0.2)
        UR_Commands.RG2_Control(ip_address, 100.0)
        del ctrl_cls
        ctrl_cls = rtde_control.RTDEControlInterface(ip_address)

        # ctrl_cls.moveL(np.concatenate((np.array([0.0, -300.0, 100.0], dtype=np.float32) * 0.001, 
        #                     np.deg2rad(np.array([-180.0, 0.0, 0.0], dtype=np.float32)))), 0.5, 0.25)




if __name__ == "__main__":
    sys.exit(main())