'''
################################
#         Camera               #
################################
'''

import os
import sys
import cv2
import math
import time
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.spatial.distance import cdist


class camera():

    def __init__(self,calibration_dir):
        self.homography_matrix = {}
        self.calibration_dir = calibration_dir

    def apply_homography_image_to_world(self,xi, yi, H_world_to_image):
        # Spatial vector xi, yi, 1
        S = np.array([xi, yi, 1]).reshape(3, 1)
        # Get 3 x 3 matrix and compute inverse
        H_world_to_image = np.array(H_world_to_image).reshape(3, 3)
        H_image_to_world = np.linalg.inv(H_world_to_image)
        # Dot product
        prj = np.dot(H_image_to_world, S)
        # Get world coordinates
        xw = (prj[0] / prj[2]).item() # latitude
        yw = (prj[1] / prj[2]).item() # longitude
        return xw, yw


    def apply_homography_world_to_image(self,xi, yi, H_world_to_image):
        # Spatial vector xi, yi, 1
        S = np.array([xi, yi, 1]).reshape(3, 1)
        # Get 3 x 3 matrix and compute inverse
        H_world_to_image = np.array(H_world_to_image).reshape(3, 3)

        # Dot product
        prj = np.dot(H_world_to_image, S)
        # Get world coordinates
        xw = (prj[0] / prj[2]).item() # latitude
        yw = (prj[1] / prj[2]).item() # longitude
        return xw, yw

    def load_homography_matrix(self,s,c):

        file = os.path.join(self.calibration_dir ,s, c,'calibration.txt')
        idf = open(file, 'r')
        line = idf.readline()
        line2 = idf.readline()
        idf.close
        s = line.replace(';', ' ')
        s = s[-(s.__len__() - line.find(':') - 1):-1]
        floats = [float(x) for x in s.split()]
        H = np.array(floats).reshape(3, 3)

        self.homography_matrix[c] = H