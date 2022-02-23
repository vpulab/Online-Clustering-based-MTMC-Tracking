'''
################################
#  Load Single Camera Tracking #
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
from misc import nms


class sct():

    def __init__(self, mtmc):
        self.data = {}
        self.detector = mtmc.detector
        self.dataset_root_dir = mtmc.dataset_root_dir
        self.fps = 10 # For AICC19

    def new_frame_data(self):
        struct={}
        struct['id_cam'] = []
        struct['id'] = []
        struct['x'] = []
        struct['y'] = []
        struct['w'] = []
        struct['h'] = []
        struct['xw'] = []
        struct['yw'] = []
        struct['features'] = []
        struct['bbox'] = []
        return struct

    def new_frame_w_data(self):
        struct={}
        struct['f'] = []
        struct['id_cam'] = []
        struct['id'] = []
        struct['x'] = []
        struct['y'] = []
        struct['w'] = []
        struct['h'] = []
        struct['xw'] = []
        struct['yw'] = []
        struct['features'] = []
        return struct


    def new(self,scene):
        self.data[scene] = {}

    def load(self,set,scene,offset, flag_filter_size,score_th):

        cameras = os.listdir(os.path.join(self.dataset_root_dir, set, scene))
        for c in cameras:
            # Directory
            dir_file_sct = os.path.join(self.dataset_root_dir, set, scene, c,  self.detector + '.txt')
            data = np.loadtxt(dir_file_sct, delimiter=',')

            #Filter detections by score
            data = data[data[:, 6] > score_th , :]


            if flag_filter_size == True:
                data = data[(data[:, 4] * data[:, 5]) > 2000, :]



            # Load ROI and filter sct before saving into class
            roi = cv2.imread(os.path.join(self.dataset_root_dir, set, scene, c,'roi.jpg'))
            assert roi is not None , " Cannot read ROI image "

            if data.size == 0:
                self.data[scene][c] = data
            else:
                data_sync = self.synchronize(data, scene, c, offset)
                self.data[scene][c] = self.filter_roi(data_sync,roi)


    def synchronize(self,data, scene, c, offset):
        # frame, id, x, y, w, h , 1, -1, -1, -1
        offset_frames = int(round((offset[scene][c]) * self.fps))
        data[:,0] = data[:,0] + offset_frames

        return data



    def filter_roi(self,data,roi):

        img_h, img_w = roi.shape[0], roi.shape[1]

        filtered_data =[]

        for line in data:

            #Get base point of the bbox
            # frame, id, x, y, w, h , 1, -1, -1, -1
            x = int(line[2])
            y = int(line[3])
            width = int(line[4])
            height = int(line[5])

            # Check if some value is wrong & Check bboxes lying outside
            if width <= 0 or height <= 0:
                continue

            if x < 0:

                line[2] = 0
                width = width - abs(x)
                line[4] = width
                x = 0

            if y < 0:
                line[3] = 0
                height = height - abs(y)
                line[5] = height
                y = 0


            if (x+width) > img_w:
                w_exceed = x + width - img_w
                line[4] = width - w_exceed
                width = width - w_exceed

            if (y+height) > img_h:
                h_exceed = y + height - img_h
                line[5] = height - h_exceed
                height = height - h_exceed

            # if x < 0.3 * img_w or x > 0.7 * img_w or y < 0.2 * img_h or y > 0.8 * img_h:
            #     continue

            # If base of the bbox is not in the ROI, not saving
            bx = x + round(width / 2)
            by = y + height

            if roi[by-1, bx-1, 0] == 0:
                continue

            filtered_data.append(line)

        return filtered_data


if __name__ == '__main__':

    sct()

    set = ['train', 'test']
    # for s in set:


