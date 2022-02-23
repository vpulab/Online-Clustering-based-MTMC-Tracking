'''
################################
#         Dataset              #
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
from torchvision import transforms


class dataset():

    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def pad(self,img, value):

        h = img.height
        w = img.width
        index, v = max(enumerate([h, w]))
        max_value = [h, w][index]

        # amount of pixels being replicated at each direction
        if index == 0:  # h > w no need to pad in top and bottom
            top = 0
            bottom = 0
            left, right = int(round((h - w) / 2)), int(round((h - w) / 2))
        elif index == 1:  # w > h no need to pad in right and left
            left = 0
            right = 0
            top, bottom = int(round((w - h) / 2)), int(round((w - h) / 2)),

        padded = transforms.functional.pad(img,(left,top, right, bottom),value, 'constant')

        return padded

    def square(self, bbox_img, frame_img, x,y):

        h = bbox_img.height
        w = bbox_img.width

        # Center of the bbox
        cx = round(x + round(w / 2))
        cy = round(y + round(h / 2))

        # Square bbox

        square_size = max(w, h)
        sx = int(cx - square_size / 2)
        sy = int(cy - square_size / 2)

        square_bbox = transforms.functional.crop(frame_img, sy, sx, square_size, square_size)
        return  square_bbox