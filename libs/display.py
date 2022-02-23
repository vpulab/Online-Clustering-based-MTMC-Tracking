'''
################################
#         Display              #
################################
'''

import os
import sys
import cv2
import math
import time
import numpy as np
from PIL import Image
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class display():

    def __init__(self,flag):
        self.flag = flag

    def show_frame(self, img, c):
        if self.flag:
            plt.figure()
            plt.imshow(img)
            plt.title('Camera ' + str(c))

    def draw_bbox(self,x,y,w,h):
        if self.flag:
            plt.gca().add_patch(Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'))
