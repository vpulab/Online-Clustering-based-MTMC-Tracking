import os
import cv2
import time
import numpy as np
from sklearn.model_selection import train_test_split







if __name__ == '__main__':

    dir = '/home/vpu/Datasets/AICC_veri/subset_train_aicc_veri'

    list_files = os.listdir(dir)
    num_files= list_files.__len__()

    for i in range(num_files):
        new_name = '0' + list_files[i]
        src = os.path.join(dir,list_files[i])
        dst = os.path.join(dir,new_name)
        os.rename(src, dst)

