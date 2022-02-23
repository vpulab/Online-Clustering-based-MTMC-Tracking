'''
################################
#    Read Veri dataset and
#   Need to be executed with
#   the root folder only with
#   AICC data until id 325
################################
'''



import os
import cv2
import time
import numpy as np
from sklearn.model_selection import train_test_split
# importing shutil module
import shutil




if __name__ == '__main__':



    # Original dataset directory
    veri_train_root_dir = '/home/vpu/Datasets/VeRi_with_plate/image_train/'
    files_veri_train = os.listdir(veri_train_root_dir)
    num_files_train= files_veri_train.__len__()

    veri_test_root_dir = '/home/vpu/Datasets/VeRi_with_plate/image_test/'
    files_veri_test = os.listdir(veri_test_root_dir)
    num_files_test = files_veri_test.__len__()

    destination_dir = '/home/vpu/Datasets/AICC_veri/subset_train_aicc_veri'
    files_dest = os.listdir(destination_dir)
    max_num_id = np.max(((np.sort((np.asarray(files_dest))))).astype(int))
    next_id = max_num_id + 1


    # Theese 2 loops calculate the new correspondent ids od the veri vechicles
    # veri_id = [1   2    3   4   6 ] e.g.
    # new_id =  [326 327 328 329 330 ]

    # veri train

    ids_veri_train = []
    for i in range(num_files_train):
        name = files_veri_train[i]
        ids_veri_train.append(int(name[0:4]))

    ids_veri_train_unique = np.unique(np.asarray(ids_veri_train))
    num_ids_train_veri = ids_veri_train_unique.__len__()

    correspondent_id_train = np.asarray(range(num_ids_train_veri)) + next_id
    next_id = np.max(correspondent_id_train) + 1

    # veri test
    ids_veri_test = []
    for i in range(num_files_test):
        name = files_veri_test[i]
        ids_veri_test.append(int(name[0:4]))

    ids_veri_test_unique = np.unique(np.asarray(ids_veri_test))
    num_ids_test_veri = ids_veri_test_unique.__len__()

    correspondent_id_test = np.asarray(range(num_ids_test_veri)) + next_id


    for i in range(num_files_train):
        name = files_veri_train[i]
        veri_id = int(name[0:4])
        idx = np.where(ids_veri_train_unique == veri_id)[0]
        new_id = correspondent_id_train[idx]

        #check if exist the correspondent folder

        id_path = os.path.join(destination_dir,  '%04d' % new_id)
        if not os.path.exists(id_path):
            os.makedirs(id_path)

        im_src = os.path.join(veri_train_root_dir, name)
        im_dst = os.path.join(id_path, name)
        shutil.copyfile(im_src, im_dst)


    for i in range(num_files_test):
        name = files_veri_test[i]
        veri_id = int(name[0:4])
        idx = np.where(ids_veri_test_unique == veri_id)[0]
        new_id = correspondent_id_test[idx]

        # check if exist the correspondent folder

        id_path = os.path.join(destination_dir, '%04d' % new_id)
        if not os.path.exists(id_path):
            os.makedirs(id_path)

        im_src = os.path.join(veri_test_root_dir, name)
        im_dst = os.path.join(id_path, name)
        shutil.copyfile(im_src, im_dst)



