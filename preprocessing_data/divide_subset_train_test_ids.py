'''
################################
#    Pre-process AICC19 data   #
################################
'''

import os
import cv2
import time
import numpy as np
from sklearn.model_selection import train_test_split
# importing shutil module
import shutil

class dataset_parameters():
    def __init__(self):

        self.fps = 10
        #offset = dict()
        self.offset= {'S01': {'c001': 0,
                        'c002': 1.640,
                        'c003': 2.049,
                        'c004': 2.177,
                        'c005': 2.235},
                 'S02': {'c006': 0,
                        'c007': 0.061,
                        'c008': 0.421,
                        'c009': 0.660},
                 'S03': { 'c010': 8.715,
                         'c011': 8.457,
                         'c012': 5.879,
                         'c013': 0,
                         'c014': 5.042,
                         'c015':8.492},
                 'S04':{ 'c016': 0,
                        'c017': 14.318,
                        'c018': 29.955,
                        'c019': 26.979,
                        'c020': 25.905,
                        'c021': 39.973,
                        'c022': 49.422,
                        'c023': 45.716,
                        'c024': 50.853,
                        'c025': 50.263,
                        'c026': 70.450,
                        'c027': 85.097,
                        'c028': 100.110,
                        'c029': 125.788,
                        'c030': 124.319,
                        'c031': 125.033,
                        'c032': 125.199,
                        'c033': 150.893,
                        'c034': 140.218,
                        'c035': 165.568,
                        'c036': 170.797,
                        'c037': 170.567,
                        'c038': 175.426,
                        'c039': 175.644,
                        'c040': 175.838},
                 'S05':{ 'c010': 0,
                        'c016': 0,
                        'c017': 0,
                        'c018': 0,
                        'c019': 0,
                        'c020': 0,
                        'c021': 0,
                        'c022': 0,
                        'c023': 0,
                        'c024': 0,
                        'c025': 0,
                        'c026': 0,
                        'c027': 0,
                        'c028': 0,
                        'c029': 0,
                        'c033': 0,
                        'c034': 0,
                        'c035': 0,
                        'c036': 0}}

        self.max_frame={'S01': 2110,
                   'S02': 2110,
                   'S03': 2422,
                   'S04': 710,
                   'S05': 4299,
                   }

def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

if __name__ == '__main__':



    params = dataset_parameters()

    # Original dataset directory
    dataset_root_dir = '/home/vpu/Datasets/AIC20/'

    subset_dir = '/home/vpu/Datasets/AIC20/validation_per_id'
    ids = os.listdir(subset_dir)


    # subset_train_dir = os.path.join(dataset_root_dir, 'subset_train' )
    # train_txt = os.path.join(subset_train_dir, 'train.txt')
    # f_train = open(train_txt, 'w+')

    subset_test_dir = os.path.join(dataset_root_dir, 'validation_quarter')
    # test_txt = os.path.join(subset_test_dir, 'test.txt')
    # f_test = open(test_txt, 'w+')

    ids_train, ids_test = train_test_split(ids, test_size=0.25, shuffle=True, random_state=5)

    # for i_train in ids_train:
    #
    #     folder_src = os.path.join(subset_dir, i_train)
    #     folder_dst = os.path.join(subset_train_dir, i_train)
    #
    #     shutil.copytree(folder_src,folder_dst)

    for i_test in ids_test:

        folder_src = os.path.join(subset_dir, i_test)
        folder_dst = os.path.join(subset_test_dir, i_test)

        shutil.copytree(folder_src, folder_dst)



        # # print('Dividing id  ' + str(i) + ' into train and test sets')
        # num_images = os.listdir(os.path.join(subset_dir, i)).__len__()
        # images_all = np.array(range(0, num_images))
        # images_all_paths = os.listdir(os.path.join(subset_dir, i))
        #
        # for i_train in images_train:
        #     path_id_train = os.path.join(subset_train_dir,str(int(i)))
        #     if not os.path.exists(path_id_train):
        #         os.makedirs(path_id_train)
        #
        #     im_src = os.path.join(subset_dir, i, '%04d.jpg' % i_train)
        #     im_dst = os.path.join(subset_train_dir, i,  '%04d.jpg' % i_train)
        #     shutil.copyfile(im_src, im_dst)
        #
        #     head_tail = os.path.split(subset_train_dir)
        #     path_in_txt = os.path.join(head_tail[1], i, '%04d.jpg' % i_train)
        #
        #     f_train.write("%s\n" % path_in_txt)
        #


        #
        # for i_test in images_test:
        #     path_id_test = os.path.join(subset_test_dir, str(int(i)))
        #     if not os.path.exists(path_id_test):
        #         os.makedirs(path_id_test)
        #
        #     im_src = os.path.join(subset_dir, i, '%04d.jpg' % i_test)
        #     im_dst = os.path.join(subset_test_dir, i, '%04d.jpg' % i_test)
        #     shutil.copyfile(im_src, im_dst)
        #
        #     head_tail = os.path.split(subset_test_dir)
        #     path_in_txt = os.path.join(head_tail[1], i, '%04d.jpg' % i_test)
        #     f_test.write("%s\n" % path_in_txt)






            # tStart = time.time()
            # print('Processing ' + s + ' ' + c + ' with offset = ' + str(params.offset[s][c]))
            #
            # gt_file = os.path.os.path.join(dataset_root_dir, s, c, 'gt/gt.txt')
            #
            # sync_gt_dir = os.path.os.path.join(dataset_root_dir, s, c, 'gt_sync')
            # gt_file_sync = os.path.os.path.join(sync_gt_dir, 'gt.txt')






                    # frame_img.crop((1, 10, 100, 200))


    # ids_train, ids_test = train_test_split(ids_all, test_size=0.2, shuffle=True)
    # for i in ids_train:
    #     os.makedirs(os.path.join('/home/vpu/Datasets/AICC19/sub_train',str(int(i))))
    #
    #
    # for i in ids_test:
    #     os.makedirs(os.path.join('/home/vpu/Datasets/AICC19/sub_test',str(int(i))))
    a=1

