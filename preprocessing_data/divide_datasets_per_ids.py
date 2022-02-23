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
from PIL import Image
import matplotlib.pyplot as plt

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


    # Current root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    params = dataset_parameters()
    mode = 'validation_S02'
    # Original dataset directory
    dataset_root_dir = os.path.join('/home/vpu/Datasets/AIC20', mode)
    subset_dir = '/home/vpu/Datasets/AIC20/S02_per_id'

    scenarios = os.listdir(dataset_root_dir)

    ids_all =[]

    for s in scenarios:

        cameras = os.listdir(os.path.join(dataset_root_dir, s))

        for c in cameras:

            tStart = time.time()
            print('Processing ' + s + ' ' + c + ' with offset = ' + str(params.offset[s][c]))

            gt_file = os.path.os.path.join(dataset_root_dir, s, c, 'gt/gt.txt')

            sync_gt_dir = os.path.os.path.join(dataset_root_dir, s, c, 'gt_sync')
            gt_file_sync = os.path.os.path.join(sync_gt_dir, 'gt.txt')


            # Sincronizar GT con los offsets
            if not os.path.exists(sync_gt_dir):

                os.makedirs(sync_gt_dir)
                f = open(gt_file, 'r+')
                gt_data = np.loadtxt(f, delimiter=',')

                offset_frames = int(round((params.offset[s][c]) * params.fps))
                sync_gt = np.copy(gt_data)


                sync_gt[:, 0] = sync_gt[:, 0] + offset_frames
                np.savetxt(gt_file_sync, sync_gt, delimiter=',',fmt='%d')
                gt_sync_data = np.copy(sync_gt)
                tEnd = time.time()
                print("It cost %f sec" % (tEnd - tStart))
            else:
                f = open(gt_file_sync, 'r+')
                gt_sync_data = np.loadtxt(f, delimiter=',')
                ids = gt_sync_data[:,1]
                ids_unique = np.unique(ids)
                ids_all = Union(list(ids_unique), list(ids_all))



            for f in range(1,  params.max_frame[s] + 1):

                bboxes = gt_sync_data[gt_sync_data[:, 0] == f, :]

                frame_img = Image.open(os.path.join(dataset_root_dir, s, c, 'img', '%06d.jpg' % f))
                for b in range(0,bboxes.__len__()):

                    box = frame_img.crop((bboxes[b,2], bboxes[b,3], bboxes[b,2]+ bboxes[b,4], bboxes[b,3]+bboxes[b,5]))
                    box_id = bboxes[b,1]
                    # plt.imshow(box)
                    # plt.show(block = False)
                    # plt.close()
                    path_box = os.path.join(subset_dir,'%04d' % (int(box_id)))
                    if not os.path.exists(path_box):
                        os.makedirs(path_box)
                    num_box = os.listdir(path_box).__len__()
                    box.save(os.path.join(path_box,  '%s_%04d.jpg' % (c , num_box)))




