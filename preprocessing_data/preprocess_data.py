'''
################################
#    Pre-process AICC19 data   #
################################
'''

import os
import cv2
import time
import numpy as np


#offset = dict()
offset= {'S01': {'c001': 0,
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
                'c036': 0},
         'S06':{ 'c041': 0,
                 'c042': 0,
                 'c043': 0,
                 'c044': 0,
                 'c045': 0,
                 'c046': 0 }}

max_frame={'S01': 2110,
           'S02': 2110,
           'S03': 2422,
           'S04': 710,
           'S05': 4299,
           'S06': 100
           }


def  process(mode,offset):
    # Current root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Original dataset directory
    dataset_dir = os.path.join('./../datasets/AIC19', mode)

    scenarios = ['S02']
    for s in scenarios:

        cameras = os.listdir(os.path.join(dataset_dir, s))

        for c in cameras:

            tStart = time.time()
            print('Processing ' + s + ' ' + c + ' with offset = ' + str(offset[s][c]))

            vdo_dir = os.path.join(dataset_dir, s, c, 'vdo.avi')
            video = cv2.VideoCapture(vdo_dir)

            num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video.get(cv2.CAP_PROP_FPS)
            h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

            blank_image = np.zeros((h, w, 3), np.uint8)

            output_dir = os.path.join(dataset_dir, s, c, 'img')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            offset_frames = int(round((offset[s][c]) * fps))
            frame_counter = 1

            # If offset > 0 : fill with blank images at the begining of the sequence
            if offset_frames > 0:
                for f in range(1, offset_frames + 1):
                    frame_name = os.path.join(output_dir, str(frame_counter).zfill(6) + ".jpg")
                    cv2.imwrite(frame_name, blank_image)

                    frame_counter += 1

            # Read video file and save image frames
            while video.isOpened():

                ret, frame = video.read()
                frame_name = os.path.join(output_dir, str(frame_counter).zfill(6) + ".jpg")

                # print(video.get(cv2.CAP_PROP_POS_FRAMES))

                if not ret:
                    print("End of video file.")
                    a = 1
                    break
                cv2.imwrite(frame_name, frame)
                frame_counter += 1

            if frame_counter < max_frame[s]:
                # Fill at the end with black frames to reach max number of frames
                for f in range(frame_counter, max_frame[s] + 1):
                    frame_name = os.path.join(output_dir, str(frame_counter).zfill(6) + ".jpg")
                    cv2.imwrite(frame_name, blank_image)
                    frame_counter += 1

            tEnd = time.time()
            print("It cost %f sec" % (tEnd - tStart))




            


if __name__ == '__main__':

    mode = 'test/'
    process(mode,offset)




