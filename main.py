'''
################################
#  Cluster-based MTMC tracking #
################################
'''

# Python modules
import os
import time
import numpy as np
from PIL import Image

import torch


from sklearn.metrics import pairwise_distances


# Own modules
from preprocessing_data import preprocess_data
from libs import camera, colors, display, dataset, features, sct, tracking, clustering
from network import resnet_elg
from network import net_id_classifier
import torchvision.transforms as transforms

import argparse
import yaml
from misc import nms


parser = argparse.ArgumentParser(description='Training classifier pair of cars')

parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path')


global CONFIG

# from torch.utils.serialization import load_lua

class mtmc():
    def __init__(self, dataset_dir, detector):
        self.dataset_root_dir = dataset_dir
        self.detector = detector

        self.max_frame = {'S01': 2132,
                          'S02': 2110,
                          'S03': 2422,
                          'S04': 710,
                          'S05': 4299,
                          }

        self.offset = {'S01': {'c001': 0,
                               'c002': 1.640,
                               'c003': 2.049,
                               'c004': 2.177,
                               'c005': 2.235},
                       'S02': {'c006': 0,
                               'c007': 0.061,
                               'c008': 0.421,
                               'c009': 0.660},
                       'S03': {'c010': 8.715,
                               'c011': 8.457,
                               'c012': 5.879,
                               'c013': 0,
                               'c014': 5.042,
                               'c015': 8.492},
                       'S04': {'c016': 0,
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
                       'S05': {'c010': 0,
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

        self.colors = colors.distinguishable_colors()

        self.preprocess_flag = False
        self.display = False
        self.dist_th = CONFIG['DIST_TH']
        self.global_tracks = list(list())

        self.global_tracks.append(list())

    # frame ,time, cam_id ,SCT_id ,latitude ,longitude,  start_x, start_y ,
    # end_x, end_y, start_time, end_time, left, top, width, heigth
    # def __init__(self, scene):


if __name__ == '__main__':

    # Decode CONFIG file information
    tic1 = time.time()
    args = parser.parse_args()
    CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))

    '''
    Train set: S01, S03, S04
    Test set: S02, S05
    '''
    dataset_dir = CONFIG['DATASET_PATH']
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    detector = CONFIG['DETECTOR']

    set = 'test'  # 'test' 'train'

    # Initialize global mtmc class
    mtmc = mtmc(dataset_dir, detector)

    # Inicialize cam class
    cam = camera.camera(os.path.join(mtmc.dataset_root_dir, set))

    # Dataset class
    aicc = dataset.dataset()

    # Display class
    display = display.display(mtmc.display)

    ### LOAD NET

    if CONFIG['MODEL'] == "Imagenet":
        # Features model pretrined
        net = resnet_elg.resnet50(pretrained=True)

    else:
        model = CONFIG['MODEL']
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/' + model)

        net = net_id_classifier.net_id_classifier('ResNet50', CONFIG['NUM_IDS'], CONFIG['SIZE_FC'])
        weights = torch.load(model_path)['state_dict']
        net.load_state_dict(weights, strict=True)

    net.cuda()
    net.eval()

    feat = features.features(aicc, net, CONFIG['MODE'])


    # # Tracking class
    track = tracking.tracking(mtmc, CONFIG)


    # Pre-processing needs to be executed only once after downloading the AICC19 dataset
    if mtmc.preprocess_flag:

        print('Preprocessing data from ' + set + 'set' + '\n')
        preprocess_data.process(set, mtmc.offset)

    # Load Single Camera Tracking data

    # Initialize sct strucure
    sct = sct.sct(mtmc)

    toc2 = time.time()
    print(toc2 - tic1, ' latency sec Elapsed')

    print('Loading SCT and homographies...')



    # For each scenario in the set
    for s in ['S02']: #change if proceed

        # Create new data dictionary in sct class
        sct.new(s)

        # Fill it with sct data: e.g.  sct.data[scene][camera] -> [ndarray]
        sct.load(set, s, mtmc.offset, flag_filter_size=CONFIG['FLAG_FILTER_SIZE'], score_th=CONFIG['SCORE_TH'])

        # Load homography matrices
        cameras = os.listdir(os.path.join(mtmc.dataset_root_dir, set, s))

        for c in cameras:
            cam.load_homography_matrix(s,c)

    print('Done.')

    # MTMC - Main Loop


    # Results file
    file_results = os.path.join(results_dir, s, CONFIG['ID'] + '.txt')
    f_id = open(file_results, 'w+')

    # Scenarios
    for s in ['S02']:

        cameras = os.listdir(os.path.join(mtmc.dataset_root_dir, set, s))
        cameras.sort()
        tic = time.time()

        # Frames
        for f in range(1,mtmc.max_frame[s] + 1): #mtmc.max_frame[s] + 1
            # print(['Frame ' + str(f)])
            mtmc.global_tracks.append(list())

            # Create empty dictionary for this frame sct
            sct_f = sct.new_frame_data()

            # Cameras
            for c in cameras:

                print('Processing ' + str(s) + ' frame ' + str(f) + ' camera ' + str(c))

                frame_img = Image.open(os.path.join(mtmc.dataset_root_dir, set, s, c, 'img', '%06d.jpg' % f))
                # display.show_frame(frame_img,c)

                sct_array = np.array(sct.data[s][c])
                sct_f_data = sct_array[sct_array[:, 0] == f, :]

                #NMS
                if CONFIG['NMS'] == True:
                    if sct_f_data.shape[0] != 0:
                        sct_f_data = nms.non_max_suppression(sct_f_data, sct_f_data[:, 6])


                # Fill sct_f dictionary with current frame information
                for i in range(sct_f_data.shape[0]):
                    sct_f['id_cam'].append(int(c[-3:]))
                    sct_f['id'].append(int(sct_f_data[i][1]))

                    x = int(round(sct_f_data[i][2]))
                    y = int(round(sct_f_data[i][3]))
                    w = int(round(sct_f_data[i][4]))
                    h = int(round(sct_f_data[i][5]))
                    sct_f['x'].append(x)
                    sct_f['y'].append(y)
                    sct_f['w'].append(w)
                    sct_f['h'].append(h)

                    # draw bbox
                    #display.draw_bbox(x, y, w, h)

                    # Crop bbox
                    bbox_img = transforms.functional.crop(frame_img, y, x, h, w)

                    # Get a square bbox to not to change the aspect ratio
                    # square_bbox = aicc.square(bbox_img,frame_img, x, y)
                    # bbox_padded = aicc.pad(bbox_img, (0, 0, 0))

                    bbox_img_norm = aicc.data_transform((bbox_img))
                    sct_f['bbox'].append(bbox_img_norm)

                    # Base of the bounding box to projection
                    bx = round(x + round(w / 2))
                    by = round(y + h)
                    xw, yw = cam.apply_homography_image_to_world(bx, by, cam.homography_matrix[c])
                    sct_f['xw'].append(xw)
                    sct_f['yw'].append(-yw)  # IMPORTANT: changed sign to positive coordinate

                    # Feature extraction

                    # plt.figure()
                    # plt.imshow(bbox_padded)

                    features_np = feat.extract(bbox_img_norm)
                    sct_f['features'].append(features_np)


            num_det_f = sct_f['id_cam'].__len__()

            if num_det_f != 0:



                # Clustering mode

                # Spatial distance
                xy = np.transpose(np.stack((np.array(sct_f['xw']), np.array(sct_f['yw'])), axis=0))

                dist_spatial = pairwise_distances(xy, xy, metric='euclidean')  # dist2 = pdist(xy,metric= metric)  #euclidean cosine cityblock

                # Set diagonal to 1 to avoid zeros
                dist_spatial = dist_spatial + (np.eye(dist_spatial.shape[0]))

                # Flag matrix with 1 when sct detections are closer than threshold
                dist_flag = (dist_spatial < mtmc.dist_th) * 1

                # norm = normalize(dist, norm='l2', axis = 0, copy = True, return_norm = False)

                #  Initialize clustering class. New clusters structure each frame
                clust = clustering.clustering(mtmc)

                # If there are some close detections and more than 1 camera
                if (sum(sum(dist_flag)) != 0) and ((np.unique(sct_f['id_cam'])).size > 1):

                    # Perform clustering using features

                    features_all = np.array(sct_f['features'])
                    dist_features = pairwise_distances(features_all, features_all, metric='euclidean')

                    #
                    if feat.characteristic == 'distance':
                        restricted_dist_features, association_matrix = feat.apply_restrictions(dist_spatial,
                                                                                                      dist_spatial,
                                                                                                      sct_f,
                                                                                                      mtmc.dist_th,
                                                                                                      feat.characteristic)
                        idx, optimal_clusters = clust.compute_clusters(restricted_dist_features, association_matrix)


                    elif feat.characteristic == 'appearance':
                        restricted_dist_features, association_matrix = feat.apply_restrictions(
                            dist_features, dist_spatial, sct_f, mtmc.dist_th, feat.characteristic)

                        idx, optimal_clusters = clust.compute_clusters(restricted_dist_features, association_matrix)

                    else:

                        # Clustering
                        restricted_dist_features, association_matrix = feat.apply_restrictions(
                            dist_features, dist_spatial, sct_f, mtmc.dist_th, feat.characteristic)
                        idx, optimal_clusters = clust.compute_clusters(restricted_dist_features, association_matrix)



                else:  # All detections are alone, no need to cluster

                    optimal_clusters = num_det_f
                    idx = np.array(range(0, optimal_clusters))
                    association_matrix = np.array([])
                    dist_features = []


                for cl in range(optimal_clusters):

                    # Initialize empty structure of the cluster
                    clust.clusters_frame.append(clust.new_cluster())

                    # Extract  detection in each    cluster
                    det_in_cluster = np.where(idx == cl)[0]

                    # Plot detections in cluster
                    # clust.display_detections_cluster(sct_f,det_in_cluster,cl)

                    # Get centroid of the cluster, mean position of every detectionin the cluster
                    mean_xw = np.mean((np.array(sct_f['xw']))[det_in_cluster])
                    mean_yw = np.mean((np.array(sct_f['yw']))[det_in_cluster])

                    clust.clusters_frame[-1]['xw'] = mean_xw
                    clust.clusters_frame[-1]['yw'] = mean_yw

                    # Plot centroid
                    # clust.display_centroid_cluster(mean_xw, mean_yw, cl)

                    for d in range(det_in_cluster.__len__()):
                        idx_det = det_in_cluster[d]
                        clust.clusters_frame[-1]['det'].append(clust.new_detection())
                        new_w = round(sct_f['w'][idx_det] + sct_f['w'][idx_det] * 0)
                        new_h = round(sct_f['h'][idx_det] + sct_f['h'][idx_det] * 0)
                        # c_x = sct_f['x'][idx_det] + round(sct_f['w'][idx_det] / 2
                        # c_y = sct_f['y'][idx_det] + round(sct_f['h'][idx_det] / 2 )
                        clust.clusters_frame[-1]['det'][-1]['x'] = sct_f['x'][idx_det] + round(sct_f['w'][idx_det] / 2 ) - round(new_w / 2)
                        clust.clusters_frame[-1]['det'][-1]['y'] = sct_f['y'][idx_det] + round(sct_f['h'][idx_det] / 2 ) - round(new_h / 2)
                        clust.clusters_frame[-1]['det'][-1]['w'] = new_w
                        clust.clusters_frame[-1]['det'][-1]['h'] = new_h
                        clust.clusters_frame[-1]['det'][-1]['id_cam'] = sct_f['id_cam'][idx_det]
                        clust.clusters_frame[-1]['det'][-1]['id_global'] = int(idx_det)


                        # clust.clusters_frame[-1]['det'][-1]['features'] = sct_f['features'][idx_det]



            # CLUSTERS - TRACKS   ASSOCIATION

            track.predict_new_locations()

            track.cluster_track_assignment(clust.clusters_frame, 1)

            # Update each assigned track with the corresponding detection.It calls the correct method of vision.KalmanFilter to correct the location estimate.
            #  Next, it stores the new bounding box, and increases the age of the track and the total  visible count by 1.
            #  Finally, the function sets the invisible count to 0.

            track.update_assigned_tracks(clust.clusters_frame)

            #Mark each unassigned track as invisible and increase its age by 1

            track.update_unassigned_tracks()

            # Delete tracks that have been invisible for too many frames

            track.delete_lost_tracks()

            track.check_unassigned_clusters(clust.clusters_frame, association_matrix, dist_features, dist_spatial)

            # Create new tracks from unassigned detections. Assume that any unassigned detection is a start of a new track.
            # In practice you can use other cues to eliminate nnoisy detections such as size, location, or appearance


            track.create_new_tracks_KF(clust.clusters_frame)

            track.save_global_tracking_data(clust.clusters_frame,f,mtmc.global_tracks,cam)


            # WRITTING RESULTS

            if track.updated_flag:

                num_tracks_f = mtmc.global_tracks[f].__len__()
                for i in range(num_tracks_f):

                    for det in range(mtmc.global_tracks[f][i]['det'].__len__()):

                        new_w = round(mtmc.global_tracks[f][i]['det'][det]['w'] + mtmc.global_tracks[f][i]['det'][det]['w']* CONFIG['AUG_SIZE'])
                        new_h = round(mtmc.global_tracks[f][i]['det'][det]['h'] + mtmc.global_tracks[f][i]['det'][det]['h']* CONFIG['AUG_SIZE'])


                        arg1 = mtmc.global_tracks[f][i]['det'][det]['id_cam']
                        arg2 = mtmc.global_tracks[f][i]['id']
                        arg3 = f
                        arg4 = mtmc.global_tracks[f][i]['det'][det]['x'] + round(mtmc.global_tracks[f][i]['det'][det]['w'] / 2) - round(new_w / 2)
                        arg5 = mtmc.global_tracks[f][i]['det'][det]['y'] + round(mtmc.global_tracks[f][i]['det'][det]['h'] / 2) - round(new_h / 2)
                        arg6 = new_w
                        arg7 = new_h


                        f_id.write("%d %d %d %d %d %d %d -1 -1\n" % (arg1, arg2, arg3, arg4, arg5, arg6, arg7))

        f_id.close()


        toc = time.time()
        print(toc - tic, 'sec Elapsed total time' )



