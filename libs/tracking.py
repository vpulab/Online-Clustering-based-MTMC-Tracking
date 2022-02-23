'''
################################
#         Spatial
              Tracking       #
################################
'''


import numpy as np
import numpy.matlib
import math

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from thirdparty import bbox
import torch

from scipy.cluster import hierarchy
from thirdparty import sklearn_dunn
import matplotlib.pyplot as plt
from munkres import Munkres, print_matrix, make_cost_matrix, DISALLOWED



class tracking():

    def __init__(self, mtmc,CONFIG):
        self.tracks_KF = list()
        self.id_track = 1
        self.unmatched_tracks, self.unmatched_clusters = [], []
        self.matches = None
        self.updated_flag = 0
        self.CONFIG = CONFIG

    def new_track(self, id, centroid, kalman, cluster_id ):
        track = {
            'id': id,
            'xw': centroid[0],
            'yw': centroid[1],
            'kalmanFilter': kalman,
            'age': 1,
            'state': 0,
            'totalVisibleCount': 1,
            'consecutiveInvisibleCount': 0,
            'fromcluster': np.asarray([cluster_id])
             }

        return track

    def new_global_track(self):

        global_track = {
            'id': [],

            'xw': [],
            'yw': [],
            'det': []}

        return global_track


    def create_new_tracks_KF(self,clusters):


        centroids_xw = [clusters[item]['xw'] for item in self.unmatched_clusters]
        centroids_yw = [clusters[item]['yw'] for item in self.unmatched_clusters]

        num_centroids = centroids_xw.__len__()
        centroids = np.zeros((num_centroids, 2))
        centroids[:, 0] = centroids_xw
        centroids[:, 1] = centroids_yw

        for i in (range(self.unmatched_clusters.__len__())): #

            # idx = self.unmatched_clusters[i]
            centroid = centroids[i,:]

            # Create Kalman Filter object
            kalman_filter = KalmanFilter(dim_x=4, dim_z=2)

            dt = 1.  # time step 1 second

            # State transition matrix
            kalman_filter.F = np.array([[1, dt, 0, 0],
                                   [0, 1, 0, 0],
                                  [0, 0, 1, dt],
                                  [0, 0, 0, 1]])


            # Assuming noise is discrete, and constant. noise in x and y are independent, so the covariance should be zero
            q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
            kalman_filter.Q = block_diag(q, q)

            # initial value for the state (position and velocity)
            kalman_filter.x = np.array([[centroid[0], 0, centroid[1], 0]]).T #CENTROIDE

            # Measurement function
            kalman_filter.H = np.array([[1 , 0, 0, 0],
                                        [0, 0, 1 , 0]])

            # Measurement noise
            kalman_filter.R = np.array([[5, 0],
                                         [0, 5]])


            # Covariance Matrix
            kalman_filter.P *= 500.


            self.tracks_KF.append(self.new_track(self.id_track,centroid, kalman_filter,self.unmatched_clusters[i]))

            self.id_track =  self.id_track + 1



    def predict_new_locations(self):

        for i in (range(self.tracks_KF.__len__())):

            # Predict the next location of the track
            self.tracks_KF[i]['kalmanFilter'].predict()

            # update centroid

            self.tracks_KF[i]['xw'] = self.tracks_KF[i]['kalmanFilter'].x[0]
            self.tracks_KF[i]['yw'] = self.tracks_KF[i]['kalmanFilter'].x[2]


    def  assign_detections_to_tracks(self, cost,clusters, pos_track_HA, ids_track_HA,tracks_id, pos_track_to_remove, pos_cluster_HA, ids_cluster_HA, clusters_id, pos_cluster_to_remove ):

        self.unmatched_tracks, self.unmatched_clusters = [], []

        # Hungarian algorithm (also known as Munkres algorithm)


        self.matches = linear_sum_assignment(cost)
        self.matches = list(self.matches)
        tracks_unassigned_HA = []
        clusters_unassigned_HA = []
        # check unassigmnents during Hungarian Algorithm

        for t in pos_track_HA:
            if (t not in  self.matches[0]):
                tracks_unassigned_HA.append(np.int64(t))

        for c in pos_cluster_HA:
            if (c not in self.matches[1]):
                clusters_unassigned_HA.append(np.int64(c))



        #  Update unassigments with filtered clusters

        clusters_assigned_HA = self.matches[1]
        ids_clusters_assigned_HA = ids_cluster_HA[clusters_assigned_HA]

        real_pos_clusters_matched = np.squeeze(np.array([np.where(clusters_id == id)[0] for id in ids_clusters_assigned_HA]))
        self.matches[1] = real_pos_clusters_matched

        ids_clusters_unassigned_HA = ids_cluster_HA[clusters_unassigned_HA]
        real_pos_clusters_unmatched = ([np.where(clusters_id == id)[0] for id in ids_clusters_unassigned_HA])
        if real_pos_clusters_unmatched.__len__() == 0:
            self.unmatched_clusters.clear()

        else:
            self.unmatched_clusters.clear()

            for i in real_pos_clusters_unmatched:
                self.unmatched_clusters.append(np.int64(i[0]))


        #  Update unassigments with filtered tracks
        tracks_assigned_HA = self.matches[0]
        ids_tracks_assigned_HA = ids_track_HA[tracks_assigned_HA]

        real_pos_tracks_matched = np.squeeze(np.array([np.where(tracks_id == id)[0] for id in ids_tracks_assigned_HA]))
        self.matches[0] = real_pos_tracks_matched

        ids_tracks_unassigned_HA = ids_track_HA[tracks_unassigned_HA]
        real_pos_tracks_unmatched = ([np.where(tracks_id == id)[0] for id in ids_tracks_unassigned_HA])
        if real_pos_tracks_unmatched.__len__() == 0:
            self.unmatched_tracks.clear()

        else:
            self.unmatched_tracks.clear()
            for i in real_pos_tracks_unmatched:
                self.unmatched_tracks.append(np.int64(i[0]))



        for i in pos_track_to_remove:
            self.unmatched_tracks.append(np.int64(i))

        for i in pos_cluster_to_remove:
            self.unmatched_clusters.append(np.int64(i))


        # for t, trk in enumerate(self.tracks_KF):
        #     if (t not in  self.matches[0]):
        #         self.unmatched_tracks.append(np.int64(t))
        #
        # for d, det in enumerate(clusters):
        #     if (d not in self.matches[1]):
        #         self.unmatched_clusters.append(np.int64(d))






    def cluster_track_assignment(self, clusters, display):

        # Clusters
        clusters_xw = np.array(list(item['xw'] for item in clusters))
        clusters_yw = np.array(list(item['yw'] for item in clusters))

        num_clusters = clusters_xw.size
        clusters_position = np.zeros((num_clusters,2))
        clusters_position[:,0] = clusters_xw
        clusters_position[:,1] = clusters_yw
        clusters_id = np.array(range(num_clusters))
        # Tracks
        tracks_xw = np.array(list(item['xw'] for item in self.tracks_KF))
        tracks_yw = np.array(list(item['yw'] for item in self.tracks_KF))
        tracks_id = np.array(list(item['id'] for item in self.tracks_KF))


        num_tracks = tracks_xw.size
        tracks_position = np.zeros((num_tracks, 2))
        tracks_position[:, 0] = tracks_xw.T
        tracks_position[:, 1] = tracks_yw.T

        # Cost matrix
        cost = np.zeros((num_tracks, num_clusters))

        if num_clusters != 0:


            for i in range(num_tracks):

                # difference = clusters_position - np.matlib.repmat(tracks_position[i, :], num_clusters, 1)
                # cost[i,:] = (np.sum(pow(difference,2),axis = 1)).T #comprobar que la suma se hace bien
                cost[i, :] = np.linalg.norm((clusters_position - np.matlib.repmat(tracks_position[i, :], num_clusters, 1)), axis=1)

            pos_track_to_remove = np.where(np.sum((cost < self.CONFIG['DIST_TH']) * 1, axis=1) == 0)[0]
            if len(pos_track_to_remove) > 0:
                a = 1

            ids_track_HA = np.delete(tracks_id, pos_track_to_remove)
            pos_track_HA = np.array(range(num_tracks - len(pos_track_to_remove)))

            cost_filtered = np.delete(cost, pos_track_to_remove, axis = 0)

             #Add filtering clusters that are close to all tracks

            pos_cluster_to_remove = np.where(np.sum((cost < (self.CONFIG['DIST_TH']+0.00001)) * 1, axis=0) == 0)[0]

            if len(pos_cluster_to_remove) > 0:
                a = 1
            ids_cluster_HA = np.delete(clusters_id, pos_cluster_to_remove)
            pos_cluster_HA = np.array(range(num_clusters - len(pos_cluster_to_remove)))

            cost_filtered = np.delete(cost_filtered, pos_cluster_to_remove, axis=1)

            # SEGUIR CNO LA SIGUIENTE FUNCION

        if num_clusters != 0 and num_tracks != 0:

              self.assign_detections_to_tracks(cost_filtered,clusters, pos_track_HA, ids_track_HA,tracks_id, pos_track_to_remove, pos_cluster_HA, ids_cluster_HA, clusters_id, pos_cluster_to_remove)
              a=1



        else:

            self.matches = []

            if num_clusters == 0:
                self.unmatched_clusters = []
                self.unmatched_tracks = np.array(range(0,num_tracks))

            if num_tracks == 0:
                self.unmatched_tracks = []
                self.unmatched_clusters = np.array(range(0, num_clusters))



    def update_assigned_tracks(self,clusters):
        # update tracks with assignments
        if self.matches.__len__() == 0:
            num_matched_tracks = 0
        else:
            num_matched_tracks = self.matches[0].__len__()

        for i in range(num_matched_tracks):

            track_id = self.matches[0][i]
            cluster_id = self.matches[1][i]

            #Correct the estimation of the object's location using the new detection. Update new centroid

            z = np.array([clusters[cluster_id]['xw'], clusters[cluster_id]['yw']])

            self.tracks_KF[track_id]['kalmanFilter'].update(z)

            # Update state, 1 = confirmed NOT USED
            if (self.tracks_KF[track_id]['totalVisibleCount'] >= 3) and (self.tracks_KF[track_id]['consecutiveInvisibleCount'] == 0):
                self.tracks_KF[track_id]['state'] = 1

            #Update track's age
            self.tracks_KF[track_id]['age'] = self.tracks_KF[track_id]['age'] + 1

            #Update visibility
            self.tracks_KF[track_id]['totalVisibleCount'] = self.tracks_KF[track_id]['totalVisibleCount'] + 1
            self.tracks_KF[track_id]['consecutiveInvisibleCount'] = 0

            self.tracks_KF[track_id]['fromcluster'] = np.array([cluster_id])

    def update_unassigned_tracks(self):

        num_unmatched_tracks = self.unmatched_tracks.__len__()

        for i in range(num_unmatched_tracks):

            track_id = self.unmatched_tracks[i]
            self.tracks_KF[track_id]['consecutiveInvisibleCount'] = self.tracks_KF[track_id]['consecutiveInvisibleCount'] +1
            self.tracks_KF[track_id]['age'] = self.tracks_KF[track_id]['age'] + 1

            self.tracks_KF[track_id]['fromcluster'] = np.array([])


    def check_unassigned_clusters(self, clusters, association_matrix,dist_features, dist_spatial):

        if self.unmatched_clusters.__len__() != 0 and self.tracks_KF.__len__() != 0 and association_matrix.shape[0] != 0 :

            matched_clusters = self.matches[1]
            # num_matched_clusters = len(matched_clusters)
            #
            # clusters_matched_xw = np.array(list(clusters[i]['xw'] for i in matched_clusters))
            # clusters_matched_yw = np.array(list(clusters[i]['yw'] for i in matched_clusters))
            #
            #
            # matched_clusters_position = np.zeros((num_matched_clusters, 2))
            # matched_clusters_position[:, 0] = clusters_matched_xw.T
            # matched_clusters_position[:, 1] = clusters_matched_yw.T
            unmatched_clusters = self.unmatched_clusters.copy()

            for u_cl in unmatched_clusters:


                if clusters[u_cl]['det'].__len__() ==1:


                   valid_matches = np.where(association_matrix[clusters[u_cl]['det'][0]['id_global'], :] == 1)[0]


                   if len(valid_matches) != 0:
                       posible_features = dist_features[u_cl, valid_matches]
                       #CAMBIAR
                       # posible_features = dist_spatial[u_cl, valid_matches]



                       #detection_to_join = valid_matches[np.where(posible_features == np.amin(posible_features))[0]]
                       # cambio debido a que EfficientDet puede sacar el mismo bbxos de dif clases me quedo con la primera ocurrencia
                       detection_to_join = valid_matches[np.argmin(posible_features)]
                       cluster_to_join = np.array([])
        #              look for the cluster containing this detection
                       "ANTIGUO funcionando bien hasta uso de det en vez de "
        #                for cl in range(0, clusters.__len__()):
        #                    for det in clusters[cl]['det']:
        #                        if det['id_global'] == detection_to_join:
        #                            dets_in_cluster_to_join = [i['id_global'] for i in clusters[cl]['det']]
        #                            if 100 not in association_matrix[clusters[u_cl]['det'][0]['id_global'],dets_in_cluster_to_join] :
        #                                cluster_to_join = cl
        #                                break
                       "Nuevo a raiz de un error 7/10"
                       for cl in range(0, clusters.__len__()):
                           for det in clusters[cl]['det']:
                               if det['id_global'] == detection_to_join:
                                   dets_in_cluster_to_join = [i['id_global'] for i in clusters[cl]['det']]
                                   if 100 not in association_matrix[
                                       clusters[u_cl]['det'][0]['id_global'], dets_in_cluster_to_join]:
                                       cluster_to_join = cl
                                       break


                       # look for track containing this cluster

                       for t in  range(0, self.tracks_KF.__len__()):

                           if cluster_to_join  in np.array(self.tracks_KF[t]['fromcluster']):

                               # self.tracks_KF[t]['fromcluster'] = (self.tracks_KF[t]['fromcluster'], u_cl)
                               self.tracks_KF[t]['fromcluster'] = np.concatenate((self.tracks_KF[t]['fromcluster'], np.array([u_cl])), axis=0)
                               self.unmatched_clusters.remove(u_cl)

                else:
                    a=1
            #         en este caso el unmatched cluster lejano tiene ms de una deteccion

    def delete_lost_tracks(self):


        if self.CONFIG['BLIND_OCCLUSION']:
            invisible_for_too_long = 10
        else:
            invisible_for_too_long = 2


        age_threshold = 8


        ages = np.array(list(item['age'] for item in self.tracks_KF))

        if ages.size != 0:
            totalVisibleCounts = np.array(list(item['totalVisibleCount'] for item in self.tracks_KF))
            visibility = totalVisibleCounts / ages

            consecutiveInvisibleCount = np.array(list(item['consecutiveInvisibleCount'] for item in self.tracks_KF))

            lostInds = np.bitwise_or(np.bitwise_and(ages < age_threshold, visibility < 0.6),   consecutiveInvisibleCount >= invisible_for_too_long)

            if len(np.where(lostInds == True)[0]) != 0:
                something_removed  = 1

            tracks_KF_clean = [item for item in self.tracks_KF if lostInds[self.tracks_KF.index(item)] == False]


            self.tracks_KF = tracks_KF_clean
            # lost_ids = consecutiveInvisibleCount >= invisible_for_too_long



    def save_global_tracking_data(self,clusters, f , global_tracks,cam):

        num_tracks = self.tracks_KF.__len__()

        # check overlap
        clusters_id = np.array(list(item['fromcluster'] for item in self.tracks_KF))
        dets = list()
        ids_cam = list()
        ids_track = list()
        id_tracks_to_clean = list()
        # np.array(list(item['fromcluster'] for item in clusters))
        # a = [item['det'] for item in clusters if clusters.index(item) in clusters_id]

        #
        if self.CONFIG['BLIND_OCCLUSION'] == True:
            invisible = 2
        else:
            invisible = 1


        if num_tracks != 0:
            self.updated_flag = 1

            for i in range(num_tracks):

                if self.tracks_KF[i]['consecutiveInvisibleCount'] < invisible: #1 2

                    global_tracks[f].append(self.new_global_track())
                    global_tracks[f][-1]['id'] = self.tracks_KF[i]['id']
                    global_tracks[f][-1]['xw'] = self.tracks_KF[i]['xw']
                    global_tracks[f][-1]['yw'] = self.tracks_KF[i]['yw']

                    from_cluster = self.tracks_KF[i]['fromcluster']

                    for s in range(from_cluster.size):

                        for c in range((clusters[int(from_cluster[s])]['det']).__len__()):

                            global_tracks[f][-1]['det'].append(clusters[int(from_cluster[s])]['det'][c])



                    if self.CONFIG['REPROJECTION']== True:
                        if from_cluster.size == 0:

                            pos_prev = [global_tracks[f-1].index(item) for item in global_tracks[f-1] if item['id'] == global_tracks[f][-1]['id']]
                            global_tracks[f][-1]['det'] = global_tracks[f-1][pos_prev[0]]['det']
                            prev_bbox = global_tracks[f - 1][pos_prev[0]]['det'][0]
                            # prev_basex = prev_bbox['x'] + (prev_bbox['w']/2)
                            # prev_basey = prev_bbox['y'] + (prev_bbox['h'])

                            centroid_x = self.tracks_KF[i]['xw']
                            centroid_y = self.tracks_KF[i]['yw']

                            base_x, base_y = cam.apply_homography_world_to_image(centroid_x, -centroid_y,
                                                                                 cam.homography_matrix[
                                                                                     'c00' + str(prev_bbox['id_cam'])])

                            x = np.round(base_x - (prev_bbox['w'] / 2))
                            y = np.round(base_y - (prev_bbox['h']))
                            global_tracks[f][-1]['det'][0]['x'] = int(x)
                            global_tracks[f][-1]['det'][0]['y'] = int(y)

            for item in global_tracks[f]:
                for item2 in item['det']:
                    dets.append([item2['x'],item2['y'],item2['w'],item2['h']])
                    ids_cam.append(item2['id_cam'])
                    ids_track.append(item['id'])


            for det1 in dets:
                for det2 in dets:
                    if det1 != det2 and dets.index(det2)>dets.index(det1):

                        # print(dets.index(det1))
                        # print(' and ')
                        # print(dets.index(det2))
                        # # enter bbox like x1 x2 y1 y

                        box1 = np.array((det1[0],det1[1],det1[0]+det1[2],det1[1]+det1[3]))
                        box2 = np.array((det2[0], det2[1], det2[0] + det2[2], det2[1] + det2[3]))                    #
                        iou = bbox.bbox_iou(torch.from_numpy(box1).cuda(),torch.from_numpy(box2).cuda())
                        if iou.item() > 0.8:

                            index1 = dets.index(det1)
                            index2 = dets.index(det2)
                            id_track1 = ids_track[index1]
                            id_track2 = ids_track[index2]
                            id_cam1 = ids_cam[index1]
                            id_cam2 = ids_cam[index2]

                            if id_cam1 == id_cam2:
                                age1 = [item['age'] for item in self.tracks_KF if item['id'] == id_track1]
                                age2 = [item['age'] for item in self.tracks_KF if item['id'] == id_track2]

                                if age1[0] > age2[0]:
                                    # delete track2 of globl and track_KF
                                    id_tracks_to_clean.append(id_track2)

                                else:
                                    id_tracks_to_clean.append(id_track1)

            tracks_KF_clean = [item for item in self.tracks_KF if item['id'] not in  id_tracks_to_clean]
            self.tracks_KF = tracks_KF_clean

            global_tracks_clean = [item for item in global_tracks[f] if item['id'] not in id_tracks_to_clean]
            global_tracks[f] = global_tracks_clean


        else:
            self.updated_flag = 0
            global_tracks[f] = []

        return global_tracks



    def display_tracks(self):
        if self.tracks_KF.__len__() > 0:
            # plt.figure()

            for t in self.tracks_KF:

                plt.plot(t['xw'], t['yw'], '*', lineWidth=1, markerSize=10, color='red')
                plt.text(t['xw'] - 0.000005, t['yw'] + 0.000005, str(t['id']), fontsize=15, color='red')






