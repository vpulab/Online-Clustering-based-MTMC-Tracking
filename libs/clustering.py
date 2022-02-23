'''l
################################
#         Spatial
              Clustering       #
################################
'''

import numpy as np
from scipy.cluster import hierarchy
from thirdparty import sklearn_dunn
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

class clustering():

    def __init__(self, mtmc):
        self.clusters_frame = list()
        self.colors = mtmc.colors.list
        self.trajectories_f_w = list()


    def new_cluster(self):
        cluster = {
            'id': None,
            'xw': None,
            'yw': None,
            'det': list()}
        return cluster

    def new_detection(self):
        det = {
            'id_cam': None,
            'x': None,
            'y': None,
            'w': None,
            'h': None,
            'id': None,
            'id_global': None}
        return det

    def new_trajectory(self):
        det = {
            'id_cam': None,
            'id': None,
            'xw': None,
            'yy': None,
            'feature_descriptor': None}
        return det


    def compute_clusters(self, distance_matrix, association_matrix):
        # flat_tril = np.ravel((np.tril(distance_matrix)), order='F')
        # vec_dist_features = flat_tril[flat_tril != 0]
        num_det_f= distance_matrix.shape[0]
        # iu1 = np.triu_indices(num_det_f)
        #
        # distance_matrix2 = np.copy(distance_matrix)
        # distance_matrix2[iu1] = np.inf
        # flat_tril = np.ravel(distance_matrix2, order='F')
        # vec_dist_features = flat_tril[flat_tril != np.inf]
        #
        # Z = hierarchy.linkage(vec_dist_features, method='complete', metric='euclidean')
        # plt.figure()
        # dn = hierarchy.dendrogram(Z)
        # plt.show()

        max_num_clusters = num_det_f - 1
        min_num_clusters = max((num_det_f - (np.where(association_matrix == 1))[0].__len__() + 1), 2)


        if min_num_clusters > max_num_clusters:  #It can occur when there are some pairs of close detections being from the same cameras by pairs

            optimal_clusters = num_det_f
            idx_clusters = np.array(range(0, optimal_clusters))

        else:

            clusters = np.array(range(min_num_clusters, max_num_clusters + 1))
            num_clusters = (max_num_clusters - min_num_clusters + 1)

            labels = list()
            indices = np.zeros(num_clusters)

            for k in range(num_clusters):

                # Metodo 1
                #  labels.append(hierarchy.fcluster(Z, clusters[k], criterion='maxclust'))

                # Metodo 2
                cluster = AgglomerativeClustering(n_clusters=clusters[k], affinity='precomputed', linkage='complete')
                labels.append(cluster.fit_predict(distance_matrix))


                indices[k] = sklearn_dunn.dunn(labels[-1], distance_matrix)

            # max_index = indices.tolist().index(min(indices))

            if indices.__len__() == 1:  # If min_num_clusters == max_num_clusters
                optimal_clusters = clusters[0]
                idx_clusters = labels[0] # -1 so clusters labels start in 0
            else:
                derivative = np.diff(indices)
                pos = derivative.tolist().index(max(derivative)) + 1
                optimal_clusters = clusters[pos]
                idx_clusters = labels[pos]  # -1 so clusters labels start in 0

        return idx_clusters, optimal_clusters

    def display_detections_cluster(self,sct_f, det_in_cluster,cl):

        for d  in range(det_in_cluster.__len__()):
            det = det_in_cluster[d]
            xw = sct_f['xw'][det]
            yw = sct_f['yw'][det]
            plt.plot(xw, yw, 'x', lineWidth = 1, markerSize = 10, color = self.colors[cl])

    def display_centroid_cluster(self, mean_xw, mean_yw, cl):

            plt.scatter(mean_xw, mean_yw, s = 80, facecolors='none', edgecolors=self.colors[cl])
            plt.text(mean_xw+0.000005, mean_yw+0.000005, str(cl), fontsize=15, color='black')
            plt.title('Tracks')
