'''
################################
#         Features
              Extraction       #
################################
'''

import torch
import numpy as np
from torchvision import transforms
from misc import normalize as norm
import torch.nn.functional as F


class features():

    def __init__(self,dataset, net, characteristic):
        self.dataset = dataset
        self.characteristic = characteristic
        self.net = net

    def load_model(self, model_path, model_name):

        model_dict = self.net.state_dict()
        list_dict = list(model_dict)


        file_layers = '/home/vpu/Clustering-based-Multi-Target-Multi-Camera-Tracking/layers.txt'

        f_id = open(file_layers, 'w+')

        if model_name == 'reid.pth': # Gotten from team 12 BUPT
            loaded_model = torch.load(model_path)
            loaded_list = list(loaded_model.items())
            count = 0
            print('Layers: ')
            for name, value in model_dict.items():
                # if name not in ['fc.weight', "fc.bias"] and name.find('num_batches_tracked') == -1:
                layer_name, weights = loaded_list[count]
                if weights.size() == value.size():
                    model_dict[name] = weights
                    print(str(count) + ". Weigths loaded for " + name)
                else:
                    print(str(count) + ". Weigths NOT loaded for " + name)
                # else:
                #     print(str(count) + ". Weigths NOT loaded for " + name)
                count += 1

        if model_name == 'net_last.pth':
            loaded_model = torch.load(model_path)
            loaded_list = list(loaded_model.items())
            count = 0
            print('Layers: ')
            for name, value in model_dict.items():
                if name.find('add_block') == -1:
                    layer_name, weights = loaded_list[count]
                    if weights.size() == value.size():
                        model_dict[name] = weights
                        print(str(count) + ". Weigths loaded for " + name)
                    else:
                        print(str(count) + ". Weigths NOT loaded for " + name)

                    count += 1

        if model_name == 'resnet50_model_120.pth':
            loaded_model = torch.load(model_path)
            loaded_list = list(loaded_model.items())
            count = 0
            print('Layers: ')
            for name, value in model_dict.items():
                if name.find('add_block') == -1:
                    layer_name, weights = loaded_list[count]
                    if weights.size() == value.size():
                        model_dict[name] = weights
                        print(str(count) + ". Weigths loaded for " + name)
                    else:
                        print(str(count) + ". Weigths NOT loaded for " + name)
                    # else:
                    #     print(str(count) + ". Weigths NOT loaded for " + name)
                    count += 1

        if model_name == 'best_checkpoint.pth.tar':
            loaded_model = torch.load(model_path)
            loaded_list = list(loaded_model['net_state_dict'].items())
            count = 0
            print('Layers: ')
            for name, value in model_dict.items():

                layer_name, weights = loaded_list[count]
                if weights.size() == value.size():
                    model_dict[name] = weights
                    print(str(count) + ". Weigths loaded for " + name)
                else:
                    print(str(count) + ". Weigths NOT loaded for " + name)
                # else:
                #     print(str(count) + ". Weigths NOT loaded for " + name)
                count += 1

            count = 0
            for name, value in loaded_model.items():
                # if name not in ['fc.weight', "fc.bias"] and name.find('num_batches_tracked') == -1:
                layer_name, weights = list_dict[count]

                f_id.write("%d  %s              %s \n" % (count, name, layer_name))

                count += 1
            f_id.close()

            self.net.load_state_dict(model_dict)

        # self.net.load_state_dict(loaded_model, strict=True)


    def extract(self, img):
        # Apply network
        # bbox_padded_tensor = self.dataset.data_transform((img))
        bbox_padded_tensor = torch.unsqueeze(img, dim=0)
        self.net.cuda()
        features = self.net(bbox_padded_tensor.cuda())
        features = torch.squeeze(features)

        # features_np = features.detach().cpu().numpy()
        # features__norm = norm.l2_norm(features)

        features_norm = F.normalize(features, p=2, dim = 0)
        features_np = features.detach().cpu().numpy()
        return features_np

    def apply_restrictions(self, dist_features, dist_spatial, sct, dist_th,mode):

        association_matrix = np.zeros((sct['id'].__len__(),sct['id'].__len__()))


        if mode == 'appearance':
            for i in range(sct['id'].__len__()):
                for j in range(sct['id'].__len__()):

                    # Only half of the symmetric matrix
                    if (i != j) & (i > j):

                        if (sct['id_cam'][i] == sct['id_cam'][j]):
                            association_matrix[i, j] = 100
                            association_matrix[j, i] = 100
                        else:
                            association_matrix[i, j] = 1
                            association_matrix[j, i] = 1

        else:
            for i in range(sct['id'].__len__()):
                for j in range(sct['id'].__len__()):

                    # Only half of the symmetric matrix
                    if (i != j) & (i > j):
                        distance = dist_spatial[i, j]

                          # Possible association if both points are close and from different cameras
                        if distance <= dist_th :
                            association_matrix[i, j] = 1
                            association_matrix[j, i] = 1

                        else:
                            association_matrix[i, j] = 10
                            association_matrix[j, i] = 10

                        if (sct['id_cam'][i] == sct['id_cam'][j]):
                            association_matrix[i, j] = 100
                            association_matrix[j, i] = 100




        # association_matrix = 1 - association_matrix
        # association_matrix = 10. * association_matrix
        # association_matrix[association_matrix == 0] = 1

        restricted_dist_features = dist_features * association_matrix

        return restricted_dist_features,association_matrix


    def apply_restrictions_logits(self, dist_features, dist_spatial, sct, dist_th,mode):

        association_matrix = np.zeros((sct['id'].__len__(),sct['id'].__len__()))


        if mode == 'appearance_only':
            for i in range(sct['id'].__len__()):
                for j in range(sct['id'].__len__()):

                    # Only half of the symmetric matrix
                    if (i != j) & (i > j):

                        if (sct['id_cam'][i] == sct['id_cam'][j]):
                            association_matrix[i, j] = 100
                            association_matrix[j, i] = 100
                        else:
                            association_matrix[i, j] = 1
                            association_matrix[j, i] = 1

        else:
            for i in range(sct['id'].__len__()):
                for j in range(sct['id'].__len__()):

                    # Only half of the symmetric matrix
                    if (i != j) & (i > j):
                        distance = dist_spatial[i, j]

                          # Possible association if both points are close and from different cameras
                        if distance > dist_th :
                            dist_features[i, j] = min(1, dist_features[i, j] + 0.5)
                            dist_features[j, i] = min(1, dist_features[j, i] + 0.5)
                            association_matrix[i, j] = 10
                            association_matrix[j, i] = 10
                        else:
                            association_matrix[i, j] = 1
                            association_matrix[j, i] = 1


                        if (sct['id_cam'][i] == sct['id_cam'][j]):
                            dist_features[i, j] = 1
                            dist_features[j, i] = 1
                            association_matrix[i, j] = 100
                            association_matrix[j, i] = 100

        # association_matrix = 1 - association_matrix
        # association_matrix = 10. * association_matrix
        # association_matrix[association_matrix == 0] = 1

        return dist_features, association_matrix









