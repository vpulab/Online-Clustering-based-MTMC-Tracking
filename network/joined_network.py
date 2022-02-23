
import numpy as np
import torch
import argparse
import torch.nn as nn
import torchvision
from torchvision.models import resnet

class net_features_cars(nn.Module):

    def __init__(self, architecture):
        """
              Initialization of the network
              :param arch: Desired backbone for RGB branch. Either ResNet-18 or ResNet-50

        """
        super(net_features_cars, self).__init__()
        if architecture == 'ResNet50':

            base = resnet.resnet50(pretrained=True)

            # First initial block
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True))

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4
        self.avgpool7 = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(2048, 2)

#         terminar falta la fc

    def forward_once(self, x):
        """
             Network forward
             :param x: RGB Image
             :param sem: Semantic Segmentation score tensor
             :retu
        """

        x, pool_indices = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        conv_features_pool = self.avgpool7(e4)

        conv_features_pool = conv_features_pool.view(conv_features_pool.size(0), -1)
        # cls = self.fc(conv_features_pool)


        return conv_features_pool

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class net_classifier(nn.Module):

    def __init__(self):

        super(net_classifier, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, input1,flag_dropout):
        output1 = self.fc1(input1)
        output1 = self.bn1(output1)
        output1 = self.relu(output1)

        if flag_dropout:
            output1 = self.drop(output1)

        output1 = self.fc2(output1)
        return output1

class joined_network(nn.Module):

    def __init__(self, architecture, flag_dropout):

        super(joined_network, self).__init__()
        self.features = net_features_cars(architecture)
        self.classifier = net_classifier()
        self.flag_dropout = flag_dropout

    def forward(self, input1, input2):
        output1, output2 = self.features(input1, input2)

        features_cat = torch.cat([output1, output2], 1)

        output_cls = self.classifier(features_cat, self.flag_dropout)

        return output_cls





