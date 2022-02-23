
import numpy as np
import torch
import argparse
import torch.nn as nn
import torchvision
from torchvision.models import resnet



class net_id_classifier(nn.Module):

    def __init__(self, architecture, num_classes,size_fc):

        super(net_id_classifier, self).__init__()

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

            # self.fc1 = nn.Linear(2048, size_fc)
            # self.BN = nn.BatchNorm1d(size_fc)
            # self.ReLU = nn.ReLU(inplace=True)

            # self.fc = nn.Linear(2048, num_classes)
            self.fc = nn.Linear(size_fc, num_classes)



        if architecture == 'ResNet101':
            base = resnet.resnet101(pretrained=True)

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


            self.fc = nn.Linear(2048, num_classes)



    def forward(self, x):
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
        #
        conv_features_pool = self.avgpool7(e4)
        conv_features_pool = conv_features_pool.view(conv_features_pool.size(0), -1)


        #
        # x2 = self.avgpool7(e4)
        # x = x2.view(x2.size(0), -1)
        #
        # x = self.fc1(x)
        #
        # conv_features_pool = x.view(x.size(0), -1)
        #
        # # cls = self.fc(x)


        return conv_features_pool






