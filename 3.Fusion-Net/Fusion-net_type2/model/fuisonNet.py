import numpy as np
import torch
import torch.nn as nn
# from .vgg import vgg

from .unet import UNet

def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, feature_map1, feature_map2):

        conv_feature_map1 = self.conv1(feature_map1)
        conv_feature_map2 = self.conv2(feature_map2)

        batch_sizeNum = conv_feature_map1.shape[0]
        attention_scores = torch.matmul(conv_feature_map1.view(batch_sizeNum, 1, -1), conv_feature_map2.view(batch_sizeNum, 1, -1).permute(0, 2, 1))
        attention_scores = self.softmax(attention_scores)

        attended_feature_map2 = torch.matmul(attention_scores, feature_map2.view(batch_sizeNum, 1, -1)).view(batch_sizeNum, 1, 80, 80)


        return attended_feature_map2

class FuseTwoMap(nn.Module):
    def __init__(self):
        super(FuseTwoMap, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))

    def forward(self, feature_map1, feature_map2):

        weighted_feature_map1 = self.weight1 * feature_map1
        weighted_feature_map2 = self.weight2 * feature_map2


        fused_feature_map = weighted_feature_map1 + weighted_feature_map2

        return fused_feature_map

class FusionNet(nn.Module):
    def __init__(self, input_channel, backbone_type):
        super(FusionNet, self).__init__()
        if backbone_type == 'unet':
            self.backbone1 = UNet(in_channels=2, num_classes=1, base_c=32)
            self.backbone2 = UNet(in_channels=2, num_classes=1, base_c=32)
        else:
            raise NotImplementedError



        self.classifier = nn.Sequential(

            nn.Linear(6400, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 80)
        )

        self.attention = AttentionModule()
        self.mapFuse = FuseTwoMap()


    def forward(self, blood_Matrix, relative_Matrix):

        blood_Matrix = self.backbone1(blood_Matrix)
        relative_Matrix = self.backbone2(relative_Matrix)

        blood_toRel = self.attention(blood_Matrix, relative_Matrix)
        relative_ToBlo = self.attention(relative_Matrix, blood_Matrix)

        final_map = self.mapFuse(blood_toRel, relative_ToBlo)
        x = torch.flatten(final_map, start_dim=1)


        x = self.classifier(x)
        return x



