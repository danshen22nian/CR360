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
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)  # Convolution for the first feature map
        self.conv2 = nn.Conv2d(1, 1, kernel_size=1)  # Convolution for the second feature map
        self.softmax = nn.Softmax(dim=2)  # Softmax along the height and width dimensions

    def forward(self, feature_map1, feature_map2):
        # Apply convolutions to the feature maps
        conv_feature_map1 = self.conv1(feature_map1)
        conv_feature_map2 = self.conv2(feature_map2)

        batch_sizeNum = conv_feature_map1.shape[0]
        # Compute attention scores using the dot product
        # attention_scores = torch.matmul(conv_feature_map1.view(8, 1, -1), conv_feature_map2.view(8, 1, -1).permute(0, 2, 1))
        attention_scores = torch.matmul(conv_feature_map1.view(batch_sizeNum, 1, -1), conv_feature_map2.view(batch_sizeNum, 1, -1).permute(0, 2, 1))
        attention_scores = self.softmax(attention_scores)

        # Apply attention to the second feature map
        # attended_feature_map2 = torch.matmul(attention_scores, feature_map2.view(8, 1, -1)).view(8, 1, 80, 80)
        attended_feature_map2 = torch.matmul(attention_scores, feature_map2.view(batch_sizeNum, 1, -1)).view(batch_sizeNum, 1, 80, 80)

        # # Combine the attended feature map with the first feature map
        # fused_feature_map = feature_map1 + attended_feature_map2

        return attended_feature_map2

class FuseTwoMap(nn.Module):
    def __init__(self):
        super(FuseTwoMap, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(1))  # Learnable parameter for the first feature map
        self.weight2 = nn.Parameter(torch.randn(1))  # Learnable parameter for the second feature map

    def forward(self, feature_map1, feature_map2):
        # Apply the learnable weights to the feature maps
        weighted_feature_map1 = self.weight1 * feature_map1
        weighted_feature_map2 = self.weight2 * feature_map2

        # Add the weighted feature maps
        fused_feature_map = weighted_feature_map1 + weighted_feature_map2

        return fused_feature_map

class FusionNet(nn.Module):
    def __init__(self, input_channel, backbone_type):
        super(FusionNet, self).__init__()
        if backbone_type == 'unet':
            # self.backbone_features = vgg(model_name=backbone_type, num_classes=5, init_weights=True)
            self.backbone1 = UNet(in_channels=2, num_classes=1, base_c=32)
            self.backbone2 = UNet(in_channels=2, num_classes=1, base_c=32)
        else:
            raise NotImplementedError



        self.classifier = nn.Sequential(
             # nn.Linear(4096, 1024),
            # nn.Linear(4096, 512),
            # nn.ReLU(True),
            # nn.Dropout(p=0.5),
            # # nn.Linear(1024, 512),
            # nn.Linear(512, 256),
            # nn.ReLU(True),
            # nn.Dropout(p=0.5),
            # # nn.Linear(512, num_classes)
            # # nn.Linear(512, 80)
            # nn.Linear(256, 80)

            nn.Linear(6400, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # nn.Linear(512, num_classes)
            # nn.Linear(512, 80)
            nn.Linear(256, 80)
        )

        self.attention = AttentionModule()
        self.mapFuse = FuseTwoMap()


    def forward(self, blood_Matrix, relative_Matrix):
        # blood_Matrix = self.backbone_features(blood_Matrix)
        # relative_Matrix = self.backbone_features(relative_Matrix)

        blood_Matrix = self.backbone1(blood_Matrix)
        relative_Matrix = self.backbone2(relative_Matrix)

        # print(blood_Matrix.shape)
        # print(relative_Matrix.shape)

        blood_toRel = self.attention(blood_Matrix, relative_Matrix)
        relative_ToBlo = self.attention(relative_Matrix, blood_Matrix)

        final_map = self.mapFuse(blood_toRel, relative_ToBlo)
        # print(final_map.shape)

        # print("{}".format())
        # 沿dimison1 拼接起来，然后送入线性层中
        # final_matrix = torch.cat((blood_Matrix, relative_Matrix), dim=1)
        # print(final_matrix.shape)

        x = torch.flatten(final_map, start_dim=1)


        x = self.classifier(x)
        # print("{}".format())
        # final_tensor = Sim_A * Sim_Matrix + Score_B * Score_Matrix + Cat_C * Cat_Matrix

        return x



