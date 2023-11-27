import numpy as np
import torch
import torch.nn as nn
from .vgg import vgg


class FusionNet(nn.Module):
    def __init__(self, input_channel, backbone_type):
        super(FusionNet, self).__init__()

        if backbone_type == 'vgg16':
            self.backbone_features = vgg(model_name=backbone_type, num_classes=5, init_weights=True)
        else:
            raise NotImplementedError

        self.classifier = nn.Sequential(


            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 80)

        )

    def forward(self, blood_Matrix, relative_Matrix):
        blood_Matrix = self.backbone_features(blood_Matrix)
        relative_Matrix = self.backbone_features(relative_Matrix)


        final_matrix = torch.cat((blood_Matrix, relative_Matrix), dim=1)

        x = torch.flatten(final_matrix, start_dim=1)
        x = self.classifier(x)

        return x



