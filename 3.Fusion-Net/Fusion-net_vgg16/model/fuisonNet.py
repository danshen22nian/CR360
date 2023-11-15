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
             # nn.Linear(4096, 1024),
            # ori
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

            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # nn.Linear(1024, 512),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # nn.Linear(512, num_classes)
            # nn.Linear(512, 80)
            nn.Linear(1024, 80)

        )

    def forward(self, blood_Matrix, relative_Matrix):
        blood_Matrix = self.backbone_features(blood_Matrix)
        relative_Matrix = self.backbone_features(relative_Matrix)

        # 拼接起来， flatten, 然后接上一个一批线性层，输出80维的向量
        # print(blood_Matrix.shape)
        # print(relative_Matrix.shape)
        # print("{}".format())
        # 沿dimison1 拼接起来，然后送入线性层中
        final_matrix = torch.cat((blood_Matrix, relative_Matrix), dim=1)
        # print(final_matrix.shape)
        x = torch.flatten(final_matrix, start_dim=1)
        x = self.classifier(x)
        # print("{}".format())
        # final_tensor = Sim_A * Sim_Matrix + Score_B * Score_Matrix + Cat_C * Cat_Matrix

        return x



