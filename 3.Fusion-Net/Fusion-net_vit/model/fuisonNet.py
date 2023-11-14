import numpy as np
import torch
import torch.nn as nn
# from .vgg import vgg
from .vit_model import vit_base_patch16_224_in21k as create_model
from .vit_model import Block
from functools import partial

# def create_model(num_classes):
#     model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
#     return model
class CustomModule(nn.Module):
    def __init__(self, input_size):
        super(CustomModule, self).__init__()
        # 定义权重参数为可学习的张量，大小为 [1, 1, input_size]
        self.weight1 = nn.Parameter(torch.rand(1, 1, input_size))
        self.weight2 = nn.Parameter(torch.rand(1, 1, input_size))
        self.bias1 = nn.Parameter(torch.rand(input_size))  # 可学习的偏置参数
        self.bias2 = nn.Parameter(torch.rand(input_size))  # 可学习的偏置参数

    def forward(self, input1, input2):
        # 使用广播机制将权重参数复制到与输入张量相同的维度上
        weight1 = self.weight1.expand(input1.size())  # 大小变为 [batch_size, 26, input_size]
        weight2 = self.weight2.expand(input2.size())  # 大小变为 [batch_size, 26, input_size]

        # 对输入张量分别应用权重和偏置参数
        input1 = input1 * weight1 + self.bias1
        input2 = input2 * weight2 + self.bias2

        # 将两个张量相加
        output = input1 + input2

        return output



class FusionNet(nn.Module):
    def __init__(self, input_channel, backbone_type):
        super(FusionNet, self).__init__()
        if backbone_type == 'unet':
            # self.backbone_features = vgg(model_name=backbone_type, num_classes=5, init_weights=True)
            # model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
            self.backbone1 = create_model(num_classes=5, has_logits=False)
            self.backbone2 = create_model(num_classes=5, has_logits=False)
        else:
            raise NotImplementedError

        self.fusion = CustomModule(input_size=768)
        # dim,
        #                  num_heads,
        #                  mlp_ratio=4.,
        #                  qkv_bias=False,
        #                  qk_scale=None,
        #                  drop_ratio=0.,
        #                  attn_drop_ratio=0.,
        #                  drop_path_ratio=0.,
        #                  act_layer=nn.GELU,
        #                  norm_layer=nn.LayerNorm

        norm_layer = None
        act_layer = None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm = norm_layer(768)
        self.pre_logits = nn.Identity()
        #
        # Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #       drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #       norm_layer=norm_layer, act_layer=act_layer)
        self.ViT_Block1 = Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
              drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
              norm_layer=norm_layer, act_layer=act_layer)
        self.ViT_Block2 = Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                               drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                               norm_layer=norm_layer, act_layer=act_layer)
        self.ViT_Block3 = Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                                drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                                norm_layer=norm_layer, act_layer=act_layer)

        # self.classifier = nn.Sequential(
        #      # nn.Linear(4096, 1024),
        #     nn.Linear(4096, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     # nn.Linear(1024, 512),
        #     nn.Linear(512, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     # nn.Linear(512, num_classes)
        #     # nn.Linear(512, 80)
        #     nn.Linear(256, 80)
        # )

        self.classifier = nn.Sequential(
            # nn.Linear(4096, 1024),
            nn.Linear(19968, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # nn.Linear(1024, 512),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            # nn.Linear(512, num_classes)
            # nn.Linear(512, 80)
            nn.Linear(1024, 80)
        )

        # self.classifier = nn.Sequential(
        #     # nn.Linear(4096, 1024),
        #     nn.Linear(768, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     # nn.Linear(1024, 512),
        #     nn.Linear(256, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.5),
        #     # nn.Linear(512, num_classes)
        #     # nn.Linear(512, 80)
        #     nn.Linear(256, 80)
        # )

    def forward(self, blood_Matrix, relative_Matrix):
        # blood_Matrix = self.backbone_features(blood_Matrix)
        # relative_Matrix = self.backbone_features(relative_Matrix)

        blood_Matrix = self.backbone1(blood_Matrix)
        relative_Matrix = self.backbone2(relative_Matrix)

        # 拼接起来， flatten, 然后接上一个一批线性层，输出80维的向量
        # print(blood_Matrix.shape)
        # print(relative_Matrix.shape)

        fusion_embd = self.fusion(blood_Matrix, relative_Matrix)
        # print(fusion_embd.shape)
        # print("{}".format())

        final_features = self.ViT_Block1(fusion_embd)

        final_features = self.norm(final_features)

        final_features = self.ViT_Block2(final_features)

        final_features = self.norm(final_features)

        # final_features = self.ViT_Block3(final_features)
        #
        # final_features = self.norm(final_features)

        # final_features = self.pre_logits(final_features[:, 0])



        # print(final_features.shape)
        x = torch.flatten(final_features, start_dim=1)

        x = self.classifier(x)




        # print("{}".format())
        # final_tensor = Sim_A * Sim_Matrix + Score_B * Score_Matrix + Cat_C * Cat_Matrix

        return x



