
import torch
import torch.nn as nn

from .vit_model import vit_base_patch16_224_in21k as create_model
from .vit_model import Block
from functools import partial

class CustomModule(nn.Module):
    def __init__(self, input_size):
        super(CustomModule, self).__init__()

        self.weight1 = nn.Parameter(torch.rand(1, 1, input_size))
        self.weight2 = nn.Parameter(torch.rand(1, 1, input_size))
        self.bias1 = nn.Parameter(torch.rand(input_size))
        self.bias2 = nn.Parameter(torch.rand(input_size))

    def forward(self, input1, input2):

        weight1 = self.weight1.expand(input1.size())
        weight2 = self.weight2.expand(input2.size())


        input1 = input1 * weight1 + self.bias1
        input2 = input2 * weight2 + self.bias2

        output = input1 + input2

        return output



class FusionNet(nn.Module):
    def __init__(self, input_channel, backbone_type):
        super(FusionNet, self).__init__()
        if backbone_type == 'vit':

            self.backbone1 = create_model(num_classes=5, has_logits=False)
            self.backbone2 = create_model(num_classes=5, has_logits=False)
        else:
            raise NotImplementedError

        self.fusion = CustomModule(input_size=768)

        norm_layer = None
        act_layer = None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm = norm_layer(768)
        self.pre_logits = nn.Identity()

        self.ViT_Block1 = Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
              drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
              norm_layer=norm_layer, act_layer=act_layer)
        self.ViT_Block2 = Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                               drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                               norm_layer=norm_layer, act_layer=act_layer)
        self.ViT_Block3 = Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                                drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                                norm_layer=norm_layer, act_layer=act_layer)

        self.classifier = nn.Sequential(
            nn.Linear(19968, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 80)
        )



    def forward(self, blood_Matrix, relative_Matrix):


        blood_Matrix = self.backbone1(blood_Matrix)
        relative_Matrix = self.backbone2(relative_Matrix)

        fusion_embd = self.fusion(blood_Matrix, relative_Matrix)


        final_features = self.ViT_Block1(fusion_embd)

        final_features = self.norm(final_features)

        final_features = self.ViT_Block2(final_features)

        final_features = self.norm(final_features)

        x = torch.flatten(final_features, start_dim=1)

        x = self.classifier(x)

        return x



