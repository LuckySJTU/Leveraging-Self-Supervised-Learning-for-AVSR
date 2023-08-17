import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class ZeroPad3D(nn.Module):
    def __init__(self,leftpad,rightpad):
        super().__init__()
        self.leftpad=leftpad
        self.rightpad=rightpad
    def forward(self,x):
        return F.pad(x,(0,0,0,0,self.leftpad,self.rightpad),"constant",0)
    
class ReplicationPad3D(nn.Module):
    def __init__(self,leftpad,rightpad):
        super().__init__()
        self.leftpad=leftpad
        self.rightpad=rightpad
    def forward(self,x):
        return F.pad(x,(0,0,0,0,self.leftpad,self.rightpad),"replicate")
    
class Conv3DAggegator(nn.Module):
    def __init__(
            self,
            conv_layers,
            embed,
            dropout,
            skip_connections,
            residual_scale,
            non_affine_group_norm,
            conv_bias,
            zero_pad,
            activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k[0] // 2
            kb = ka - 1 if k[0] % 2 == 0 else ka

            if zero_pad:
                pad = ZeroPad3D(ka+kb,0)
            else:
                pad = ReplicationPad3D(ka+kb,0)

            return nn.Sequential(
                pad,
                nn.Conv3d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                #norm_block(True, n_out, affine=not non_affine_group_norm),
                #activation,
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv3d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x): #16*1*29*112*112
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x

class VisFeatureExtractionModel(nn.Module):
    def __init__(self, frontend):
        super(VisFeatureExtractionModel, self).__init__()
        # Conv3D
        self.conv3D = Conv3DAggegator(
            conv_layers=[(64,(3,4,4),(1,2,2)),(64,(3,2,2),(1,2,2)),(64,(1,2,2),(1,1,1))],
            embed = 1,
            dropout=0,
            skip_connections=False,
            residual_scale=0.5,
            non_affine_group_norm=False,
            conv_bias=True,
            zero_pad=False,
            activation=nn.ReLU()
        )
        MoCoModel = models.__dict__[frontend]()
        MoCoModel.fc = nn.Identity()
        MoCoModel.conv1 = nn.Identity()
        MoCoModel.bn1 = nn.Identity()
        MoCoModel.relu = nn.Identity()
        MoCoModel.maxpool = nn.Identity()
        self.MoCoModel = MoCoModel

        # print(self.MoCoModel)

    def forward(self, x):
        #input x : 16*29*1*112*112
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3D(x) # 16*64*29*26*26
        x = x.permute(0, 2, 1, 3, 4) #16*29*64*26*26
        # x: B x N x C x H x W
        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) #464*64*26*26
        x = self.MoCoModel(x) #464*2048
        return x
        # return shape: (B*N) x 512

        # refer models/moco_visual_frontend.py