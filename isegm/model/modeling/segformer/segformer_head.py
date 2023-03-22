# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from ..att import Att, Multihead, MultiScaleAtt
#from mmseg.ops import resize
#from ..builder import HEADS
#from .decode_head import BaseDecodeHead
#from mmseg.models.utils import *
import attr

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

relu_inplace = True
#@HEADS.register_module()
class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, in_channels, num_classes, in_index, dropout_ratio, 
                    input_transform='multiple_select', **kwargs):
        super(SegFormerHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.in_index = in_index
        self.num_classes = num_classes

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            #norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        # what about channel problem?
        self.att = Multihead(270, reduction=3, heads=3, ratio=4)
        self.upatt = MultiScaleAtt(high_dim=270, low_dim=64, out_dim=270, reduction=3, ratio=2)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def forward(self, inputs, low_level,points=None):
        #x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        x =  [inputs[i] for i in self.in_index]
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _c=self.att(_c, points)
        _c=self.upatt(_c,low_level,points)
        
        x = self.dropout(_c)
        feature = x
        x = self.linear_pred(x)
        return x,feature


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           ):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)