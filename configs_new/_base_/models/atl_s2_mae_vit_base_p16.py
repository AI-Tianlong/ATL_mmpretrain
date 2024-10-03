# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (MAE, MAEPretrainDecoder, MAEPretrainHead,
                               MAEViT, PixelReconstructionLoss)

# model settings
model = dict(
    type=MAE,   # 相当于 mmseg 中的 Encoder-Decoder
    backbone=dict(
        type=MAEViT,      
        arch='b', 
        img_size=224,
        in_channels = 10, #10波段
        patch_size=16, 
        mask_ratio=0.75),
    neck=dict(
        type=MAEPretrainDecoder,
        patch_size=16,
        in_chans=10,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type=MAEPretrainHead,
        in_channels=10,
        norm_pix=True,  # 是否对重建目标进行归一化
        patch_size=16,
        loss=dict(type=PixelReconstructionLoss, criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])
