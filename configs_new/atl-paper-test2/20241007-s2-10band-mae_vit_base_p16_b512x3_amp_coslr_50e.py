# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, LinearLR
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.optim.adamw import AdamW


from mmpretrain.models import (MAE, MAEPretrainDecoder, MAEPretrainHead,
                               MAEViT, PixelReconstructionLoss)

with read_base():
    from .._base_.models.atl_s2_mae_vit_base_p16 import *
    from .._base_.datasets.atl_s2_10band_mae import *
    from .._base_.default_runtime import *

# model settings
model = dict(
    type=MAE,   # 相当于 mmseg 中的 Encoder-Decoder
    backbone=dict(
        type=MAEViT,  # backbone是encoder
        arch='b',      # size: dim, num_layers, num_heads, feedforward_channels 
        img_size=224,  # s:768,8,8,768*3 | b:768,12,12,3072 | l:1024,24,16,4096  | h:1280,32,16,5120 | 'eva-g':1408,40,16,6144
        in_channels = 10, #10波段
        patch_size=16, 
        mask_ratio=0.75),
    neck=dict(      # neck 是decoder
        type=MAEPretrainDecoder,
        patch_size=16,
        in_chans=10,
        embed_dim=768,  # 和base什么得得对上
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type=MAEPretrainHead,   # 所以我要改重建的目标，就是改这里！ 怎么说，通过encode 和 decoder 简单的进行一个逐像素的分割。
        in_channels=10,
        norm_pix=False,  # 是否对重建目标进行归一化，默认True，因为本身就是0-1反射率，这里弄成False，可以弄一个消融实验。
        patch_size=16,
        loss=dict(type=PixelReconstructionLoss, criterion='L2')), # Pixel重建LOSS L2
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

# backbone 是 encoder, neck是decoder 在head中计算重建损失

# optimizer wrapper
optim_wrapper = dict(
    type=AmpOptimWrapper,
    loss_scale='dynamic',
    optimizer=dict(
        type=AdamW,
        lr=1.5e-4 * 512*3 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=45,
        by_epoch=True,
        begin=5,
        end=50,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=50)
# only keeps the latest 3 checkpoints
default_hooks.checkpoint = dict(
    type=CheckpointHook, interval=1, max_keep_ckpts=3)

randomness.update(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
per_gpu_batch_size = train_dataloader.batch_size
auto_scale_lr = dict(base_batch_size=per_gpu_batch_size*3)
