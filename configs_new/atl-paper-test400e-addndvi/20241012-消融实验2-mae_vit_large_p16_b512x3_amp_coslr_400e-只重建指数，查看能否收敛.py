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

from mmpretrain.models.selfsup.atl_mae import ATL_MAE
from mmpretrain.models.heads.atl_mae_head import ATL_MAEPretrainHead
from mmpretrain.models.necks.atl_mae_neck import ATL_MAEPretrainDecoder
from mmpretrain.models.losses.atl_reconstruction_loss import ATL_PixelReconstructionLoss

with read_base():
    from .._base_.models.atl_s2_mae_vit_base_p16 import *
    from .._base_.datasets.atl_s2_10band_mae import *
    from .._base_.default_runtime import *

# model settings
model = dict(
    type=ATL_MAE,   # 相当于 mmseg 中的 Encoder-Decoder
    backbone=dict(
        type=MAEViT,  # backbone是encoder  # 这个backbone不动，就是个ViT
        arch='l',      # size: dim, num_layers, num_heads, feedforward_channels 
        img_size=224,  # s:768,8,8,768*3 | b:768,12,12,3072 | l:1024,24,16,4096  | h:1280,32,16,5120 | 'eva-g':1408,40,16,6144
        in_channels = 10, #10波段
        patch_size=16, 
        mask_ratio=0.75),
    neck=dict(      # neck 是decoder
        type=ATL_MAEPretrainDecoder,
        patch_size=16,
        in_chans=10,
        embed_dim=1024,  # 和backbone的 embed_dim 0得对上
        decoder_embed_dim=512,  #这里也要调整吧？因为要恢复10个波段
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        index_name = ['SAVI','MNDWI','PII','-MBSI'], # 植被, 水体, 裸地，人造地表
        # index_name = None,
    ),
    head=dict(
        type=ATL_MAEPretrainHead,   # 所以我要改重建的目标，就是改这里！ 怎么说，通过encode 和 decoder 简单的进行一个逐像素的分割。
        in_channels=10,
        norm_pix=False,  # 是否对重建目标进行归一化，默认True，因为本身就是0-1反射率，这里弄成False，可以弄一个消融实验。
        patch_size=16,
        index_name = ['SAVI','MNDWI','PII','-MBSI'], # 植被, 水体, 裸地，人造地表
        loss=dict(type=ATL_PixelReconstructionLoss, criterion='L2')), # Pixel重建LOSS L2
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
        lr=1.5e-4 * 512*4 / 256,  # 这里和batchsize有关系啊！！！
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
        end=40,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=360,
        by_epoch=True,
        begin=40,
        end=400,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=400)
# only keeps the latest 3 checkpoints
default_hooks.checkpoint = dict(
    type=CheckpointHook, interval=1, max_keep_ckpts=5)

randomness.update(seed=0, diff_rank_seed=True)

# auto resume
resume = True
find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.


train_dataloader = dict(
    batch_size=512,  # batch=128*4 = 512。我扩大成4096呢？
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # data_prefix = 'img_dir/',
        pipeline=train_pipeline))

per_gpu_batch_size = train_dataloader.batch_size
auto_scale_lr = dict(base_batch_size=per_gpu_batch_size*4)
