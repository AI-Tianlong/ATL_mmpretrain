# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmcv.transforms import LoadImageFromFile, RandomFlip
from mmengine.dataset.sampler import DefaultSampler
from mmengine.dataset import default_collate

from mmpretrain.datasets import  PackInputs, RandomResizedCrop, BEiTMaskGenerator
from mmpretrain.models import SelfSupDataPreprocessor
from mmpretrain.datasets.atl_s2_10band_mae import ATL_S2_MAE_DATASET
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile

# dataset settings
dataset_type = ATL_S2_MAE_DATASET
data_root = 'data/atl_s2_10band_mae/'

data_preprocessor = dict(
    type=SelfSupDataPreprocessor,
    mean=None,
    std=None,
    # to_rgb=True
    )

train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),
    # dict(
    #     type=RandomResizedCrop,
    #     scale=224,
    #     crop_ratio_range=(0.2, 1.0),
    #     backend='pillow',
    #     interpolation='bicubic'),
    # dict(type=RandomFlip, prob=0.5),
    dict(
        type=BEiTMaskGenerator,
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=75,
        min_num_patches=16),
    dict(type=PackInputs)
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix = 'img_dir/',
        pipeline=train_pipeline))
