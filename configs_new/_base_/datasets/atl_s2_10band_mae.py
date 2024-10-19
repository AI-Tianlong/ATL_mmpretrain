# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
# from mmcv.transforms import LoadImageFromFile, RandomFlip
from mmengine.dataset.sampler import DefaultSampler
from mmcv.transforms.processing import (RandomFlip, RandomResize, Resize,
                                        TestTimeAug)



from mmpretrain.datasets import  PackInputs, RandomResizedCrop, RandomCrop
from mmpretrain.models import SelfSupDataPreprocessor
from mmpretrain.datasets.atl_s2_10band_mae import ATL_S2_MAE_DATASET
from mmseg.datasets.transforms.loading import LoadSingleRSImageFromFile

# dataset settings
dataset_type = ATL_S2_MAE_DATASET
data_root = 'data/0-paper-pretrain/'
crop_size = (224, 224) 

# 本身就是,除了10000的反射率，不进行归一化处理了？
# 还是说在进行一下归一化处理？
data_preprocessor = dict(
    type=SelfSupDataPreprocessor,
    mean=None,
    std=None,
    # to_rgb=True
    )

train_pipeline = [
    dict(type=LoadSingleRSImageFromFile),  #还是需要随机裁剪的。MAE论文里说的！
    dict(
        type=RandomResizedCrop,
        scale=224,
        crop_ratio_range=(0.2, 1.0),
        backend='cv2',
        interpolation='bicubic'),
    # dict(  # 来自mmseg的随机裁剪
    #     type=RandomResize,
    #     scale=(224, 224),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type=RandomCrop, crop_size=crop_size), #这里有问题需要重写
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackInputs)
]

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
