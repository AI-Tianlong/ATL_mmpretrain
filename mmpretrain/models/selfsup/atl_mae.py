# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from mmpretrain.models import HiViT, VisionTransformer
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import build_2d_sincos_position_embedding
from .base import BaseSelfSupervisor


# 这是type，MAE方法。 MAEViT是backbone，MAEPretrainDecoder是neck，MAEPretrainHead是head
@MODELS.register_module()
class ATL_MAE(BaseSelfSupervisor):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """
    # 重写了 extract_feat 
    def extract_feat(self, inputs: torch.Tensor):  
        return self.backbone(inputs, mask=None) # [2,10,224,224]-->[1,50,1024]
    
    # 重写了 loss 的计算
    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        #(128,50,768) (128,196) (1281,196)
        latent, mask, ids_restore = self.backbone(inputs) #过backbone,输出(128,50,1024) (128,196) (128,196)
        pred, pred_rs_index = self.neck(latent, ids_restore) # 过neck，恢复特征 ([128, 196, 2560]) 2590: 16*16*10 = 2560 一个token这么多像素--->（3通道是128,196,768）
        loss = self.head.loss(pred, pred_rs_index, inputs, mask) # 过loss的forward
        losses = dict(loss=loss) # {'loss': tensor(1.3443, device='cuda:0', grad_fn=<DivBackward0>)}
        return losses
