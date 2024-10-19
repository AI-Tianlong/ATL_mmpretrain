# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS



@MODELS.register_module()
class ATL_PixelReconstructionLoss(BaseModule):
    """Loss for the reconstruction of pixel in Masked Image Modeling.

    This module measures the distance between the target image and the
    reconstructed image and compute the loss to optimize the model. Currently,
    This module only provides L1 and L2 loss to penalize the reconstructed
    error. In addition, a mask can be passed in the ``forward`` function to
    only apply loss on visible region, like that in MAE.

    Args:
        criterion (str): The loss the penalize the reconstructed error.
            Currently, only supports L1 and L2 loss
        channel (int, optional): The number of channels to average the
            reconstruction loss. If not None, the reconstruction loss
            will be divided by the channel. Defaults to None.
    """

    def __init__(self, criterion: str, channel: Optional[int] = None) -> None:
        super().__init__()

        if criterion == 'L1':    # 
            self.penalty = torch.nn.L1Loss(reduction='none')
        elif criterion == 'L2':  # 均方误差 ---> 让这两值尽量接近。
            self.penalty = torch.nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f'Currently, PixelReconstructionLoss \
            only supports L1 and L2 loss, but get {criterion}')

        self.channel = channel if channel is not None else 1

    # atl_head: loss = self.loss_module(pred, pred_index, target, target_index, mask)
    def forward(self,
                pred: torch.Tensor,
                pred_rs_index: torch.Tensor,
                target: torch.Tensor,
                target_index: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function to compute the reconstrction loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        # Pred 和 target 计算均方误差 loss
        loss_image = self.penalty(pred, target) # L2 loss, 能算出来一个值。 #[20,196,2560], 一个像素一个值。
        # pred_rs_index 和 target_index 计算均方误差 loss
        loss_index = self.penalty(pred_rs_index, target_index)  # [20,196,1024] # 这里算完，去sum就有nan，查一下

        # import pdb; pdb.set_trace()
        # if the dim of the loss is 3, take the average of the loss
        # along the last dim
        if len(loss_image.shape) == 3:
            loss_image = loss_image.mean(dim=-1)  # [batch, 196]--> 一个patch内一个loss   单个 0.38+

        if len(loss_index.shape) == 3:
            loss_index = loss_index.mean(dim=-1)  # [batch, 196]--> 一个patch内一个loss   单个 0.8+

        if mask is None:          # 如果没有mask，直接返回loss的均值
            loss_image = loss_image.mean()  # 
            loss_index = loss_index.mean()
        # loss的 尺寸：(1, 196, 768) * (1,196)
        # 计算 loss * mask, 没有掩码掉的地方，loss为0
        # 掩码掉的地方 loss 为 1
        # 否则，掩码的地方计算loss，没掩码的地方mask=0，loss=0 
        else:
            # import pdb;pdb.set_trace()
            loss_image_mean = (loss_image * mask).sum() / mask.sum() / self.channel  #平均像素的loss # 0.3821
            loss_index_mean = loss_index.mean() # 所有像素都做loss      # 0.7740, 但是这里会出现nan！！！！！ 出现nan怎么办！！！
            # 掩码了的loss，总共为943.2922, 平均在每个像素上的loss为0.0003
        
        loss = 1.0* loss_image_mean + 0.4 * loss_index_mean  # 保持一个大小？
        
        # rank = torch.distributed.get_rank()
        # if rank == 0:
        #     print('【ATL-loss_image_mean:', loss_image_mean, '【ATL-loss_index_mean:', loss_index_mean)
        
        # loss = loss_index_mean # 去测试，重建指数能否收敛，find_unuserd_parm=True。 0.7663附近

        if loss.isnan().any():
            rank = torch.distributed.get_rank()
            if rank == 0:
                print('【ATL-loss_image_mean:', loss_image_mean, '【ATL-loss_index_mean:', loss_index_mean)
            import numpy as np
            # np.save('/data/AI-Tianlong/openmmlab/mmpretrain/configs_new/atl-paper-test400e-addndvi/1-loss_image_mean-81npy', loss_image_mean.cpu().detach().numpy())
            np.save('/share/home/aitlong/AI-Tianlong/checkpoints/2-预训练过程中的问题/1-loss_index_mean-数.npy', loss_index_mean.cpu().detach().numpy())
            # np.save('/data/AI-Tianlong/openmmlab/mmpretrain/configs_new/atl-paper-test400e-addndvi/2-loss_image_mean-67npy', self.penalty(pred, target).cpu().detach().numpy())
            np.save('/share/home/aitlong/AI-Tianlong/checkpoints/2-预训练过程中的问题/2-loss_index_mean-67.npy', self.penalty(pred_rs_index, target_index).cpu().detach().numpy())
            np.save('/share/home/aitlong/AI-Tianlong/checkpoints/2-预训练过程中的问题/3-pred_rs_index-token.npy', pred_rs_index.cpu().detach().numpy())
            np.save('/share/home/aitlong/AI-Tianlong/checkpoints/2-预训练过程中的问题/3-target_index-token.npy', target_index.cpu().detach().numpy())
            np.save('/share/home/aitlong/AI-Tianlong/checkpoints/2-预训练过程中的问题/3-unpatchify-target_index-token.npy', unpatchify(target_index,4).cpu().detach().numpy())

            np.save('/share/home/aitlong/AI-Tianlong/checkpoints/2-预训练过程中的问题/4-unpatchify_pred.npy', unpatchify(pred,10).cpu().detach().numpy())
            np.save('/share/home/aitlong/AI-Tianlong/checkpoints/2-预训练过程中的问题/4-unpatchify_target.npy', unpatchify(target,10).cpu().detach().numpy())

            print('【ATL-LOG】nan in loss')
            import pdb;pdb.set_trace()

        return loss
    

def unpatchify(x: torch.Tensor,chanes) -> torch.Tensor:
    r"""Combine non-overlapped patches into images.

    Args:
        x (torch.Tensor): The shape is
            :math:`(B, L, \text{patch_size}^2 \times C)`. (1, 196, 768)

    Returns:
        torch.Tensor: The shape is :math:`(B, C, H, W)`.
    """
    # 16
    p = 16
    # h = w = 14
    h = w = int(x.shape[1]**.5) # [128,196,16*16*10]=14
    assert h * w == x.shape[1]
    # (1,196,768)-->(1,14,14,16,16,3)
    x = x.reshape(shape=(x.shape[0], h, w, p, p, chanes))  # [128,196,16*16*10]-->[128,14,14,16,16,10]
    # (1,14,14,16,16,3)-->(1,3,14,16,14,16)
    x = torch.einsum('nhwpqc->nchpwq', x) # [128,14,14,16,16,10]-->[128,10,14,16,14,16]
    # (1,3,14,16,14,16)-->(1,3,224,224)
    imgs = x.reshape(shape=(x.shape[0], chanes, h * p, h * p)) # [128,10,14,16,14,16] -> [128,10,224,224]
    return imgs