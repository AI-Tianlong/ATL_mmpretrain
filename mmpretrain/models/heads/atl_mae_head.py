# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from typing import List, Optional 


# Sentinel-2 Band name & array index
def Get_Sentinel2_Band_index():
    Sentinel_2_Band_map=dict(
        B2 = 0,
        B3 = 1,
        B4 = 2,
        B5 = 3,
        B6 = 4,
        B7 = 5,
        B8 = 6,
        B8A = 7,
        B11 = 8,
        B12 = 9
    )
    return Sentinel_2_Band_map

@MODELS.register_module()
class ATL_MAEPretrainHead(BaseModule):
    """Head for MAE Pre-training.

    Args:
        loss (dict): Config of loss.
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
        in_channels (int): Number of input channels. Defaults to 3.
        index_name (List[str]): The construct target name of RS index. Defaults to None.
    """

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 index_name: Optional[List[str]] = None) -> None:
        super().__init__()
        self.index_name = index_name

        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.loss_module = MODELS.build(loss)

    # 把图像打成不重叠的patch [128,10,224,224]-->[128,196,16*16*10]=[128,196,2560]
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        r"""Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images. The shape should
                be :math:`(B, C, H, W)`.        # (1,3,224,224)

        Returns:                                # (1,196,768)
            torch.Tensor: Patchified images. The shape is
            :math:`(B, L, \text{patch_size}^2 \times C)`.
        """
        B, C, H, W = imgs.shape # [128,10+4,224,224]

        p = self.patch_size # 16  (1,14,224,224) 224/16=14···0 可以整除
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # h = w =224 / 16 = 14
        h = w = imgs.shape[2] // p
        # (1,14,224,224) -> (1,14, 14,16, 14,16) [128,14,224,224]-->[128,14,14,16,14,16]
        x = imgs.reshape(shape=(imgs.shape[0], C, h, p, w, p))
        # (1,14,14,16,14,16)-->(1, 14,14, 16,16,14)
        x = torch.einsum('nchpwq->nhwpqc', x) # [128,10,14,16,14,16]->[128,14,14,16,16,10]
        # (1,14,14,16,16,14)-->(1, 14*14, 16*16*14)-->(1,196,3584)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * C)) # [128,14,14,16,16,10]-->[128,196,16*16*10]=[128,196,2560]
        return x

    # 把patch恢复成图像 [128,196,2560]-->[128,10,224,224]
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Combine non-overlapped patches into images.

        Args:
            x (torch.Tensor): The shape is
                :math:`(B, L, \text{patch_size}^2 \times C)`. (1, 196, 768)

        Returns:
            torch.Tensor: The shape is :math:`(B, C, H, W)`.
        """
        # 16
        p = self.patch_size
        # h = w = 14
        h = w = int(x.shape[1]**.5) # [128,196,16*16*10]=14
        assert h * w == x.shape[1]
        # (1,196,768)-->(1,14,14,16,16,3)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_channels))  # [128,196,16*16*10]-->[128,14,14,16,16,10]
        # (1,14,14,16,16,3)-->(1,3,14,16,14,16)
        x = torch.einsum('nhwpqc->nchpwq', x) # [128,14,14,16,16,10]-->[128,10,14,16,14,16]
        # (1,3,14,16,14,16)-->(1,3,224,224)
        imgs = x.reshape(shape=(x.shape[0], self.in_channels, h * p, h * p)) # [128,10,14,16,14,16] -> [128,10,224,224]
        return imgs


    # 本身图像就是反射率，这里不需要，所以要去掉！！！
    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        """Construct the reconstruction target.

        In addition to splitting images into tokens, this module will also
        normalize the image according to ``norm_pix``.  
        # 将对图像规范化

        Args:
            target (torch.Tensor): Image with the shape of B x C x H x W (1,3,224,224) [128,10,224,224]
                                                        
        Returns:
            torch.Tensor: Tokenized images with the shape of B x L x C (1,196,16*16*10)  # [128,196,2560] 指定的patch_size和embed dim
        """
        # B x len(index) x H x W 
        # 0:B2  1:B3  2:B4  3:B5  4:B6  5:B7  6:B8  7:B8A  8:B11  9:B12
        # [B, 4, H, W] # [128, 4, 224, 224]
        def limit_output(output):
            output = torch.clamp(output, -1, 1) # 限制所有值在【-2, -2之间】,防止过大导致梯度出问题
            return output
        S2 = Get_Sentinel2_Band_index()
        # 注意，这里的指数计算，可能会由于图像中含有nan值-->某些像素为0-->导致指数计算出现nan值，需要屏蔽掉这些nan值，指数中为nan的值变成0。
        target_index = torch.zeros(target.shape[0], len(self.index_name), target.shape[2], target.shape[3], device=target.device) # [B, 4, 224, 224]
        for i, rs_index in enumerate(self.index_name):
            if rs_index == 'SAVI': # SAVI = ((NIR-RED)/(NIR+RED+L))*(1+L) = ((B8-B4)/(B8+B4+0.5))*(1+0.5)  # B8A-B4/B8A+B4+0.5
                target_index[:, i] = ((target[:, S2['B8']] - target[:, S2['B4']]) / 
                                      (target[:, S2['B8']] + target[:, S2['B4']] + 0.5)) * (1+0.5) # 用不用+1.e-6
                target_index[:, i] = limit_output(target_index[:, i])
            elif rs_index == 'MNDWI': # MNDWI = (GREEN-SWIR1)/(GREEN+SWIR1) = (B3-B11)/(B3+B11)
                target_index[:, i] = (target[:, S2['B3']] - target[:, S2['B11']]) / (target[:, S2['B3']] + target[:, S2['B11']])
                target_index[:, i] = limit_output(target_index[:, i])
            elif rs_index == 'PII':  # 需要对水进行掩码处理
                # PII = 0.905*B2 -0.435*B8 +0.019
                target_index[:, i] = 0.905 * target[:, S2['B2']] - 0.435 * target[:, S2['B8']] + 0.019
                target_index[:, i] = limit_output(target_index[:, i])
            elif rs_index == '-MBSI': # MBSI = 2*(RED-GREEN)/(RED+GREEN-2) = 2*(B4-B3)/(B4+B3-2) # 裸土是负值，小的，所以处理要注意加-号？
                target_index[:, i] = (-1) * 2*(target[:, S2['B4']] - target[:, S2['B3']]) / (target[:, S2['B4']] + target[:, S2['B3']] - 2)
                target_index[:, i] = limit_output(target_index[:, i])

        if torch.isnan(target_index).any():
            target_index = torch.nan_to_num(target_index, nan=0.) #所有的nan变成了0.

        # # 保存target_index到本地
        # import numpy as np
        # np.save('/data/AI-Tianlong/openmmlab/mmpretrain/configs_new/atl-paper-test400e-addndvi/target_index.npy', target_index.cpu().numpy())
        # np.save('/data/AI-Tianlong/openmmlab/mmpretrain/configs_new/atl-paper-test400e-addndvi/target.npy', target.cpu().numpy())
        # import pdb;pdb.set_trace()
        #  (1,3,224,224) --> (1,196,768)

        # target = torch.cat([target, target_index], dim=1) # [128,10,224,224],[128,4,224,224]-->[128,196,2560+1024]
        target = self.patchify(target) # 图像打成patch--> [128,10,224,224]-->[128,196,2560]
        target_index = self.patchify(target_index) # RS 指数打成patch--> [128,4,224,224]-->[128,196,1024]
        if self.norm_pix:               # 弄成Fasle，否则根据目标的mean和var进行归一化。
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
            mean_index = target_index.mean(dim=-1, keepdim=True)
            var_index = target_index.var(dim=-1, keepdim=True)
            target_index = (target_index - mean_index) / (var_index + 1.e-6)**.5

        # import pdb;pdb.set_trace()
        return target, target_index  # 返回重建的目标，这个和neck的输出是一致的。

    def loss(self, 
             pred: torch.Tensor, 
             pred_index: torch.Tensor,
             target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            pred (torch.Tensor): The reconstructed RS index.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        # (1,3,224,224) -> 归一化的(1,196,768)
        target, target_index = self.construct_target(target)  # 一个[128,10,1024,1024]的图像变为 [128,196,2560+1024]
        # loss = pred - target 除了mask计算loss. MAE的forward传进来的
        # 去执行 ATL_construction_loss的forward
        loss = self.loss_module(pred, pred_index, target, target_index, mask)   #pred是neck的输出【128，196，2560】和targets做损失。mask是一个超参。

        return loss
