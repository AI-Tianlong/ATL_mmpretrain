# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MAEPretrainHead(BaseModule):
    """Head for MAE Pre-training.

    Args:
        loss (dict): Config of loss.
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
        in_channels (int): Number of input channels. Defaults to 3.
    """

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 patch_size: int = 16,
                 in_channels: int = 3) -> None:
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.loss_module = MODELS.build(loss)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        r"""Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images. The shape should
                be :math:`(B, C, H, W)`.  # (1,3,224,224)

        Returns:
            torch.Tensor: Patchified images. The shape is
            :math:`(B, L, \text{patch_size}^2 \times C)`.
        """
        p = self.patch_size # 16  (1,3,224,224) 224/16=14···0 可以整除
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        # h = w =224 / 16 = 14
        h = w = imgs.shape[2] // p
        # (1,3,224,224) -> (1,3,14,16,14,16)
        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, p, w, p))
        # (1,3,14,16,14,16)-->(1,14,14,16,16,3)
        x = torch.einsum('nchpwq->nhwpqc', x)
        # (1,14,14,16,16,3)-->(1,14*14,16*16*3)(1,196,768)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_channels))
        return x

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
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        # (1,196,768)-->(1,14,14,16,16,3)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_channels))
        # (1,14,14,16,16,3)-->(1,3,14,16,14,16)
        x = torch.einsum('nhwpqc->nchpwq', x)
        # (1,3,14,16,14,16)-->(1,3,224,224)
        imgs = x.reshape(shape=(x.shape[0], self.in_channels, h * p, h * p))
        return imgs

    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        """Construct the reconstruction target.

        In addition to splitting images into tokens, this module will also
        normalize the image according to ``norm_pix``.
        # 将对图像规范化

        Args:
            target (torch.Tensor): Image with the shape of B x C x H x W (1,3,224,224)
                                                        
        Returns:
            torch.Tensor: Tokenized images with the shape of B x L x C (1,196,768)
        """

        #  (1,3,224,224) --> (1,196,768)
        target = self.patchify(target)
        if self.norm_pix:
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return target

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        # (1,3,224,224) -> 归一化的(1,196,768)
        target = self.construct_target(target)
        # loss = pred - target 除了mask计算loss. MAE的forward传进来的
        loss = self.loss_module(pred, target, mask)

        return loss
