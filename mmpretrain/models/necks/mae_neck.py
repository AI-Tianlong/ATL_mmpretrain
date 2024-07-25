# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from ..backbones.vision_transformer import TransformerEncoderLayer
from ..utils import build_2d_sincos_position_embedding


@MODELS.register_module()
class MAEPretrainDecoder(BaseModule):
    """Decoder for MAE Pre-training.

    Some of the code is borrowed from `https://github.com/facebookresearch/mae`. # noqa

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 1024.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmpretrain.models import MAEPretrainDecoder
        >>> import torch
        >>> self = MAEPretrainDecoder()
        >>> self.eval()
        >>> inputs = torch.rand(1, 50, 1024)
        >>> ids_restore = torch.arange(0, 196).unsqueeze(0)
        >>> level_outputs = self.forward(inputs, ids_restore)
        >>> print(tuple(level_outputs.shape))
        (1, 196, 768)
    """

    def __init__(self,
                 num_patches: int = 196,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: int = 4,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 predict_feature_dim: Optional[float] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_patches = num_patches

        # used to convert the dim of features from encoder to the dim
        # compatible with that of decoder  (1,50,768)-->(1,50,512)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # (1,1,512)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # create new position embedding, different from that in encoder
        # and is not learnable
        # 不可学习的参数(1, 196+1, 512)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False)

        # decoder layers, 8个decoder block
        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

        #()
        self.decoder_norm_name, decoder_norm = build_norm_layer(
            norm_cfg, decoder_embed_dim, postfix=1)
        self.add_module(self.decoder_norm_name, decoder_norm)

        # Used to map features to pixels
        if predict_feature_dim is None:  # 16*16*3 = 768
            predict_feature_dim = patch_size**2 * in_chans
        self.decoder_pred = nn.Linear(  # (512)--->(768)
            decoder_embed_dim, predict_feature_dim, bias=True)

    def init_weights(self) -> None:
        """Initialize position embedding and mask token of MAE decoder."""
        super().init_weights()

        decoder_pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.decoder_pos_embed.shape[-1],
            cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

        torch.nn.init.normal_(self.mask_token, std=.02)

    @property
    def decoder_norm(self):
        """The normalization layer of decoder."""
        return getattr(self, self.decoder_norm_name)

    def forward(self, x: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """The forward function.

        The process computes the visible patches' features vectors and the mask
        tokens to output feature vectors, which will be used for
        reconstruction.

        Args:
            x (torch.Tensor): hidden features, which is of shape
                    B x (L * mask_ratio) x C.  (1,40,3) ？
            ids_restore (torch.Tensor): ids to restore original image.

        Returns:
            torch.Tensor: The reconstructed feature vectors, which is of
            shape B x (num_patches) x C.
        """
        # embed tokens
        # 这里可以打印一下
        x = self.decoder_embed(x) #(1,50,768)-->(1,50,512)

        # append mask tokens to sequence
        # (1,1,512)-->(1,147,512)  # 196+1-50=147个mask token

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # (1,49,512)+(1,147,512) = (1,196,512) -->恢复成了原来的196个patch
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        # 把 196个 patch 弄回原来打乱前的顺序 (1,196,512)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # (1,1,512)+(1,196,512) = (1,197,512)
        x = torch.cat([x[:, :1, :], x_], dim=1)

        
        # add pos embed
        # token embed + pos embed
        # (1, 196+1, 512) + (1, 196+1, 512) = (1, 196+1, 512)，这里的pos_embed是不可学习的，197是因为有个cls_token
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection

        # (1,197,512)--->(1,197,768)
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


@MODELS.register_module()
class ClsBatchNormNeck(BaseModule):
    """Normalize cls token across batch before head.

    This module is proposed by MAE, when running linear probing.

    Args:
        input_features (int): The dimension of features.
        affine (bool): a boolean value that when set to ``True``, this module
            has learnable affine parameters. Defaults to False.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 input_features: int,
                 affine: bool = False,
                 eps: float = 1e-6,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg)
        self.bn = nn.BatchNorm1d(input_features, affine=affine, eps=eps)

    def forward(
            self,
            inputs: Tuple[List[torch.Tensor]]) -> Tuple[List[torch.Tensor]]:
        """The forward function."""
        # Only apply batch norm to cls_token
        inputs = [self.bn(input_) for input_ in inputs]
        return tuple(inputs)
