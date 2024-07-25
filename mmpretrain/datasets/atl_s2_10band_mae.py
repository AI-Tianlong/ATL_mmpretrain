# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .categories import IMAGENET_CATEGORIES
from .custom import CustomDataset


@DATASETS.register_module()
class ATL_S2_MAE_DATASET(CustomDataset):
    """
    # 对于无监督任务
        atl_s2_image
        ├── train
        │   ├── x1.tif
        │   ├── y1.tif
        │   └── ...
        ├── val
        │   ├── x3.tif
        │   ├── y3.tif
        │   └── ...
        └── meta
            ├── train.txt
            └── val.txt

    Examples:
        >>> train_dataloader = dict(
        >>> ...
        >>> # 训练数据集配置
        >>> dataset=dict(
        >>>     type= ATL_S2_MAE_DATASET,
                data_root = data_root,
        >>>     data_prefix='path/to/data_prefix',
        >>>     with_label=True,  # 对于无监督任务，使用 False
        >>>     pipeline=...
        >>>     )
        >>> )
    """

    def __init__(self,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 **kwargs):

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body
