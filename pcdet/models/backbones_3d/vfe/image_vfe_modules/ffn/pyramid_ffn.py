import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ifn
from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class Pyramid2DFFN(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize 2D feature network via pretrained model
        Args:
            model_cfg: EasyDict, Dense classification network config
            downsample_factor: int, feature map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.is_optimize = self.model_cfg.get('OPTIMIZE', True)

        # Create modules
        self.ifn = ifn.__all__[model_cfg.IFN.NAME](
            num_classes=model_cfg.IFN.NUM_CLASSES,
            backbone_name=model_cfg.IFN.BACKBONE_NAME,
            **model_cfg.IFN.ARGS
        )
        self.reduce_blocks = torch.nn.ModuleList()
        self.out_channels = {}
        for _idx, _channel in enumerate(model_cfg.CHANNEL_REDUCE["in_channels"]):
            _channel_out = model_cfg.CHANNEL_REDUCE["out_channels"][_idx]
            self.out_channels[model_cfg.IFN.ARGS['feat_extract_layer'][_idx]] = _channel_out
            block_cfg = {"in_channels": _channel,
                         "out_channels": _channel_out,
                         "kernel_size": model_cfg.CHANNEL_REDUCE["kernel_size"][_idx],
                         "stride": model_cfg.CHANNEL_REDUCE["stride"][_idx],
                         "bias": model_cfg.CHANNEL_REDUCE["bias"][_idx]}
            self.reduce_blocks.append(BasicBlock2D(**block_cfg))
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.out_channels

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        # Pixel-wise depth classification
        images = batch_dict["images"]
        ifn_result = self.ifn(images)

        for _idx, _layer in enumerate(self.model_cfg.IFN.ARGS['feat_extract_layer']):
            image_features = ifn_result[_layer]
            # Channel reduce
            if self.reduce_blocks[_idx] is not None:
                image_features = self.reduce_blocks[_idx](image_features)

            batch_dict[_layer+"_feat2d"] = image_features
        
        if self.training:
            # detach feature from graph if not optimize
            if "logits" in ifn_result:
                ifn_result["logits"].detach_()
            if not self.is_optimize:
                image_features.detach_()

            self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
        return batch_dict

    def get_loss(self):
        """
        Gets loss
        Args:
        Returns:
            loss: (1), Network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        return None, None