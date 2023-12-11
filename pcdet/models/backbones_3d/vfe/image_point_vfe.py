import torch

from .vfe_template import VFETemplate
from .image_vfe_modules import ffn, f2v
from ... import backbones_3d

class ImagePointVFE(VFETemplate):
    def __init__(self, model_cfg, grid_size, point_cloud_range, voxel_size, depth_downsample_factor, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.grid_size = grid_size
        self.pc_range = point_cloud_range
        self.voxel_size = voxel_size
        self.downsample_factor = depth_downsample_factor
        self.model_cfg = model_cfg
        self.module_topology = [
            'ffn', 'f2v', 'backbone_3d'
        ]
        self.build_modules()

    def build_modules(self):
        """
        Builds modules
        """
        for module_name in self.module_topology:
            module = getattr(self, 'build_%s' % module_name)()
            self.add_module(module_name, module)

    def build_backbone_3d(self):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=self.model_cfg.BACKBONE_3D.NUM_POINT_FEATURES,
            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            point_cloud_range=self.pc_range
        )
        return backbone_3d_module

    def build_ffn(self):
        """
        Builds frustum feature network
        Returns:
            ffn_module: nn.Module, Frustum feature network
        """
        ffn_module = ffn.__all__[self.model_cfg.FFN.NAME](
            model_cfg=self.model_cfg.FFN,
            downsample_factor=self.downsample_factor
        )
        self.disc_cfg = ffn_module.disc_cfg
        return ffn_module

    def build_f2v(self):
        """
        Builds frustum to voxel transformation
        Returns:
            f2v_module: nn.Module, Frustum to voxel transformation
        """
        f2v_module = f2v.__all__[self.model_cfg.F2V.NAME](
            model_cfg=self.model_cfg.F2V,
            grid_size=self.grid_size,
            pc_range=self.pc_range,
            disc_cfg=self.disc_cfg
        )
        return f2v_module

    def get_output_feature_dim(self):
        """
        Gets number of output channels
        Returns:
            out_feature_dim: int, Number of output channels
        """
        out_feature_dim = self.ffn.get_output_feature_dim()
        return out_feature_dim

    def get_mean_voxel(self, batch_dict):
        """
        Get normalized input points useing mean_vfe
        Returns:
            points_mean: Mean feature of voxelized points
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
            **kwargs:
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        batch_dict = self.ffn(batch_dict)

        # get mean voxelized points
        batch_dict = self.get_mean_voxel(batch_dict)
        #print("batch_dict:",batch_dict['voxel_features'].shape)
        batch_dict = self.backbone_3d(batch_dict, fuse_func=self.f2v)
        if self.training:
            self.ray_loss = {}
            for _key in batch_dict:
                if 'loss_layer' in _key:
                    self.ray_loss[_key] = batch_dict[_key]

        return batch_dict

    def get_loss(self, tb_dict):
        """
        Gets Network loss
        Returns:
            loss: (1), Network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        for _key in self.ray_loss:
            tb_dict[_key] = self.ray_loss[_key].item()
        loss = sum(self.ray_loss.values())
        return loss, tb_dict
