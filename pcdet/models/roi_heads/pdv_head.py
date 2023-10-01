import math

import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG, StackSAModuleMSGAttention
from ...utils import common_utils, voxel_aggregation_utils, density_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.attention_utils import TransformerEncoder, get_positional_encoder


class PDVHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        layer_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        # for i, src_name in enumerate(self.pool_cfg.FEATURE_LOCATIONS):
        i = 0
        mlps = layer_cfg.MLPS
        for k in range(len(mlps)):
            mlps[k] = [self.model_cfg.ROI_GRID_POOL.NUM_FEATURES[i]] + mlps[k]
        stack_sa_module_msg = StackSAModuleMSGAttention if self.pool_cfg.get('ATTENTION', {}).get('ENABLED') else StackSAModuleMSG
        pool_layer = stack_sa_module_msg(
            radii=layer_cfg.POOL_RADIUS,
            nsamples=layer_cfg.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=layer_cfg.POOL_METHOD,
            use_density=layer_cfg.get('USE_DENSITY')
        )

        self.roi_grid_pool_layers.append(pool_layer)
        c_out += sum([x[-1] for x in mlps])

        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            assert self.pool_cfg.ATTENTION.NUM_FEATURES == c_out, f'ATTENTION.NUM_FEATURES must equal voxel aggregation output dimension of {c_out}.'
            pos_encoder = get_positional_encoder(self.pool_cfg)
            self.attention_head = TransformerEncoder(self.pool_cfg.ATTENTION, pos_encoder)

            # TODO: Check if this is necessary
            # Hack for xavier initialization (https://github.com/pashu123/Transformers/blob/master/train.py#L26-L29)
            for p in self.attention_head.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        # Density confidence predction
        if self.model_cfg.get('DENSITY_CONFIDENCE', {}).get('ENABLED'):
            self.cls_layers = self.make_fc_layers(
                input_channels=(3 +
                                self.model_cfg.DENSITY_CONFIDENCE.GRID_SIZE ** 3 +
                                (pre_channel if self.model_cfg.DENSITY_CONFIDENCE.ADD_SHARED_FEATURES else 0)),
                output_channels = self.num_class,
                fc_list=self.model_cfg.CLS_FC
            )
        else:
            self.cls_layers = self.make_fc_layers(
                input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
            )

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        batch_rois = batch_dict['rois']

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            batch_dict, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)

        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)
        new_xyz = global_roi_grid_points.view(-1, 3)

        pooled_features_list = []
        ball_idxs_list = []

        # for k, src_name in enumerate(self.pool_cfg.FEATURE_LOCATIONS):
        k = 0
        # point_coords = batch_dict['point_coords'][src_name]
        # point_features = batch_dict['point_features'][src_name]
        point_coords = batch_dict['fp_point_coords']
        point_features = batch_dict['fp_point_features']

        pool_layer = self.roi_grid_pool_layers[k]

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pool_output = pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            _, pooled_features, ball_idxs = pool_output
        else:
            _, pooled_features = pool_output

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        pooled_features_list.append(pooled_features)

        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            ball_idxs = ball_idxs.view(
                -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE **3,
                ball_idxs.shape[-1]
            )
            ball_idxs_list.append(ball_idxs)

        all_pooled_features = torch.cat(pooled_features_list, dim=-1)
        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            all_ball_idxs = torch.cat(ball_idxs_list, dim=-1)
        else:
            all_ball_idxs = []
        return all_pooled_features, global_roi_grid_points, local_roi_grid_points, all_ball_idxs

    def get_global_grid_points_of_roi(self, batch_dict, grid_size):
        rois = batch_dict['rois']
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)

        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_point_voxel_features(self, batch_dict):
        raise NotImplementedError

    def get_positional_input(self, points, rois, local_roi_grid_points):
        points_per_part = density_utils.find_num_points_per_part_multi(points,
                                                                       rois,
                                                                       self.model_cfg.ROI_GRID_POOL.GRID_SIZE,
                                                                       self.pool_cfg.ATTENTION.MAX_NUM_BOXES,
                                                                       return_centroid=self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density_centroid')
        points_per_part_num_features = 1 if len(points_per_part.shape) <= 5 else points_per_part.shape[-1]
        points_per_part = points_per_part.view(points_per_part.shape[0]*points_per_part.shape[1], -1, points_per_part_num_features).float()
        # First feature is density, other potential features are xyz
        points_per_part[..., 0] = torch.log10(points_per_part[..., 0] + 0.5) - (math.log10(0.5) if self.model_cfg.get('DENSITY_LOG_SHIFT') else 0)
        if self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'grid_points':
            positional_input = local_roi_grid_points
        elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density':
            positional_input = points_per_part
        elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density_grid_points':
            positional_input = torch.cat((local_roi_grid_points, points_per_part), dim=-1)
        else:
            positional_input = None
        return positional_input

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        # batch_dict['point_features'], batch_dict['point_coords'] = self.get_point_voxel_features(batch_dict)
        batch_dict['point_features'], batch_dict['point_coords'] = batch_dict['fp_point_features'], batch_dict['fp_point_coords']

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features, global_roi_grid_points, local_roi_grid_points, ball_idxs = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        batch_size_rcnn = pooled_features.shape[0]

        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            src_key_padding_mask = None
            if self.pool_cfg.ATTENTION.get('MASK_EMPTY_POINTS'):
                src_key_padding_mask = (ball_idxs == 0).all(-1)

            positional_input = self.get_positional_input(batch_dict['points'], batch_dict['rois'], local_roi_grid_points)
            # Attention
            attention_output = self.attention_head(pooled_features, positional_input, src_key_padding_mask) # (BxN, 6x6x6, C)

            if self.pool_cfg.ATTENTION.get('COMBINE'):
                attention_output = pooled_features + attention_output

            # Permute
            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            batch_size_rcnn = attention_output.shape[0]
            pooled_features = attention_output.permute(0, 2, 1).\
                contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size) # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if self.model_cfg.get('DENSITY_CONFIDENCE', {}).get('ENABLED'):
            with torch.no_grad():
                # Calculate number of points in each rcnn_reg
                _, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=None, box_preds=rcnn_reg
                )
                points_per_part = density_utils.find_num_points_per_part_multi(batch_dict['points'],
                                                                               batch_box_preds,
                                                                               self.model_cfg.DENSITY_CONFIDENCE.GRID_SIZE,
                                                                               self.model_cfg.DENSITY_CONFIDENCE.MAX_NUM_BOXES)
                points_per_part = torch.log10(points_per_part.float() + 0.5).reshape(-1, self.model_cfg.DENSITY_CONFIDENCE.GRID_SIZE ** 3, 1) - (math.log10(0.5) if self.model_cfg.get('DENSITY_LOG_SHIFT') else 0)
                point_cloud_range = torch.tensor(self.point_cloud_range, device=batch_box_preds.device)
                batch_box_preds_xyz = batch_box_preds.reshape(-1, batch_box_preds.shape[-1], 1)[:, :3]
                batch_box_preds_xyz /= (point_cloud_range[3:] - point_cloud_range[:3]).unsqueeze(0).unsqueeze(-1)

                density_features = [points_per_part, batch_box_preds_xyz]
                if self.model_cfg.DENSITY_CONFIDENCE.ADD_SHARED_FEATURES:
                    density_features.append(shared_features)

            density_features = torch.cat(density_features, dim=1)
            rcnn_cls = self.cls_layers(density_features)  # (B, 1 or 2)
        else:
            rcnn_cls = self.cls_layers(shared_features)

        rcnn_cls = rcnn_cls.transpose(1, 2).contiguous().squeeze(dim=1)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
