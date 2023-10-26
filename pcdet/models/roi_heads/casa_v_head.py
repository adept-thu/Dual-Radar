import torch
import torch.nn as nn
from .cascade_roi_head_template import CascadeRoIHeadTemplate
from ...utils import common_utils, spconv_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from functools import partial
import pickle
from ..model_utils.ctrans import *

# part multi-att no shared
class CasA_V_V1(CascadeRoIHeadTemplate):
    def __init__(self, input_channels, backbone_channels, model_cfg, point_cloud_range=None, voxel_size=None, num_frames=1, num_class=1,
                 **kwargs):
        super().__init__(num_class=num_class, num_frames=num_frames, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.stages = model_cfg.STAGES

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        self.shared_fc_layers = nn.ModuleList()

        for i in range(self.stages):
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layers.append(nn.Sequential(*shared_fc_list))

        self.shared_channel = pre_channel

        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()

        for i in range(self.stages):
            pre_channel = self.model_cfg.SHARED_FC[-1]*2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers.append(cls_fc_layers)

            pre_channel = self.model_cfg.SHARED_FC[-1]*2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers.append(reg_fc_layers)

        self.grid_offsets = self.model_cfg.PART.GRID_OFFSETS
        self.featmap_stride = self.model_cfg.PART.FEATMAP_STRIDE
        part_inchannel = self.model_cfg.PART.IN_CHANNEL
        self.num_parts = self.model_cfg.PART.SIZE**2

        self.conv_part = nn.Sequential(
            nn.Conv2d(part_inchannel, part_inchannel, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(part_inchannel, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(part_inchannel, self.num_parts, 1, 1, padding=0, bias=False),
        )
        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets,
                                   spatial_scale=1 / self.featmap_stride)

        self.cross_attention_layers = nn.ModuleList()
        for i in range(self.stages):
            self.cross_attention_layers.append(CrossAttention(self.shared_channel))

        self.init_weights()


    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [ self.cls_layers, self.reg_layers]:
            for stage_module in module_list:
                for m in stage_module.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        for module_list in [self.cls_layers, self.reg_layers]:
            for stage_module in module_list:
                nn.init.normal_(stage_module[-1].weight, 0, 0.01)
                nn.init.constant_(stage_module[-1].bias, 0)
        for m in self.shared_fc_layers.modules():
            if isinstance(m, nn.Linear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def obtain_conf_preds(self, confi_im, anchors):

        confi = []

        for i, im in enumerate(confi_im):
            boxes = anchors[i]
            im = confi_im[i]
            if len(boxes) == 0:
                confi.append(torch.empty(0).type_as(im))
            else:
                (xs, ys) = self.gen_grid_fn(boxes)
                out = bilinear_interpolate_torch_gridsample(im, xs, ys)
                x = torch.mean(out, 0).view(-1, 1)
                confi.append(x)

        confi = torch.cat(confi)


        return confi

    def roi_part_pool(self, batch_dict, parts_feat):
        rois = batch_dict['rois'].clone()
        confi_preds = self.obtain_conf_preds(parts_feat, rois)

        return confi_preds

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
        rois = batch_dict['rois'].clone()

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

                if with_vf_transform:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
                else:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

            if src_name=='points_bev':
                point_coords = batch_dict['point_coords']
                point_features = batch_dict['point_features']
                xyz = point_coords[:, 1:4]
                xyz_batch_cnt = xyz.new_zeros(batch_size).int()
                xyz_batch_idx = point_coords[:, 0]
                for k in range(batch_size):
                    xyz_batch_cnt[k] = (xyz_batch_idx == k).sum()
                cur_sp_tensors = batch_dict['multi_scale_3d_features']['x_conv4']
                cur_stride = batch_dict['multi_scale_3d_strides']['x_conv4']

                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()

                spatial_shape = cur_sp_tensors.spatial_shape

                new_indexs = point_coords.new_zeros(point_coords.shape)
                new_indexs[:, 0] = point_coords[:, 0]
                new_indexs[:, 1] = (point_coords[:, 3] - self.point_cloud_range[2]) // self.voxel_size[2]
                new_indexs[:, 2] = (point_coords[:, 2] - self.point_cloud_range[1]) // self.voxel_size[1]
                new_indexs[:, 3] = (point_coords[:, 1] - self.point_cloud_range[0]) // self.voxel_size[0]
                new_indexs[:, 1:] = new_indexs[:, 1:] // cur_stride

                h,w,l = spatial_shape
                new_indexs[:, 1]=torch.clamp(new_indexs[:, 1], 0, h-1)
                new_indexs[:, 2] = torch.clamp(new_indexs[:, 2], 0, w - 1)
                new_indexs[:, 3] = torch.clamp(new_indexs[:, 3], 0, l - 1)

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds2(batch_size,spatial_shape,new_indexs)

                pooled_features = pool_layer(
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=point_features,
                    voxel2point_indices=v2p_ind_tensor
                )
                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
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

    def get_gts_rois(self,batch_dict):
        rois = batch_dict['rois'] # 2*700*7
        roi_scores = batch_dict['roi_scores'] #2*700
        roi_labels = batch_dict['roi_labels'] #2*700

        gt_boxes = batch_dict['gt_boxes'] #2*34*10

        rois = torch.cat([rois,gt_boxes[...,:7]],1)

        new_scores = gt_boxes[...,-1].clone()
        new_scores[new_scores>0] = 100.

        roi_scores = torch.cat([roi_scores,new_scores],1)

        roi_labels = torch.cat([roi_labels,gt_boxes[...,-1].long()],1)

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels

        return batch_dict

    def forward(self, batch_dict):

        targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )

        #if self.training:
        #    batch_dict = self.get_gts_rois(batch_dict)
        feat_2d = batch_dict['st_features_2d']

        parts_feat = self.conv_part(feat_2d)

        all_preds = []
        all_scores =[]

        all_shared_features = []

        for i in range(self.stages):

            stage_id = str(i)

            if self.training:

                targets_dict = self.assign_targets(batch_dict, i)
                batch_dict['rois'] = targets_dict['rois']

                batch_dict['roi_labels'] = targets_dict['roi_labels']

            pooled_features = self.roi_grid_pool(batch_dict)

            part_scores = self.roi_part_pool(batch_dict, parts_feat)

            pooled_features = pooled_features.view(pooled_features.size(0), -1)

            shared_features = self.shared_fc_layers[i](pooled_features)
            shared_features = shared_features.unsqueeze(0)  # 1,B,C
            all_shared_features.append(shared_features)
            pre_feat = torch.cat(all_shared_features, 0)

            cur_feat = self.cross_attention_layers[i](pre_feat, shared_features)

            cur_feat = torch.cat([cur_feat, shared_features], -1)
            cur_feat = cur_feat.squeeze(0) # B, C*2

            rcnn_cls = self.cls_layers[i](cur_feat)
            rcnn_reg = self.reg_layers[i](cur_feat)

            rcnn_cls = part_scores + rcnn_cls

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            if not self.training:
                all_preds.append(batch_box_preds)
                all_scores.append(batch_cls_preds)
            else:
                targets_dict['rcnn_cls'] = rcnn_cls
                targets_dict['rcnn_reg'] = rcnn_reg

                self.forward_ret_dict['targets_dict'+stage_id] = targets_dict

            batch_dict['rois'] = batch_box_preds
            batch_dict['roi_scores'] = batch_cls_preds.squeeze(-1)

        if not self.training:
            batch_dict['batch_box_preds'] = torch.mean(torch.stack(all_preds), 0)
            batch_dict['batch_cls_preds'] = torch.mean(torch.stack(all_scores), 0)

        return batch_dict


# part multi-att att no shared
class CasA_V(CascadeRoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range=None, voxel_size=None, num_frames=1, num_class=1,
                 **kwargs):
        super().__init__(num_class=num_class, num_frames=num_frames, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.stages = model_cfg.STAGES

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        self.shared_fc_layers = nn.ModuleList()

        for i in range(self.stages):
            pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
            shared_fc_list = []
            for k in range(0, self.model_cfg.SHARED_FC.__len__()):
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel = self.model_cfg.SHARED_FC[k]

                if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layers.append(nn.Sequential(*shared_fc_list))
            break

        self.shared_channel = pre_channel

        self.cls_layers = nn.ModuleList()
        self.reg_layers = nn.ModuleList()

        for i in range(self.stages):
            pre_channel = self.model_cfg.SHARED_FC[-1]*2
            cls_fc_list = []
            for k in range(0, self.model_cfg.CLS_FC.__len__()):
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.CLS_FC[k]

                if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias=True))
            cls_fc_layers = nn.Sequential(*cls_fc_list)
            self.cls_layers.append(cls_fc_layers)

            pre_channel = self.model_cfg.SHARED_FC[-1]*2
            reg_fc_list = []
            for k in range(0, self.model_cfg.REG_FC.__len__()):
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                    nn.ReLU()
                ])
                pre_channel = self.model_cfg.REG_FC[k]

                if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True))
            reg_fc_layers = nn.Sequential(*reg_fc_list)
            self.reg_layers.append(reg_fc_layers)
            break

        self.grid_offsets = self.model_cfg.PART.GRID_OFFSETS
        self.featmap_stride = self.model_cfg.PART.FEATMAP_STRIDE
        part_inchannel = self.model_cfg.PART.IN_CHANNEL
        self.num_parts = self.model_cfg.PART.SIZE**2

        self.conv_part = nn.Sequential(
            nn.Conv2d(part_inchannel, part_inchannel, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(part_inchannel, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(part_inchannel, self.num_parts, 1, 1, padding=0, bias=False),
        )
        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets,
                                   spatial_scale=1 / self.featmap_stride)

        self.cross_attention_layers = nn.ModuleList()
        for i in range(self.stages):
            self.cross_attention_layers.append(CrossAttention(self.shared_channel))

        self.init_weights()


    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [ self.cls_layers, self.reg_layers]:
            for stage_module in module_list:
                for m in stage_module.modules():
                    if isinstance(m, nn.Linear):
                        init_func(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        for module_list in [self.cls_layers, self.reg_layers]:
            for stage_module in module_list:
                nn.init.normal_(stage_module[-1].weight, 0, 0.01)
                nn.init.constant_(stage_module[-1].bias, 0)
        for m in self.shared_fc_layers.modules():
            if isinstance(m, nn.Linear):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def obtain_conf_preds(self, confi_im, anchors):

        confi = []

        for i, im in enumerate(confi_im):
            boxes = anchors[i]
            im = confi_im[i]
            if len(boxes) == 0:
                confi.append(torch.empty(0).type_as(im))
            else:
                (xs, ys) = self.gen_grid_fn(boxes)
                out = bilinear_interpolate_torch_gridsample(im, xs, ys)
                x = torch.mean(out, 0).view(-1, 1)
                confi.append(x)

        confi = torch.cat(confi)


        return confi

    def roi_part_pool(self, batch_dict, parts_feat):
        rois = batch_dict['rois'].clone()
        confi_preds = self.obtain_conf_preds(parts_feat, rois)

        return confi_preds

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
        rois = batch_dict['rois'].clone()

        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

                if with_vf_transform:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
                else:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                # get voxel2point tensor

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()
                # voxel neighbor aggregation
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz.contiguous(),
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=cur_sp_tensors.features.contiguous(),
                    voxel2point_indices=v2p_ind_tensor
                )

                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

            if src_name=='points_bev':
                point_coords = batch_dict['point_coords']
                point_features = batch_dict['point_features']
                xyz = point_coords[:, 1:4]
                xyz_batch_cnt = xyz.new_zeros(batch_size).int()
                xyz_batch_idx = point_coords[:, 0]
                for k in range(batch_size):
                    xyz_batch_cnt[k] = (xyz_batch_idx == k).sum()
                cur_sp_tensors = batch_dict['multi_scale_3d_features']['x_conv4']
                cur_stride = batch_dict['multi_scale_3d_strides']['x_conv4']

                cur_roi_grid_coords = roi_grid_coords // cur_stride
                cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
                cur_roi_grid_coords = cur_roi_grid_coords.int()

                spatial_shape = cur_sp_tensors.spatial_shape

                new_indexs = point_coords.new_zeros(point_coords.shape)
                new_indexs[:, 0] = point_coords[:, 0]
                new_indexs[:, 1] = (point_coords[:, 3] - self.point_cloud_range[2]) // self.voxel_size[2]
                new_indexs[:, 2] = (point_coords[:, 2] - self.point_cloud_range[1]) // self.voxel_size[1]
                new_indexs[:, 3] = (point_coords[:, 1] - self.point_cloud_range[0]) // self.voxel_size[0]
                new_indexs[:, 1:] = new_indexs[:, 1:] // cur_stride

                h,w,l = spatial_shape
                new_indexs[:, 1]=torch.clamp(new_indexs[:, 1], 0, h-1)
                new_indexs[:, 2] = torch.clamp(new_indexs[:, 2], 0, w - 1)
                new_indexs[:, 3] = torch.clamp(new_indexs[:, 3], 0, l - 1)

                v2p_ind_tensor = spconv_utils.generate_voxel2pinds2(batch_size,spatial_shape,new_indexs)

                pooled_features = pool_layer(
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                    features=point_features,
                    voxel2point_indices=v2p_ind_tensor
                )
                pooled_features = pooled_features.view(
                    -1, self.pool_cfg.GRID_SIZE ** 3,
                    pooled_features.shape[-1]
                )  # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)

        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
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

    def get_gts_rois(self,batch_dict):
        rois = batch_dict['rois'] # 2*700*7
        roi_scores = batch_dict['roi_scores'] #2*700
        roi_labels = batch_dict['roi_labels'] #2*700

        gt_boxes = batch_dict['gt_boxes'] #2*34*10

        rois = torch.cat([rois,gt_boxes[...,:7]],1)

        new_scores = gt_boxes[...,-1].clone()
        new_scores[new_scores>0] = 100.

        roi_scores = torch.cat([roi_scores,new_scores],1)

        roi_labels = torch.cat([roi_labels,gt_boxes[...,-1].long()],1)

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels

        return batch_dict

    def forward(self, batch_dict):


        targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )

        #if self.training:
        #    batch_dict = self.get_gts_rois(batch_dict)
        feat_2d = batch_dict['st_features_2d']

        parts_feat = self.conv_part(feat_2d)

        all_preds = []
        all_scores = []
        all_rois = []


        all_shared_features = []

        for i in range(self.stages):

            stage_id = str(i)

            if self.training:

                targets_dict = self.assign_targets(batch_dict, i)

                batch_dict['rois'] = targets_dict['rois']

                batch_dict['roi_labels'] = targets_dict['roi_labels']

            pooled_features = self.roi_grid_pool(batch_dict)

            part_scores = self.roi_part_pool(batch_dict, parts_feat)

            pooled_features = pooled_features.view(pooled_features.size(0), -1)

            shared_features = self.shared_fc_layers[0](pooled_features)
            shared_features = shared_features.unsqueeze(0)  # 1,B,C
            all_shared_features.append(shared_features)
            pre_feat = torch.cat(all_shared_features, 0)

            cur_feat = self.cross_attention_layers[i](pre_feat, shared_features)

            cur_feat = torch.cat([cur_feat, shared_features], -1)
            cur_feat = cur_feat.squeeze(0) # B, C*2

            rcnn_cls = self.cls_layers[0](cur_feat)
            rcnn_reg = self.reg_layers[0](cur_feat)

            rcnn_cls = part_scores + rcnn_cls

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            if not self.training :
                all_preds.append(batch_box_preds)
                all_scores.append(batch_cls_preds)
            else:
                targets_dict['rcnn_cls'] = rcnn_cls
                targets_dict['rcnn_reg'] = rcnn_reg

                self.forward_ret_dict['targets_dict'+stage_id] = targets_dict

            all_rois.append(batch_box_preds)

            batch_dict['rois'] = batch_box_preds
            batch_dict['roi_scores'] = batch_cls_preds.squeeze(-1)

        if not self.training:
            batch_dict['batch_box_preds'] = torch.mean(torch.stack(all_preds), 0)
            batch_dict['batch_cls_preds'] = torch.mean(torch.stack(all_scores), 0)

        return batch_dict





