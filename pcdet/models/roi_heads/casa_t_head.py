import torch.nn as nn
import pdb
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F

from ...utils import common_utils
from .cascade_roi_head_template import CascadeRoIHeadTemplate
from ..model_utils.ctrans import *

from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from functools import partial

class CrossAttention(nn.Module):

    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.att = nn.MultiheadAttention(hidden_dim, 4)


    def forward(self, inputs, Q_in):

        Q = self.Q_linear(Q_in)
        K = self.K_linear(inputs)
        V = self.V_linear(inputs)

        out = self.att(Q, K, V)

        return out[0]


# multiatt shared
class CasA_T(CascadeRoIHeadTemplate):
    def __init__(self, input_channels, backbone_channels, model_cfg, voxel_size, point_cloud_range,num_frames=1, num_class=1):
        # notice backbone_channels is not uesd
        super().__init__(num_class=num_class, model_cfg=model_cfg,num_frames=num_frames)
        self.model_cfg = model_cfg

        self.up_dimension = MLP(input_dim = 28, hidden_dim = 64, output_dim = 256, num_layers = 3)

        num_queries = model_cfg.Transformer.num_queries
        hidden_dim = model_cfg.Transformer.hidden_dim
        self.num_points = model_cfg.Transformer.num_points

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = build_transformer(model_cfg.Transformer)
        self.aux_loss = model_cfg.Transformer.aux_loss

        self.shared_channel = hidden_dim

        self.stages = model_cfg.STAGES

        self.grid_offsets = self.model_cfg.PART.GRID_OFFSETS
        self.featmap_stride = self.model_cfg.PART.FEATMAP_STRIDE
        part_inchannel = self.model_cfg.PART.IN_CHANNEL
        self.num_parts = self.model_cfg.PART.SIZE ** 2

        pre_channel = self.model_cfg.SHARED_FC[-1] * 2
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
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        pre_channel = self.model_cfg.SHARED_FC[-1] * 2
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
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.conv_part = nn.Sequential(
            nn.Conv2d(part_inchannel, part_inchannel, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(part_inchannel, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(part_inchannel, self.num_parts, 1, 1, padding=0, bias=False),
        )
        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets,
                                   spatial_scale=1 / self.featmap_stride)

        self.cross_attention_layer = CrossAttention(self.shared_channel)

        self.init_weights()
        self.init_weights2(weight_init='xavier')

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    def init_weights2(self, weight_init='xavier'):
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

    def get_global_grid_points_of_roi(self, rois):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_corner_points(rois, batch_size_rcnn)  # (BxN, 2x2x2, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        # pdb.set_trace()

        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_corner_points(rois, batch_size_rcnn):
        faked_features = rois.new_ones((2, 2, 2))

        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 2x2x2, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 2x2x2, 3)
        return roi_grid_points
    
    def spherical_coordinate(self, src, diag_dist):
        assert (src.shape[-1] == 27)
        device = src.device
        indices_x = torch.LongTensor([0,3,6,9,12,15,18,21,24]).to(device)  #
        indices_y = torch.LongTensor([1,4,7,10,13,16,19,22,25]).to(device) # 
        indices_z = torch.LongTensor([2,5,8,11,14,17,20,23,26]).to(device) 
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / diag_dist
        src = torch.cat([dis, phi, the], dim = -1)
        return src

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )



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

            rois = batch_dict['rois']
            batch_size = batch_dict['batch_size']
            num_rois = batch_dict['rois'].shape[-2]

            part_scores = self.roi_part_pool(batch_dict, parts_feat)

            # corner
            corner_points, _ = self.get_global_grid_points_of_roi(rois)  # (BxN, 2x2x2, 3)
            corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])  # (B, N, 2x2x2, 3)

            num_sample = self.num_points
            src = rois.new_zeros(batch_size, num_rois, num_sample, 4)

            for bs_idx in range(batch_size):
                cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:5]
                cur_batch_boxes = batch_dict['rois'][bs_idx]
                cur_radiis = torch.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.2
                dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
                point_mask = (dis <= cur_radiis.unsqueeze(-1))
                for roi_box_idx in range(0, num_rois):
                    cur_roi_points = cur_points[point_mask[roi_box_idx]]

                    if cur_roi_points.shape[0] >= num_sample:
                        random.seed(0)
                        index = np.random.randint(cur_roi_points.shape[0], size=num_sample)
                        cur_roi_points_sample = cur_roi_points[index]

                    elif cur_roi_points.shape[0] == 0:
                        cur_roi_points_sample = cur_roi_points.new_zeros(num_sample, 4)

                    else:
                        empty_num = num_sample - cur_roi_points.shape[0]
                        add_zeros = cur_roi_points.new_zeros(empty_num, 4)
                        add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                        cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim = 0)

                    src[bs_idx, roi_box_idx, :, :] = cur_roi_points_sample

            src = src.view(batch_size * num_rois, -1, src.shape[-1])  # (b*128, 256, 4)


            corner_points = corner_points.view(batch_size * num_rois, -1)
            corner_add_center_points = torch.cat([corner_points, rois.view(-1, rois.shape[-1])[:,:3]], dim = -1)
            pos_fea = src[:,:,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1).repeat(1,num_sample,1)  # 27 维度
            lwh = rois.view(-1, rois.shape[-1])[:,3:6].unsqueeze(1).repeat(1,num_sample,1)
            diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
            pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))

            src = torch.cat([pos_fea, src[:,:,-1].unsqueeze(-1)], dim = -1)

            src = self.up_dimension(src)

            # Transformer
            pos = torch.zeros_like(src)
            hs = self.transformer(src, self.query_embed.weight, pos)[0]

            shared_features = hs.squeeze(2)  # 1,B,C
            all_shared_features.append(shared_features)
            pre_feat = torch.cat(all_shared_features, 0)

            cur_feat = self.cross_attention_layer(pre_feat, shared_features)

            cur_feat = torch.cat([cur_feat, shared_features], -1)
            cur_feat = cur_feat.squeeze(0)  # B, C*2

            rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(cur_feat))
            rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(cur_feat))

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

                self.forward_ret_dict['targets_dict' + stage_id] = targets_dict

            batch_dict['rois'] = batch_box_preds
            batch_dict['roi_scores'] = batch_cls_preds.squeeze(-1)

        if not self.training:
            batch_dict['batch_box_preds'] = torch.mean(torch.stack(all_preds), 0)
            batch_dict['batch_cls_preds'] = torch.mean(torch.stack(all_scores), 0)

        return batch_dict

