import numpy as np
import torch
import torch.nn as nn

from ...utils import loss_utils
from .anchor_head_template import AnchorHeadTemplate
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner_add_gt import AxisAlignedTargetAssigner

import pdb


class AnchorHeadRDIoU(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )


        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_target_assigner(self, anchor_target_cfg, sec = False):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT,
                sec = sec
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidQualityFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )
    
    def get_rdiou(self, bboxes1, bboxes2):
        x1u, y1u, z1u = bboxes1[:,:,0], bboxes1[:,:,1], bboxes1[:,:,2]
        l1, w1, h1 =  torch.exp(bboxes1[:,:,3]), torch.exp(bboxes1[:,:,4]), torch.exp(bboxes1[:,:,5])
        t1 = torch.sin(bboxes1[:,:,6]) * torch.cos(bboxes2[:,:,6])
        x2u, y2u, z2u = bboxes2[:,:,0], bboxes2[:,:,1], bboxes2[:,:,2]
        l2, w2, h2 =  torch.exp(bboxes2[:,:,3]), torch.exp(bboxes2[:,:,4]), torch.exp(bboxes2[:,:,5])
        t2 = torch.cos(bboxes1[:,:,6]) * torch.sin(bboxes2[:,:,6])

        # we emperically scale the y/z to make their predictions more sensitive.
        x1 = x1u
        y1 = y1u * 2
        z1 = z1u * 2
        x2 = x2u
        y2 = y2u * 2
        z2 = z2u * 2

        # clamp is necessray to aviod inf.
        l1, w1, h1 = torch.clamp(l1, max=10), torch.clamp(w1, max=10), torch.clamp(h1, max=10)
        j1, j2 = torch.ones_like(h2), torch.ones_like(h2)

        volume_1 = l1 * w1 * h1 * j1
        volume_2 = l2 * w2 * h2 * j2

        inter_l = torch.max(x1 - l1 / 2, x2 - l2 / 2)
        inter_r = torch.min(x1 + l1 / 2, x2 + l2 / 2)
        inter_t = torch.max(y1 - w1 / 2, y2 - w2 / 2)
        inter_b = torch.min(y1 + w1 / 2, y2 + w2 / 2)
        inter_u = torch.max(z1 - h1 / 2, z2 - h2 / 2)
        inter_d = torch.min(z1 + h1 / 2, z2 + h2 / 2)
        inter_m = torch.max(t1 - j1 / 2, t2 - j2 / 2)
        inter_n = torch.min(t1 + j1 / 2, t2 + j2 / 2)

        inter_volume = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0) \
            * torch.clamp((inter_d - inter_u),min=0) * torch.clamp((inter_n - inter_m),min=0)
        
        c_l = torch.min(x1 - l1 / 2,x2 - l2 / 2)
        c_r = torch.max(x1 + l1 / 2,x2 + l2 / 2)
        c_t = torch.min(y1 - w1 / 2,y2 - w2 / 2)
        c_b = torch.max(y1 + w1 / 2,y2 + w2 / 2)
        c_u = torch.min(z1 - h1 / 2,z2 - h2 / 2)
        c_d = torch.max(z1 + h1 / 2,z2 + h2 / 2)
        c_m = torch.min(t1 - j1 / 2,t2 - j2 / 2)
        c_n = torch.max(t1 + j1 / 2,t2 + j2 / 2)

        inter_diag = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 + (t2 - t1)**2
        c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2 + torch.clamp((c_d - c_u),min=0)**2  + torch.clamp((c_n - c_m),min=0)**2

        union = volume_1 + volume_2 - inter_volume
        u = (inter_diag) / c_diag
        rdiou = inter_volume / union
        return u, rdiou

    def get_clsreg_targets(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels'].clone()
        batch_size = int(box_preds.shape[0])
        h = box_preds.shape[1]
        w = box_preds.shape[2]

        # enlarge the positive samples
        box_cls_labels = box_cls_labels.view(batch_size, h, w, -1)
        box_cls_labels_an0 = box_cls_labels[:,:,:,0].unsqueeze(-1)
        box_cls_labels_an1 = box_cls_labels[:,:,:,1].unsqueeze(-1)
        box_cls_labels_an0_tmp1 = box_cls_labels_an0.roll(shifts = 1, dims = 1)
        box_cls_labels_an0_tmp2 = box_cls_labels_an0.roll(shifts = -1, dims = 1)
        box_cls_labels_an1_tmp1 = box_cls_labels_an1.roll(shifts = 1, dims = 2)
        box_cls_labels_an1_tmp2 = box_cls_labels_an1.roll(shifts = -1, dims = 2)
        box_cls_labels_an1[box_cls_labels_an1_tmp1==1] = 1
        box_cls_labels_an1[box_cls_labels_an1_tmp2==1] = 1
        box_cls_labels_an0[box_cls_labels_an0_tmp1==1] = 1
        box_cls_labels_an0[box_cls_labels_an0_tmp2==1] = 1

        box_cls_labels = torch.cat([box_cls_labels_an0, box_cls_labels_an1], dim = -1)
        re_box_cls_labels = box_cls_labels.view(batch_size, -1)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        _, rdiou =  self.get_rdiou(box_preds, box_reg_targets)

        # filter the background.
        with torch.no_grad():
            rdiou_guided_cls_labels = re_box_cls_labels * rdiou.detach()
        return re_box_cls_labels, rdiou_guided_cls_labels


    def get_rdiou_guided_reg_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.re_box_cls_labels
        batch_size = int(box_preds.shape[0])
 
        box_cls_labels = box_cls_labels.view(batch_size, -1)
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        u, rdiou = self.get_rdiou(box_preds, box_reg_targets)

        rdiou_loss_n = rdiou - u
        rdiou_loss_n = torch.clamp(rdiou_loss_n,min=-1.0,max = 1.0)
        rdiou_loss_m = 1 - rdiou_loss_n
        rdiou_loss_src = rdiou_loss_m * reg_weights
        rdiou_loss = rdiou_loss_src.sum() / batch_size
        rdiou_loss = rdiou_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = rdiou_loss
        tb_dict = {
            'rpn_loss_loc': rdiou_loss.item()
        }


        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=reg_weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    

    def get_rdiou_guided_cls_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.rdiou_guided_cls_labels
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        cls_loss_src = self.cls_loss_func(cls_preds, cls_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict



    def get_loss(self):
        self.re_box_cls_labels, self.rdiou_guided_cls_labels = self.get_clsreg_targets()
        box_loss, tb_dict = self.get_rdiou_guided_reg_loss()
        cls_loss, tb_dict_cls = self.get_rdiou_guided_cls_loss()
        tb_dict.update(tb_dict_cls)

        rpn_loss = cls_loss + box_loss

        if rpn_loss.isnan():
            print(cls_loss)
            print(box_loss)
            pdb.set_trace()

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict


    def assign_targets_sec(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner_sec.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        box_preds = self.conv_box(spatial_features_2d)
        cls_preds = self.conv_cls(spatial_features_2d) 

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = torch.clamp(box_preds, max = 3.0)
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds


        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

