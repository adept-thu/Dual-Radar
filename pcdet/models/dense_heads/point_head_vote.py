import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...utils import box_coder_utils, box_utils, common_utils, loss_utils
from .point_head_template import PointHeadTemplate


class PointHeadVote(PointHeadTemplate):
    """
    A simple vote-based detection head, which is used for 3DSSD.
    Reference Paper: https://arxiv.org/abs/2002.10187
    3DSSD: Point-based 3D Single Stage Object Detector
    """
    def __init__(self, num_class, input_channels, fp_input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        use_bn = self.model_cfg.USE_BN
        self.predict_boxes_when_training = predict_boxes_when_training

        self.vote_cfg = self.model_cfg.VOTE_CONFIG
        self.vote_layers = self.make_fc_layers(
            input_channels=input_channels,
            output_channels=3,
            fc_list=self.vote_cfg.VOTE_FC
        )

        self.sa_cfg = self.model_cfg.SA_CONFIG
        channel_in, channel_out = input_channels, 0

        mlps = self.sa_cfg.MLPS.copy()
        for idx in range(mlps.__len__()):
            mlps[idx] = [channel_in] + mlps[idx]
            channel_out += mlps[idx][-1]

        self.SA_module = pointnet2_modules.PointnetSAModuleFSMSG(
            radii=self.sa_cfg.RADIUS,
            nsamples=self.sa_cfg.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            bn=use_bn
        )

        channel_in = channel_out
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(channel_in, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            channel_in = self.model_cfg.SHARED_FC[k]

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        channel_in = self.model_cfg.SHARED_FC[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=num_class if not self.model_cfg.LOSS_CONFIG.LOSS_CLS == 'CrossEntropy' else num_class + 1,
            fc_list=self.model_cfg.CLS_FC
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size,
            fc_list=self.model_cfg.REG_FC
        )

        self.fp_cls_layers = self.make_fc_layers(
            fc_list=self.model_cfg.FP_CLS_FC,
            input_channels=fp_input_channels,
            output_channels=num_class
        )
        self.fp_part_reg_layers = self.make_fc_layers(
            fc_list=self.model_cfg.PART_FC,
            input_channels=fp_input_channels,
            output_channels=3
        )
        self.fp_part_reg_image_layers = self.make_fc_layers(
            fc_list=self.model_cfg.PART_FC,
            input_channels=fp_input_channels,
            output_channels=3
        )

        self.segmentation_loss_func = nn.CrossEntropyLoss(ignore_index=255)
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

    def build_losses(self, losses_cfg):
        # classification loss
        if losses_cfg.LOSS_CLS.startswith('WeightedBinaryCrossEntropy'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedBinaryCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS == 'WeightedCrossEntropy':
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS == 'FocalLoss':
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidFocalClassificationLoss(
                    **losses_cfg.get('LOSS_CLS_CONFIG', {})
                )
            )
        else:
            raise NotImplementedError

        # regression loss
        if losses_cfg.LOSS_REG == 'WeightedSmoothL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
                    **losses_cfg.get('LOSS_REG_CONFIG', {})
                )
            )
        elif losses_cfg.LOSS_REG == 'WeightedL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
                )
            )
        else:
            raise NotImplementedError

        # sasa loss
        loss_sasa_cfg = losses_cfg.get('LOSS_SASA_CONFIG', None)
        if loss_sasa_cfg is not None:
            self.enable_sasa = True
            self.add_module(
                'loss_point_sasa',
                loss_utils.PointSASALoss(**loss_sasa_cfg)
            )
        else:
            self.enable_sasa = False

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def assign_stack_targets_simple(self, points, gt_boxes, extend_gt_boxes=None, set_ignore_flag=True):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: (B, M, 8), required if set ignore flag
            set_ignore_flag:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignore
            point_reg_labels: (N1 + N2 + N3 + ..., 3), corresponding object centroid
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert not set_ignore_flag or extend_gt_boxes is not None
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((points.shape[0], 3))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            if extend_gt_boxes is not None:
                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idx_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[box_fg_flag]]
            point_cls_labels_single[box_fg_flag] = 1
            point_cls_labels[bs_mask] = point_cls_labels_single

            point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), 3))
            point_reg_labels_single[box_fg_flag] = gt_box_of_fg_points[:, 0:3]
            point_reg_labels[bs_mask] = point_reg_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
        }
        return targets_dict

    def assign_targets_simple(self, points, gt_boxes, extra_width=None, set_ignore_flag=True):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extra_width: (dx, dy, dz) extra width applied to gt boxes
            assign_method: binary or distance
            set_ignore_flag:
        Returns:
            point_vote_labels: (N1 + N2 + N3 + ..., 3)
        """
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert points.shape.__len__() in [2], 'points.shape=%s' % str(points.shape)
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1]) \
            if extra_width is not None else gt_boxes
        if set_ignore_flag:
            targets_dict = self.assign_stack_targets_simple(points=points, gt_boxes=gt_boxes,
                                                            extend_gt_boxes=extend_gt_boxes,
                                                            set_ignore_flag=set_ignore_flag)
        else:
            targets_dict = self.assign_stack_targets_simple(points=points, gt_boxes=extend_gt_boxes,
                                                            set_ignore_flag=set_ignore_flag)
        return targets_dict

    def assign_stack_targets_mask(self, points, gt_boxes, extend_gt_boxes=None,
                                  set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            set_ignore_flag:
            use_ball_constraint:
            central_radius:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_reg_labels: (N1 + N2 + N3 + ..., code_size)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = gt_boxes.new_zeros(points.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((points.shape[0], self.box_coder.code_size))
        point_box_labels = gt_boxes.new_zeros((points.shape[0], gt_boxes.size(2) - 1))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
                ignore_flag = fg_flag ^ box_fg_flag
                point_cls_labels_single[ignore_flag] = -1
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if gt_box_of_fg_points.shape[0] > 0:
                point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), self.box_coder.code_size))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_reg_labels_single[fg_flag] = fg_point_box_labels
                point_reg_labels[bs_mask] = point_reg_labels_single

                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), gt_boxes.size(2) - 1))
                point_box_labels_single[fg_flag] = gt_box_of_fg_points[:, :-1]
                point_box_labels[bs_mask] = point_box_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
            'point_box_labels': point_box_labels
        }
        return targets_dict

    def assign_stack_targets_iou(self, points, pred_boxes, gt_boxes,
                                 pos_iou_threshold=0.5, neg_iou_threshold=0.35):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            pred_boxes: (N, 7/8)
            gt_boxes: (B, M, 8)
            pos_iou_threshold:
            neg_iou_threshold:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_reg_labels: (N1 + N2 + N3 + ..., code_size)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(pred_boxes.shape) == 2 and pred_boxes.shape[1] >= 7, 'pred_boxes.shape=%s' % str(pred_boxes.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = gt_boxes.new_zeros(pred_boxes.shape[0]).long()
        point_reg_labels = gt_boxes.new_zeros((pred_boxes.shape[0], self.box_coder.code_size))
        point_box_labels = gt_boxes.new_zeros((pred_boxes.shape[0], 7))
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            pred_boxes_single = pred_boxes[bs_mask]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            pred_boxes_iou = iou3d_nms_utils.boxes_iou3d_gpu(
                pred_boxes_single,
                gt_boxes[k][:, :7]
            )
            pred_boxes_iou, box_idxs_of_pts = torch.max(pred_boxes_iou, dim=-1)
            fg_flag = pred_boxes_iou > pos_iou_threshold
            ignore_flag = (pred_boxes_iou > neg_iou_threshold) ^ fg_flag
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels_single[ignore_flag] = -1
            point_cls_labels[bs_mask] = point_cls_labels_single

            if gt_box_of_fg_points.shape[0] > 0:
                point_reg_labels_single = point_reg_labels.new_zeros((bs_mask.sum(), self.box_coder.code_size))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_reg_labels_single[fg_flag] = fg_point_box_labels
                point_reg_labels[bs_mask] = point_reg_labels_single

                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 7))
                point_box_labels_single[fg_flag] = gt_box_of_fg_points[:, :-1]
                point_box_labels[bs_mask] = point_box_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_reg_labels': point_reg_labels,
            'point_box_labels': point_box_labels
        }
        return targets_dict

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        assign_method = self.model_cfg.TARGET_CONFIG.ASSIGN_METHOD  # mask or iou
        if assign_method == 'mask':
            points = input_dict['point_vote_coords']
            gt_boxes = input_dict['gt_boxes']
            assert points.shape.__len__() == 2, 'points.shape=%s' % str(points.shape)
            assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            central_radius = self.model_cfg.TARGET_CONFIG.get('GT_CENTRAL_RADIUS', 2.0)
            targets_dict = self.assign_stack_targets_mask(
                points=points, gt_boxes=gt_boxes,
                set_ignore_flag=False, use_ball_constraint=True, central_radius=central_radius
            )
        elif assign_method == 'iou':
            points = input_dict['point_vote_coords']
            pred_boxes = input_dict['point_box_preds']
            gt_boxes = input_dict['gt_boxes']
            assert points.shape.__len__() == 2, 'points.shape=%s' % str(points.shape)
            assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert pred_boxes.shape.__len__() == 2, 'pred_boxes.shape=%s' % str(pred_boxes.shape)
            pos_iou_threshold = self.model_cfg.TARGET_CONFIG.POS_IOU_THRESHOLD
            neg_iou_threshold = self.model_cfg.TARGET_CONFIG.NEG_IOU_THRESHOLD
            targets_dict = self.assign_stack_targets_iou(
                points=points, pred_boxes=pred_boxes, gt_boxes=gt_boxes,
                pos_iou_threshold=pos_iou_threshold, neg_iou_threshold=neg_iou_threshold
            )
        else:
            raise NotImplementedError

        targets_dict['segmentation_label'] = input_dict['segmentation_label']
        return targets_dict

    def assign_targets_fp(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['fp_point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.PART_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=True, ret_box_labels=False
        )

        return targets_dict

    def get_vote_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['vote_cls_labels'] > 0
        vote_reg_labels = self.forward_ret_dict['vote_reg_labels']
        vote_reg_preds = self.forward_ret_dict['point_vote_coords']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        vote_loss_reg_src = self.reg_loss_func(
            vote_reg_preds[None, ...],
            vote_reg_labels[None, ...],
            weights=reg_weights[None, ...])
        vote_loss_reg = vote_loss_reg_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        vote_loss_reg = vote_loss_reg * loss_weights_dict['vote_reg_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss_reg': vote_loss_reg.item()})
        return vote_loss_reg, tb_dict

    @torch.no_grad()
    def generate_centerness_label(self, point_base, point_box_labels, pos_mask, epsilon=1e-6):
        """
        Args:
            point_base: (N1 + N2 + N3 + ..., 3)
            point_box_labels: (N1 + N2 + N3 + ..., 7)
            pos_mask: (N1 + N2 + N3 + ...)
            epsilon:
        Returns:
            centerness_label: (N1 + N2 + N3 + ...)
        """
        centerness = point_box_labels.new_zeros(pos_mask.shape)

        point_box_labels = point_box_labels[pos_mask, :]
        canonical_xyz = point_base[pos_mask, :] - point_box_labels[:, :3]
        rys = point_box_labels[:, -1]
        canonical_xyz = common_utils.rotate_points_along_z(
            canonical_xyz.unsqueeze(dim=1), -rys
        ).squeeze(dim=1)

        distance_front = point_box_labels[:, 3] / 2 - canonical_xyz[:, 0]
        distance_back = point_box_labels[:, 3] / 2 + canonical_xyz[:, 0]
        distance_left = point_box_labels[:, 4] / 2 - canonical_xyz[:, 1]
        distance_right = point_box_labels[:, 4] / 2 + canonical_xyz[:, 1]
        distance_top = point_box_labels[:, 5] / 2 - canonical_xyz[:, 2]
        distance_bottom = point_box_labels[:, 5] / 2 + canonical_xyz[:, 2]

        centerness_l = torch.min(distance_front, distance_back) / torch.max(distance_front, distance_back)
        centerness_w = torch.min(distance_left, distance_right) / torch.max(distance_left, distance_right)
        centerness_h = torch.min(distance_top, distance_bottom) / torch.max(distance_top, distance_bottom)
        centerness_pos = torch.clamp(centerness_l * centerness_w * centerness_h, min=epsilon) ** (1 / 3.0)

        centerness[pos_mask] = centerness_pos

        return centerness

    def get_axis_aligned_iou_loss_lidar(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Args:
            pred_boxes: (N, 7) float Tensor.
            gt_boxes: (N, 7) float Tensor.
        Returns:
            iou_loss: (N) float Tensor.
        """
        assert pred_boxes.shape[0] == gt_boxes.shape[0]

        pos_p, len_p, *cps = torch.split(pred_boxes, 3, dim=-1)
        pos_g, len_g, *cgs = torch.split(gt_boxes, 3, dim=-1)

        len_p = torch.clamp(len_p, min=1e-5)
        len_g = torch.clamp(len_g, min=1e-5)
        vol_p = len_p.prod(dim=-1)
        vol_g = len_g.prod(dim=-1)

        min_p, max_p = pos_p - len_p / 2, pos_p + len_p / 2
        min_g, max_g = pos_g - len_g / 2, pos_g + len_g / 2

        min_max = torch.min(max_p, max_g)
        max_min = torch.max(min_p, min_g)
        diff = torch.clamp(min_max - max_min, min=0)
        intersection = diff.prod(dim=-1)
        union = vol_p + vol_g - intersection
        iou_axis_aligned = intersection / torch.clamp(union, min=1e-5)

        iou_loss = 1 - iou_axis_aligned
        return iou_loss

    def get_corner_loss_lidar(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Args:
            pred_boxes: (N, 7) float Tensor.
            gt_boxes: (N, 7) float Tensor.
        Returns:
            corner_loss: (N) float Tensor.
        """
        assert pred_boxes.shape[0] == gt_boxes.shape[0]

        pred_box_corners = box_utils.boxes_to_corners_3d(pred_boxes)
        gt_box_corners = box_utils.boxes_to_corners_3d(gt_boxes)

        gt_boxes_flip = gt_boxes.clone()
        gt_boxes_flip[:, 6] += np.pi
        gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_boxes_flip)
        # (N, 8, 3)
        corner_loss = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(pred_box_corners - gt_box_corners, 1.0)
        corner_loss_flip = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(pred_box_corners - gt_box_corners_flip, 1.0)
        corner_loss = torch.min(corner_loss.sum(dim=2), corner_loss_flip.sum(dim=2))

        return corner_loss.mean(dim=1)

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        positives = point_cls_labels > 0
        negatives = point_cls_labels == 0
        cls_weights = positives * 1.0 + negatives * 1.0

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        self.forward_ret_dict['point_cls_labels_onehot'] = one_hot_targets

        loss_cfgs = self.model_cfg.LOSS_CONFIG
        if 'WithCenterness' in loss_cfgs.LOSS_CLS:
            point_base = self.forward_ret_dict['point_vote_coords']
            point_box_labels = self.forward_ret_dict['point_box_labels']
            centerness_label = self.generate_centerness_label(point_base, point_box_labels, positives)
            
            loss_cls_cfg = loss_cfgs.get('LOSS_CLS_CONFIG', None)
            centerness_min = loss_cls_cfg['centerness_min'] if loss_cls_cfg is not None else 0.0
            centerness_max = loss_cls_cfg['centerness_max'] if loss_cls_cfg is not None else 1.0
            centerness_label = centerness_min + (centerness_max - centerness_min) * centerness_label
            
            one_hot_targets *= centerness_label.unsqueeze(dim=-1)

        point_loss_cls = self.cls_loss_func(point_cls_preds, one_hot_targets[..., 1:], weights=cls_weights)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_pos_num': positives.sum().item()
        })
        return point_loss_cls, cls_weights, tb_dict  # point_loss_cls: (N)

    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_reg_preds = self.forward_ret_dict['point_reg_preds']
        point_reg_labels = self.forward_ret_dict['point_reg_labels']

        reg_weights = pos_mask.float()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        if tb_dict is None:
            tb_dict = {}

        point_loss_offset_reg = self.reg_loss_func(
            point_reg_preds[None, :, :6],
            point_reg_labels[None, :, :6],
            weights=reg_weights[None, ...]
        )
        point_loss_offset_reg = point_loss_offset_reg.sum(dim=-1).squeeze()

        if hasattr(self.box_coder, 'pred_velo') and self.box_coder.pred_velo:
            point_loss_velo_reg = self.reg_loss_func(
                point_reg_preds[None, :, 6 + 2 * self.box_coder.angle_bin_num:8 + 2 * self.box_coder.angle_bin_num],
                point_reg_labels[None, :, 6 + 2 * self.box_coder.angle_bin_num:8 + 2 * self.box_coder.angle_bin_num],
                weights=reg_weights[None, ...]
            )
            point_loss_velo_reg = point_loss_velo_reg.sum(dim=-1).squeeze()
            point_loss_offset_reg = point_loss_offset_reg + point_loss_velo_reg

        point_loss_offset_reg *= loss_weights_dict['point_offset_reg_weight']

        if isinstance(self.box_coder, box_coder_utils.PointBinResidualCoder):
            point_angle_cls_labels = \
                point_reg_labels[:, 6:6 + self.box_coder.angle_bin_num]
            point_loss_angle_cls = F.cross_entropy(  # angle bin cls
                point_reg_preds[:, 6:6 + self.box_coder.angle_bin_num],
                point_angle_cls_labels.argmax(dim=-1), reduction='none') * reg_weights

            point_angle_reg_preds = point_reg_preds[:, 6 + self.box_coder.angle_bin_num:6 + 2 * self.box_coder.angle_bin_num]
            point_angle_reg_labels = point_reg_labels[:, 6 + self.box_coder.angle_bin_num:6 + 2 * self.box_coder.angle_bin_num]
            point_angle_reg_preds = (point_angle_reg_preds * point_angle_cls_labels).sum(dim=-1, keepdim=True)
            point_angle_reg_labels = (point_angle_reg_labels * point_angle_cls_labels).sum(dim=-1, keepdim=True)
            point_loss_angle_reg = self.reg_loss_func(
                point_angle_reg_preds[None, ...],
                point_angle_reg_labels[None, ...],
                weights=reg_weights[None, ...]
            )
            point_loss_angle_reg = point_loss_angle_reg.squeeze()

            point_loss_angle_cls *= loss_weights_dict['point_angle_cls_weight']
            point_loss_angle_reg *= loss_weights_dict['point_angle_reg_weight']

            point_loss_box = point_loss_offset_reg + point_loss_angle_cls + point_loss_angle_reg  # (N)
        else:
            point_angle_reg_preds = point_reg_preds[:, 6:]
            point_angle_reg_labels = point_reg_labels[:, 6:]
            point_loss_angle_reg = self.reg_loss_func(
                point_angle_reg_preds[None, ...],
                point_angle_reg_labels[None, ...],
                weights=reg_weights[None, ...]
            )
            point_loss_angle_reg *= loss_weights_dict['point_angle_reg_weight']
            point_loss_box = point_loss_offset_reg + point_loss_angle_reg

        if reg_weights.sum() > 0:
            point_box_preds = self.forward_ret_dict['point_box_preds']
            point_box_labels = self.forward_ret_dict['point_box_labels']
            point_loss_box_aux = 0

            if self.model_cfg.LOSS_CONFIG.get('AXIS_ALIGNED_IOU_LOSS_REGULARIZATION', False):
                point_loss_iou = self.get_axis_aligned_iou_loss_lidar(
                    point_box_preds[pos_mask, :],
                    point_box_labels[pos_mask, :]
                )
                point_loss_iou *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_iou_weight']
                point_loss_box_aux = point_loss_box_aux + point_loss_iou

            if self.model_cfg.LOSS_CONFIG.get('CORNER_LOSS_REGULARIZATION', False):
                point_loss_corner = self.get_corner_loss_lidar(
                    point_box_preds[pos_mask, 0:7],
                    point_box_labels[pos_mask, 0:7]
                )
                point_loss_corner *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['point_corner_weight']
                point_loss_box_aux = point_loss_box_aux + point_loss_corner
            
            point_loss_box[pos_mask] = point_loss_box[pos_mask] + point_loss_box_aux

        return point_loss_box, reg_weights, tb_dict  # point_loss_box: (N)

    def get_sasa_layer_loss(self, tb_dict=None):
        if self.enable_sasa:
            point_loss_sasa_list = self.loss_point_sasa.loss_forward(
                self.forward_ret_dict['point_sasa_preds'],
                self.forward_ret_dict['point_sasa_labels']
            )
            point_loss_sasa = 0
            tb_dict = dict()
            for i in range(len(point_loss_sasa_list)):
                cur_point_loss_sasa = point_loss_sasa_list[i]
                if cur_point_loss_sasa is None:
                    continue
                point_loss_sasa = point_loss_sasa + cur_point_loss_sasa
                tb_dict['point_loss_sasa_layer_%d' % i] = point_loss_sasa_list[i].item()
            tb_dict['point_loss_sasa'] = point_loss_sasa.item()
            return point_loss_sasa, tb_dict
        else:
            return None, None

    def get_segmentation_loss(self, tb_dict=None):
        x = self.forward_ret_dict['segmentation_preds']
        target = self.forward_ret_dict['segmentation_label'].long()
        # segmentation_loss = nn.functional.cross_entropy(x, target)
        
        # print('#', x.min(), x.max(), target.min(), target.max())
        segmentation_loss = self.segmentation_loss_func(x, target)
        # print("# seg loss", segmentation_loss)
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'segmentation_loss': segmentation_loss.item()})
        return segmentation_loss, tb_dict

    def get_fp_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['fp_point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['fp_point_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 15.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        # point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        point_loss_cls = point_loss_cls * loss_weights_dict.get('fp_point_cls_weight', 1.0)
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'fp_point_loss_cls': point_loss_cls.item(),
            'fp_point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_fp_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['fp_point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['fp_point_part_labels']
        point_part_preds = self.forward_ret_dict['fp_point_part_preds']
        # import pdb;pdb.set_trace()
        # point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        # point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        point_loss_part_src = self.reg_loss_func(
            point_part_preds[None, ...], point_part_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_part = point_loss_part_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict.get('fp_point_part_weight', 1.0)
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'fp_point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_fp_part_image_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['fp_point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['fp_point_part_labels']
        point_part_preds = self.forward_ret_dict['fp_point_part_image_preds']
        # point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        # point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        point_loss_part_src = self.reg_loss_func(
            point_part_preds[None, ...], point_part_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_part = point_loss_part_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict.get('fp_point_part_image_weight', 1.0)
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'fp_point_loss_part_image': point_loss_part.item()})
        return point_loss_part, tb_dict


    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_vote, tb_dict_0 = self.get_vote_layer_loss()

        point_loss_cls, cls_weights, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, box_weights, tb_dict_2 = self.get_box_layer_loss()
        segmentation_loss, tb_dict_seg = self.get_segmentation_loss()
        fp_point_loss_cls, tb_dict_fp_cls = self.get_fp_cls_layer_loss()
        fp_point_loss_part, tb_dict_fp_part = self.get_fp_part_layer_loss()
        fp_point_loss_part_image, tb_dict_fp_part_image = self.get_fp_part_image_layer_loss()

        point_loss_cls = point_loss_cls.sum() / torch.clamp(cls_weights.sum(), min=1.0)
        point_loss_box = point_loss_box.sum() / torch.clamp(box_weights.sum(), min=1.0)
        tb_dict.update({
            'point_loss_vote': point_loss_vote.item(),
            'point_loss_cls': point_loss_cls.item(),
            'point_loss_box': point_loss_box.item()
        })

        point_loss = point_loss_vote + point_loss_cls + point_loss_box + segmentation_loss + fp_point_loss_cls + fp_point_loss_part + fp_point_loss_part_image
        tb_dict.update(tb_dict_0)
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        tb_dict.update(tb_dict_seg)
        tb_dict.update(tb_dict_fp_cls)
        tb_dict.update(tb_dict_fp_part)
        tb_dict.update(tb_dict_fp_part_image)

        point_loss_sasa, tb_dict_3 = self.get_sasa_layer_loss()
        if point_loss_sasa is not None:
            tb_dict.update(tb_dict_3)
            point_loss += point_loss_sasa
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_scores (optional): (B, N)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        batch_size = batch_dict['batch_size']

        fp_point_features = batch_dict['fp_point_features']
        fp_point_image_features = batch_dict['fp_point_image_features']
        fp_point_coords = batch_dict['fp_point_coords']
        fp_batch_idx, fp_point_coords = fp_point_coords[:, 0], fp_point_coords[:, 1:4]
        fp_point_coords = fp_point_coords.view(batch_size, -1, 3).contiguous()
        # import pdb;pdb.set_trace()
        fp_point_features = fp_point_features.reshape(
            batch_size, fp_point_coords.size(1), -1
        ).permute(0, 2, 1).contiguous()

        # import pdb;pdb.set_trace()
        if self.training:
            fp_point_image_features = fp_point_image_features.reshape(
                batch_size, fp_point_coords.size(1), -1
            ).permute(0, 2, 1).contiguous() # (bs, c, n)
        else:
            _bs, _, _h, _w = batch_dict['segmentation_preds'].shape
            fp_point_image_features = fp_point_image_features.reshape(
                batch_size, _h*_w, -1
            ).permute(0, 2, 1).contiguous() # (bs, c, hxw)


        fp_point_cls_preds = self.fp_cls_layers(fp_point_features)  # (total_points, num_class)
        fp_point_part_preds = self.fp_part_reg_layers(fp_point_features)
        fp_point_part_image_preds = self.fp_part_reg_image_layers(fp_point_image_features)

        fp_point_cls_preds = fp_point_cls_preds.permute(0, 2, 1).contiguous()
        fp_point_cls_preds = fp_point_cls_preds.view(-1, fp_point_cls_preds.shape[-1]).contiguous()
        fp_point_part_preds = fp_point_part_preds.permute(0, 2, 1).contiguous()
        fp_point_part_preds = fp_point_part_preds.view(-1, fp_point_part_preds.shape[-1]).contiguous()
        fp_point_part_image_preds = fp_point_part_image_preds.permute(0, 2, 1).contiguous()# (bs, n, 3)
        fp_point_part_image_preds = fp_point_part_image_preds.view(-1, fp_point_part_image_preds.shape[-1]).contiguous()# (bs*n, 3)

        if not self.training:
            fp_point_part_image_preds = fp_point_part_image_preds.view(_bs,_h,_w,3)
            fp_point_part_image_preds = fp_point_part_image_preds.permute(0,3,1,2).contiguous()
            batch_dict['part_image_preds'] = fp_point_part_image_preds #(bs,3,h,w)


        ret_dict = {
            'batch_size': batch_size,
            'fp_point_cls_preds': fp_point_cls_preds,
            'fp_point_part_preds': fp_point_part_preds,
            'fp_point_part_image_preds': fp_point_part_image_preds
        }

        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        batch_idx, point_coords = point_coords[:, 0], point_coords[:, 1:4]
        batch_idx = batch_idx.view(batch_size, -1, 1)
        point_coords = point_coords.view(batch_size, -1, 3).contiguous()
        point_features = point_features.reshape(
            batch_size,
            point_coords.size(1),
            -1
        ).permute(0, 2, 1).contiguous()

        # candidate points sampling
        sample_range = self.model_cfg.SAMPLE_RANGE
        sample_batch_idx = batch_idx[:, sample_range[0]:sample_range[1], :].contiguous()

        candidate_coords = point_coords[:, sample_range[0]:sample_range[1], :].contiguous()
        candidate_features = point_features[:, :, sample_range[0]:sample_range[1]].contiguous()

        # generate vote points
        vote_offsets = self.vote_layers(candidate_features)  # (B, 3, N)
        vote_translation_range = np.array(self.vote_cfg.MAX_TRANSLATION_RANGE, dtype=np.float32)
        vote_translation_range = torch.from_numpy(vote_translation_range).cuda().unsqueeze(dim=0).unsqueeze(dim=-1)
        vote_offsets = torch.max(vote_offsets, -vote_translation_range)
        vote_offsets = torch.min(vote_offsets, vote_translation_range)
        vote_coords = candidate_coords + vote_offsets.permute(0, 2, 1).contiguous()

        # ret_dict = {'batch_size': batch_size,
        #             'point_candidate_coords': candidate_coords.view(-1, 3).contiguous(),
        #             'point_vote_coords': vote_coords.view(-1, 3).contiguous()}
        ret_dict['point_candidate_coords'] = candidate_coords.view(-1, 3).contiguous()
        ret_dict['point_vote_coords'] = vote_coords.view(-1, 3).contiguous()

        sample_batch_idx_flatten = sample_batch_idx.view(-1, 1).contiguous()  # (N, 1)
        batch_dict['batch_index'] = sample_batch_idx_flatten.squeeze(-1)
        batch_dict['point_candidate_coords'] = torch.cat(  # (N, 4)
            (sample_batch_idx_flatten, ret_dict['point_candidate_coords']), dim=-1)
        batch_dict['point_vote_coords'] = torch.cat(  # (N, 4)
            (sample_batch_idx_flatten, ret_dict['point_vote_coords']), dim=-1)

        if self.training:  # assign targets for vote loss
            extra_width = self.model_cfg.TARGET_CONFIG.get('VOTE_EXTRA_WIDTH', None)
            targets_dict = self.assign_targets_simple(batch_dict['point_candidate_coords'],
                                                      batch_dict['gt_boxes'],
                                                      extra_width=extra_width,
                                                      set_ignore_flag=False)
            ret_dict['vote_cls_labels'] = targets_dict['point_cls_labels']  # (N)
            ret_dict['vote_reg_labels'] = targets_dict['point_reg_labels']  # (N, 3)

        _, point_features, _ = self.SA_module(
            point_coords,
            point_features,
            new_xyz=vote_coords
        )

        # import pdb;pdb.set_trace()
        point_features = self.shared_fc_layer(point_features)
        point_cls_preds = self.cls_layers(point_features)
        point_reg_preds = self.reg_layers(point_features)

        point_cls_preds = point_cls_preds.permute(0, 2, 1).contiguous()
        point_cls_preds = point_cls_preds.view(-1, point_cls_preds.shape[-1]).contiguous()
        point_reg_preds = point_reg_preds.permute(0, 2, 1).contiguous()
        point_reg_preds = point_reg_preds.view(-1, point_reg_preds.shape[-1]).contiguous()

        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'] = point_cls_scores

        point_box_preds = self.box_coder.decode_torch(point_reg_preds,
                                                      ret_dict['point_vote_coords'])
        batch_dict['point_box_preds'] = point_box_preds

        ret_dict.update({'point_cls_preds': point_cls_preds,
                         'point_reg_preds': point_reg_preds,
                         'point_box_preds': point_box_preds,
                         'point_cls_scores': point_cls_scores,
                         'segmentation_preds': batch_dict['segmentation_preds']
                         })

        if self.training:
            # get cls and part label for fp_points
            targets_dict_fp = self.assign_targets_fp(batch_dict)
            ret_dict['fp_point_cls_labels'] = targets_dict_fp['point_cls_labels']
            ret_dict['fp_point_part_labels'] = targets_dict_fp['point_part_labels']

            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_reg_labels'] = targets_dict['point_reg_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            ret_dict['segmentation_label'] = targets_dict['segmentation_label']

            if self.enable_sasa:
                point_sasa_labels = self.loss_point_sasa(
                    batch_dict['point_coords_list'],
                    batch_dict['point_scores_list'],
                    batch_dict['gt_boxes']
                )
                ret_dict.update({
                    'point_sasa_preds': batch_dict['point_scores_list'],
                    'point_sasa_labels': point_sasa_labels
                })

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_vote_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_reg_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict
