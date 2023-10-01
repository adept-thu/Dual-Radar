import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
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
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        """
            分类loss:Focal Loss for Classification
            为了解决样本中前景和背景类的不平衡，作者引入focal loss，分类损失具有以下形式：
            FL(pt)=−αt(1−pt)^γlog(pt)
            其中 pt是模型的估计概率，α 和 γ 是focal loss的参数。在训练过程中使用 α = 0.25 和 γ = 2。
        """
        """
        all_targets_dict = {
            'box_cls_labels': cls_labels, # (batch_size，321408）
            'box_reg_targets': bbox_targets, # (batch_size，321408,7）
            'reg_weights': reg_weights # (batch_size，321408）
            321408 = 248 * 216 * 2 *3 3为3个类别，骑车，自行车，行人 因为是单独分开去处理的 在axis_aligned_target_assigner.py里
        }
        return all_targets_dict
        """
        # (batch_size, 248, 216, 18) 网络类别预测
        cls_preds = self.forward_ret_dict['cls_preds']
        # (batch_size, 321408) 前景anchor类别 321408 = 248 * 216 * 2 *3 3为3个类别，骑车，自行车，行人 因为是单独分开去处理的 在axis_aligned_target_assigner.py里
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        # print(cls_preds.shape)
        # print(box_cls_labels.shape)
        batch_size = int(cls_preds.shape[0])

        # print("=============")
        # print("预测部分的batch_size为",batch_size)


        # [batch_szie, num_anchors]--> (batch_size, 321408)
        # 关心的anchor  选取出前景背景anchor, 在0.45到0.6之间的设置为仍然是-1，不参与loss计算
        cared = box_cls_labels >= 0  # [N, num_anchors]
        # (batch_size, 321408) 前景anchor
        positives = box_cls_labels > 0
        # (batch_size, 321408) 背景anchor
        negatives = box_cls_labels == 0
        # 背景anchor赋予权重
        negative_cls_weights = negatives * 1.0
        # 将每个anchor分类(前景+背景)的损失权重都设置为1，在0.45到0.6之间的设置为仍然是-1，不参与loss计算
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        # 每个正样本anchor的回归损失权重，设置为1，其他为0，回归损失loss只计算正样本，其他均不参与计算
        reg_weights = positives.float()
        # 如果只有一类
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        # 正则化并计算权重     求出每个数据中有多少个正例，即shape=（batch， 1）
        pos_normalizer = positives.sum(1, keepdim=True).float()
        # 正则化回归损失-->(batch_size, 321408)，最小值为1,根据论文中所述，用(正样本数量)来正则化回归损失
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # 正则化分类损失-->(batch_size, 321408)，根据论文中所述，用（正样本+负样本）数量来正则化分类损失
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        # care包含了背景和前景的anchor，但是这里只需要得到前景部分的类别即可不关注-1和0
        # cared.type_as(box_cls_labels) 将cared中为False的那部分不需要计算loss的anchor变成了0
        # 对应位置相乘后，所有背景和iou介于match_threshold和unmatch_threshold之间的anchor都设置为0
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels) # (batch_size, 321408) 所有背景和iou介于match_threshold和unmatch_threshold之间的anchor都设置为0
        # 在最后一个维度扩展一次
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_targets = cls_targets.squeeze(dim=-1)# (batch_size, 321408) 所有背景和iou介于match_threshold和unmatch_threshold之间的anchor都设置为0
        
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        # (batch_size, 321408, 4)，这里的类别数+1是考虑背景 #cls_preds : (batch_size, 248, 216, 18) 网络类别预测

        # target.scatter(dim, index, src)
        # scatter_函数的一个典型应用就是在分类问题中，
        # 将目标标签转换为one-hot编码形式 https://blog.csdn.net/guofei_fly/article/details/104308528
        # 这里表示在最后一个维度，将cls_targets.unsqueeze(dim=-1)所索引的位置设置为1

        """
                    dim=1: 表示按照列进行填充
                    index=batch_data.label:表示把batch_data.label里面的元素值作为下标，
                    去下标对应位置(这里的"对应位置"解释为列，如果dim=0，那就解释为行)进行填充
                    src=1:表示填充的元素值为1
        """
        # (batch_size, 321408, 4) #cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        #cls_preds : (batch_size, 248, 216, 18) 网络类别预测 (batch_size, 248, 216, 18) --> (batch_size, 321408, 3)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        # (batch_size, 321408, 3) 不计算背景分类损失
        one_hot_targets = one_hot_targets[..., 1:]
        # print(cls_preds.shape)
        # print(one_hot_targets.shape)

        # 计算分类损失 # [N, M] # (batch_size, 321408, 3) cls_weights:# 正则化分类损失-->(batch_size, 321408)，根据论文中所述，用（正样本+负样本）数量来正则化分类损失
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M] 
        # 求和并除以batch数目
        cls_loss = cls_loss_src.sum() / batch_size
        # loss乘以分类权重 --> cls_weight=1.0
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6): #box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        # 针对角度添加sin损失，有效防止-pi和pi方向相反时损失过大       # sin(a - b) = sinacosb-cosasinb
        assert dim != -1
        # (batch_size, 321408, 1)  torch.sin() - torch.cos() 的 input (Tensor) 都是弧度制数据，不是角度制数据。
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        # (batch_size, 321408, 1)
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        # (batch_size, 321408, 7) 将编码后的结果放回
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        # (batch_size, 321408, 7) 将编码后的结果放回
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        """
            在VoxelNet中，一个3D BBox被建模为一个7维向量表示，分别为(x_c,y_c,z_c,l,w,h,θ)，训练过程中，对这7个变量采用Smooth L1损失进行回归训练。
            当同一个3D检测框的预测方向恰好与真实方向相反的时候，上述的7个变量的前6个变量的回归损失较小，而最后一个方向的回归损失会很大，这其实并不利
            于模型训练。为了解决这个问题，作者引入角度回归的正弦误差损失，定义如下：
                L_θ = SmoothL1(sin(θ_p − θ_t))
            θ_p为预测的方向角，θ_t为真实的方向角。那么当θ_p与θ_t相差π的时候，该损失趋向于0，这样更利于模型训练。
            那这样的话，模型预测方向很可能与真实方向相差180度，为了解决这种损失将具有相反方向的框视为相同的问题，在 RPN 的输出中添加了一个简单的方向
            分类器，该方向分类器使用 softmax 损失函数。围绕z轴的偏航旋转高于0，则结果为正；否则为负数。
        """
        # (batch_size, 248, 216, 42） anchor_box的7个回归参数
        box_preds = self.forward_ret_dict['box_preds']
        # (batch_size, 248, 216, 12） anchor_box的方向预测
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        # (batch_size, 321408, 7) 每个anchor和GT编码的结果
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        # (batch_size, 321408)
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0]) #batch_size

        # 获取所有anchor中属于前景anchor的mask  shape : (batch_size, 321408)
        positives = box_cls_labels > 0
        # 设置回归参数为1.    [True, False] * 1. = [1., 0.]  只考虑前景的，其余均不管
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float() # (batch_size, 1) 所有正例的和 eg:[[162.],[166.],[155.],[108.]]
        # 正则化回归损失-->(batch_size, 321408)，最小值为1,根据论文中所述，用(正样本数量)来正则化回归损失
        reg_weights /= torch.clamp(pos_normalizer, min=1.0) # (batch_size, 321408)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        # (1, 248*216, 7） --> (batch_size, 248*216, 7）
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # (batch_size, 248*216*6, 7） box_preds： (batch_size, 248, 216, 42） anchor_box的7个回归参数
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        # (batch_size, 321408, 7)  box_reg_targets:(batch_size, 321408, 7) 每个anchor和GT编码的结果
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight'] # loc_weight = 2.0 损失乘以回归权重
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        # 如果存在方向预测，则添加方向损失
        if box_dir_cls_preds is not None:
            # (batch_size, 321408, 2)  box_reg_targets:(batch_size, 321408, 7) 每个anchor和GT编码的结果
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET, # 方向偏移量 0.78539 = π/4 
                num_bins=self.model_cfg.NUM_DIR_BINS # BINS的方向数 = 2
            )
            # 方向预测值 (batch_size, 321408, 2)
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            # 只要正样本的方向预测值 (batch_size, 321408) positives_shape : (batch_size, 321408)
            weights = positives.type_as(dir_logits)
            # 正则化回归损失-->(batch_size, 321408)，最小值为1,根据论文中所述，用(正样本数量)来正则化回归损失
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            # 方向损失计算
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            # 损失权重，dir_weight: 0.2
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            # 将方向损失加入box损失
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict
#减去dir_offset(45度)的原因可以参考这个issue:
# https://github.com/open-mmlab/OpenPCDet/issues/80
# 说的呢就是因为大部分目标都集中在0度和180度，270度和90度，
# 这将导致网络在从这些角度预测对象时不断摆动。所以为了解决这个问题，<details><summary>*<font color='gray'>[En]</font>*</summary>*<font color='gray'>This will cause the network to wobble constantly in the prediction of objects from these angles. So in order to solve this problem,</font>*</details>
# 将方向分类的角度判断减去45度再进行判断，如下图所示。
#         这里减掉45度之后，在预测推理的时候，同样预测的角度解码之后
# 也要减去45度再进行之后测nms等操作。

    def get_loss(self):
        """
            总损失函数最终形式如下：
                L^total = β^1 L^cls + ​β^2 (L^reg-θ + L^reg-other) + ​β^3 L^dir
                其中L^cls是分类损失,L^reg-other是位置和维度的回归损失,L^reg-θ是新的角度损失,L^dir是方向分类损失
                β^1 = 1.0、 β^2 = 2.0和 β^3 = 0.2 是损失公式的常数系数，使用相对较小的 β^3 值来避免网络难以识别物体方向的情况。
        """
        # 计算classifiction layer的loss，tb_dict内容和cls_loss相同，形式不同，一个是torch.tensor一个是字典值
        cls_loss, tb_dict = self.get_cls_layer_loss()
        """
            cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            tb_dict = {
                'rpn_loss_cls': cls_loss.item()
            }
            return cls_loss, tb_dict
        """
        # 计算regression layer的loss
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        # 在tb_dict中添加tb_dict_box，在python的字典中添加值，如果添加的也是字典，用updae方法，如果是键值对则采用赋值的方式
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        # 在tb_dict中添加rpn_loss，此时tb_dict中包含cls_loss,reg_loss和rpn_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
