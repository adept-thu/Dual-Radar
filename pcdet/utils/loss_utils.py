import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    多分类
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha # 0.25
        self.gamma = gamma # 2.0

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor. eg:(4, 321408, 3)
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor. eg:(4, 321408, 3)
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor. eg:(4, 321408)
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        #cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M] 
        pred_sigmoid = torch.sigmoid(input) # (batch_size, 321408, 3) f(x) = 1 / (1 + e^(-x))
        # print("target:",target.shape,"  pred_sigmoid:",pred_sigmoid.shape)
        # print(pred_sigmoid.shape)
        # print(target.shape)
        # 这里的加权主要是解决正负样本不均衡的问题:正样本的权重为0.25，负样本的权重为0.75
        # 交叉熵来自KL散度，衡量两个分布之间的相似性，针对二分类问题：
        # 合并形式: L = -(y * log(y^) + (1 - y) * log(1 - y^)) <-->
        # 分段形式:y = 1, L = -y * log(y^); y = 0, L = -(1 - y) * log(1 - y^)
        # 这两种形式等价，只要是0和1的分类问题均可以写成两种等价形式，针对focal loss做类似处理
        # 相对熵 = 信息熵 + 交叉熵， 且交叉熵是凸函数，求导时能够得到全局最优值-->(sigma(s)- y)x
        # https://zhuanlan.zhihu.com/p/35709485
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()
        # weights参数使用正anchor数目进行平均，使得每个样本的损失与样本中目标的数量无关
        return loss * weights

class WeightedClassificationLoss(nn.Module):
    def __init__(self):
        super(WeightedClassificationLoss, self).__init__()
        
    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights=None,  reduction='none'):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        
        if weights is not None:                
            if weights.shape.__len__() == 2 or \
                    (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
                weights = weights.unsqueeze(-1)

            assert weights.shape.__len__() == bce_loss.shape.__len__()
            
            loss = weights * bce_loss
        else:
            loss = bce_loss

        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            loss = loss.sum(dim=-1)
        elif reduction == 'mean':
            loss = loss.mean(dim=-1)
        return loss   

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta # 默认值1/9=0.111
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32) # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.code_weights = torch.from_numpy(self.code_weights).cuda() # 将权重放到GPU上
 
    @staticmethod
    def smooth_l1_loss(diff, beta):
        # 如果beta非常小，则直接用abs计算，否则按照正常的Smooth L1 Loss计算
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff) # (batch_size, 321408, 7) 输出绝对值
            # smoothL1公式，如上面所示 --> (batch_size, 321408, 7)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        #loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M] anchor_head_template中用到此块的代码
        #box_preds_sin:(batch_size, 321408, 7)  reg_targets_sin:(batch_size, 321408, 7)  reg_weights:(batch_size, 321408) 初始正样本权值为1
        # 如果target为nan,则等于input,否则等于target
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target # (batch_size, 321408, 7)
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1) #(batch_size, 321408, 7) 乘以box每一项的权重

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            # weights参数使用正anchor数目进行平均，使得每个样本的损失与样本中目标的数量无关
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss

class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').mean(dim=-1) * weights
        return loss

class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)

class WeightedCrossEntropyLoss_ver1(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        
    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor,  reduction='none'):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        loss = self.sigmoid_cross_entropy_with_logits(input, target)

        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            loss = loss.sum(dim=-1)
        elif reduction == 'mean':
            loss = loss.mean(dim=-1)
        return loss      

class PointSASALoss(nn.Module):
    """
    Layer-wise point segmentation loss, used for SASA.
    """
    def __init__(self,
                 func: str = 'BCE',
                 layer_weights: list = None,
                 extra_width: list = None,
                 set_ignore_flag: bool = False):
        super(PointSASALoss, self).__init__()

        self.layer_weights = layer_weights
        if func == 'BCE':
            self.loss_func = WeightedBinaryCrossEntropyLoss()
        elif func == 'Focal':
            self.loss_func = SigmoidFocalClassificationLoss()
        else:
            raise NotImplementedError

        assert not set_ignore_flag or (set_ignore_flag and extra_width is not None)
        self.extra_width = extra_width
        self.set_ignore_flag = set_ignore_flag
    
    def assign_target(self, points, gt_boxes):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, \
            'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1]) \
            if self.extra_width is not None else gt_boxes
        
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()

        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())

            if not self.set_ignore_flag:
                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                box_fg_flag = (box_idxs_of_pts >= 0)

            else:
                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                box_fg_flag = (box_idxs_of_pts >= 0)

                extend_box_idx_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                ignore_flag = box_fg_flag ^ (extend_box_idx_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            point_cls_labels_single[box_fg_flag] = 1
            point_cls_labels[bs_mask] = point_cls_labels_single
        
        return point_cls_labels # (N, ) 0: bg, 1: fg, -1: ignore

    def forward(self, l_points, l_scores, gt_boxes):
        """
        Args:
            l_points: List of points, [(N, 4): bs_idx, x, y, z]
            l_scores: List of points, [(N, 1): predicted point scores]
            gt_boxes: (B, M, 8)
        Returns:
            l_labels: List of labels: [(N, 1): assigned segmentation labels]
        """
        l_labels = []
        for i in range(len(self.layer_weights)):
            li_scores = l_scores[i]
            if li_scores is None or self.layer_weights[i] == 0:
                l_labels.append(None)
                continue
            # binary segmentation labels: 0: bg, 1: fg, -1: ignore
            li_labels = self.assign_target(l_points[i], gt_boxes)
            l_labels.append(li_labels)

        return l_labels

    def loss_forward(self, l_scores, l_labels):
        """
        Args:
            l_scores: List of points, [(N, 1): predicted point scores]
            l_labels: List of points, [(N, 1): assigned segmentation labels]
        Returns:
            l_loss: List of segmentation loss
        """
        l_loss = []
        for i in range(len(self.layer_weights)):
            li_scores, li_labels = l_scores[i], l_labels[i]
            if li_scores is None or li_labels is None:
                l_loss.append(None)
                continue

            positives, negatives = li_labels > 0, li_labels == 0
            cls_weights = positives * 1.0 + negatives * 1.0 # (N, 1)
            pos_normalizer = cls_weights.sum(dim=0).float()

            one_hot_targets = li_scores.new_zeros(
                *list(li_labels.shape), 2
            )
            one_hot_targets.scatter_(-1, (li_labels > 0).long().unsqueeze(-1), 1.0)
            one_hot_targets = one_hot_targets[:, 1:] # (N, 1)

            li_loss = self.loss_func(li_scores[None],
                                     one_hot_targets[None],
                                     cls_weights.reshape(1, -1))
            li_loss = self.layer_weights[i] * li_loss.sum() / torch.clamp(
                pos_normalizer, min=1.0)
            l_loss.append(li_loss)

        return l_loss

def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss

class SigmoidQualityFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
        """
        super(SigmoidQualityFocalClassificationLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        # alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        func = F.binary_cross_entropy_with_logits
        # negatives are supervised by 0 quality score
        pred_sigmoid = torch.sigmoid(input)
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(input.shape)
        loss = func(input, zerolabel, reduction='none') * scale_factor.pow(self.gamma)

        # positives are supervised by bbox quality (IoU) score
        positives = target > 0
        scale_factor = target[positives] - pred_sigmoid[positives]
        loss[positives] = func(input[positives], target[positives], reduction='none') * scale_factor.abs().pow(self.gamma)

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights * self.alpha