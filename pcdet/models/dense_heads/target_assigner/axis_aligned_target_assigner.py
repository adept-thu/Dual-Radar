import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils
#处理一批数据中所有点云的anchors和gt_boxes，分类别计算每个anchor属于前景还是背景，为每个前景的anchor分配类别和计算box的回归残差和回归权重
#assign_targets完成对一帧点云数据中所有的类别和anchor的正负样本分配，
#assign_targets_single完成对一帧中每个类别的GT和anchor的正负样本分配。
#一个Batch样本中anchor与GT的匹配这里是逐帧逐类别进行的。与图像目标检测中稍有不同。

class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()
        # anchor生成配置参数 此参数包括类别，anchor_sizes，anchor_rotations，anchor_bottom_heights，align_center，feature_map_stride，matched_threshold，unmatched_threshold
        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        # 为预测box找对应anchor的参数
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        # 编码box的7个残差参数(x, y, z, w, l, h, θ) --> pcdet.utils.box_coder_utils.ResidualCoder
        self.box_coder = box_coder #ResidualCoder
        # 在PointPillars中指定正负样本的时候由BEV视角计算GT和先验框的iou，不需要进行z轴上的高度的匹配，
        # 想法是：1、点云中的物体都在同一个平面上，没有物体在Z轴发生重叠的情况
        #        2、每个类别的高度相差不是很大，直接使用SmoothL1损失就可以达到很好的高度回归效果
        self.match_height = match_height #False
        # 类别名称['Car', 'Pedestrian', 'Cyclist']
        self.class_names = np.array(class_names)
        # ['Car', 'Pedestrian', 'Cyclist']
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        # anchor_target_cfg.POS_FRACTION = -1 < 0 --> None
        # 前景、背景采样系数 PointPillars不考虑
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None #None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE #512
        # False 前景权重由 1/前景anchor数量 PointPillars不考虑
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES #False
        # 类别iou匹配为正样本阈值{'Car':0.6, 'Pedestrian':0.5, 'Cyclist':0.5}
        self.matched_thresholds = {}
        # 类别iou匹配为负样本阈值{'Car':0.45, 'Pedestrian':0.35, 'Cyclist':0.35}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            # {'Car':0.6, 'Pedestrian':0.5, 'Cyclist':0.5}
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            # {'Car':0.45, 'Pedestrian':0.35, 'Cyclist':0.35}
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False) # False
        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    # 完成对一帧点云数据中所有的类别和anchor的正负样本分配
    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        处理一批数据中所有点云的anchors和gt_boxes,计算每个anchor属于前景还是背景,为每个前景的anchor分配类别和计算box的回归残差和回归权重
        Args:
            all_anchors: [(N, 7), ...] [(1,248,216,1,2,7),(1,248,216,1,2,7),(1,248,216,1,2,7)] 对应着[Z,X,Y,anchor_sizes,anchor_rotations]
            gt_boxes: (B, M, 8) # 最后维度数据为 (x, y, z, w, l, h, θ,class)
        Returns:
            all_targets_dict = {
                # 每个anchor的类别
                'box_cls_labels': cls_labels, # (batch_size,num_of_anchors)
                # 每个anchor的回归残差 -->(∆x, ∆y, ∆z, ∆l, ∆w, ∆h, ∆θ）
                'box_reg_targets': bbox_targets, # (batch_size,num_of_anchors,7)
                # 每个box的回归权重
                'reg_weights': reg_weights # (batch_size,num_of_anchors)
            }
        """
        # 1.初始化结果list并提取对应的gt_box和类别
        bbox_targets = []
        cls_labels = []
        reg_weights = []
        # 得到批大小
        batch_size = gt_boxes_with_classes.shape[0] #B
        # 得到所有GT的类别
        gt_classes = gt_boxes_with_classes[:, :, -1] #(B,M) M:num_of_gt
        gt_boxes = gt_boxes_with_classes[:, :, :-1] #(B,M,7)
        # 2.对batch中的所有数据逐帧匹配anchor的前景和背景
        for k in range(batch_size):
            cur_gt = gt_boxes[k] # 取出当前帧中的 gt_boxes (num_of_gt，7）
            cnt = cur_gt.__len__() - 1 # 得到一批数据中最多有多少个GT
            """
            由于在OpenPCDet的数据预处理时,以一批数据中拥有GT数量最多的帧为基准,
            其他帧中GT数量不足,则会进行补0操作,使其成为一个矩阵，例:
            [
                [1,1,2,2,3,2],
                [2,2,3,1,0,0],
                [3,1,2,0,0,0]
            ]
            因此这里从每一行的倒数第二个类别开始判断，截取最后一个非零元素的索引,来取出当前帧中真实的GT数据
            """
            # 这里的循环是找到最后一个非零的box，因为预处理的时候会按照batch最大box的数量处理，不足的进行补0
            while cnt > 0 and cur_gt[cnt].sum() == 0: #cur_gt:当前帧中的 gt_boxes (num_of_gt，7）
                cnt -= 1
            # 2.1提取当前帧非零的box和类别
            cur_gt = cur_gt[:cnt + 1] #当前帧有效的gt框 (cn,7) cn:有效gt的数量
            # cur_gt_classes 例: tensor([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3], device='cuda:0', dtype=torch.int32)
            cur_gt_classes = gt_classes[k][:cnt + 1].int()#当前帧有效的gt类别 (cn,1)

            target_list = []
            # 2.2 对每帧中的anchor和GT分类别，单独计算前背景
            # 计算时候 每个类别的anchor是独立计算的 不同于在ssd中整体计算iou并取最大值
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                # anchor_class_name : 车 | 行人 | 自行车
                # anchors : (1, 200, 176, 1, 2, 7)  7 --> (x, y, z, l, w, h, θ)
                if cur_gt_classes.shape[0] > 1: #有效gt数量cn
                    # self.class_names : ["car", "person", "cyclist"]
                    # 这里减1是因为列表索引从0开始，目的是得到属于列表中gt中哪些类别是与当前处理的了类别相同，得到类别mask
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)#(cn,1) self.class_names = np.array(class_names)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)
                # 在检测头中是否使用多头，默认为False
                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    # 2.2.1 计算所需的变量 得到特征图的大小
                    feature_map_size = anchors.shape[:3] # feature_map_size =（1, 248, 216）
                    # 将所有的anchors展平  shape : (216, 248, 1, 1, 2, 7) -->  (107136, 7)
                    anchors = anchors.view(-1, anchors.shape[-1]) #(107136, 7)
                    # List: 根据mask索引得到该帧中当前需要处理的类别  --> 车 | 行人 | 自行车
                    selected_classes = cur_gt_classes[mask] #(c1,1) c1:当前帧当前类别的数量
                # 2.2.2 使用assign_targets_single来单独为某一类别的anchors分配gt_boxes，
                # 并为前景、背景的box设置编码和回归权重
                # assign_targets_single：完成对一帧中每个类别的GT和anchor的正负样本分配
                single_target = self.assign_targets_single(
                    anchors, # 该类的所有anchor
                    cur_gt[mask], # GT_box  shape : （c1, 7） c1:当前帧当前类别的数量
                    gt_classes=selected_classes, # 当前选中的类别 (c1,1) c1:当前帧当前类别的数量
                    matched_threshold=self.matched_thresholds[anchor_class_name], # 当前类别anchor与GT匹配为正样本的阈值
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name] # 当前类别anchor与GT匹配为负样本的阈值
                )
                """
                    ret_dict = {
                    'box_cls_labels': labels, # (107136,)  每个anchor的类别
                    'box_reg_targets': bbox_targets, # (107136,7)编码后的结果 每个anchor的回归残差  (∆x, ∆y, ∆z, ∆l, ∆w, ∆h, ∆θ）
                    'reg_weights': reg_weights, #(107136,) 每个box的回归权重
                        }
                """
                # 到目前为止，处理完该帧单个类别和该类别anchor的前景和背景分配
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    # feature_map_size:(1，248，216, 2）
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],# [(1，248，216, 2, 1),(1，248，216, 2, 1),(1，248，216, 2, 1)]
                    # [(1，248，216, 2, 7),(1，248，216, 2, 7),(1，248，216, 2, 7)]
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    # [(1，248，216, 2, 1),(1，248，216, 2, 1),(1，248，216, 2, 1)]
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }
                # list : 3*anchor (1, 248, 216, 2, 7) --> (1, 248, 216, 6, 7) -> (321408, 7)
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)
                # list:3 (1, 248, 216, 2) --> (1，248, 216, 6) -> (1*248*216*6, )
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)#view(-1)将X里面的所有维度数据转化成一维
                # list:3 (1, 248, 216, 2) --> (1, 248, 216, 6) -> (1*248*216*6, )
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)
                

            # 将结果填入对应的容器
            bbox_targets.append(target_dict['box_reg_targets']) 
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
            # 到这里该batch的点云全部处理完

        # 3.将结果stack并返回 stack的维度为第0维度即batch的维度因为上面处理是分开batch分开类别去处理的，batch有batch_size个，类别有3个
        bbox_targets = torch.stack(bbox_targets, dim=0)# (batch_size，321408，7）

        cls_labels = torch.stack(cls_labels, dim=0) # (batch_size，321408）
        reg_weights = torch.stack(reg_weights, dim=0) # (batch_size，321408）
        all_targets_dict = {
            'box_cls_labels': cls_labels, # (batch_size，321408）
            'box_reg_targets': bbox_targets, # (batch_size，321408,7）
            'reg_weights': reg_weights # (batch_size，321408）

        }
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        """
        single_target = self.assign_targets_single(
                    anchors, # 该类的所有anchor
                    cur_gt[mask], # GT_box  shape : （c1, 7） c1:当前帧当前类别的数量
                    gt_classes=selected_classes, # 当前选中的类别 (c1,1) c1:当前帧当前类别的数量
                    matched_threshold=self.matched_thresholds[anchor_class_name], # 当前类别anchor与GT匹配为正样本的阈值
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name] # 当前类别anchor与GT匹配为负样本的阈值
                )
        针对某一类别的anchors和gt_boxes，计算前景和背景anchor的类别，box编码和回归权重
        Args:
            anchors: (107136,7)
            gt_boxes: （c1，7）
            gt_classes: (c1,1) c1:当前帧当前类别的数量
            matched_threshold:0.6 
            unmatched_threshold:0.45
        Returns:
        前景anchor
            ret_dict = {
                'box_cls_labels': labels, # (107136,)
                'box_reg_targets': bbox_targets,  # (107136,7)
                'reg_weights': reg_weights, # (107136,)
            }
        """
        #----------------------------1.初始化-------------------------------#
        num_anchors = anchors.shape[0] # 107136 该帧中该类别的anchor数量 每个类别都是这个数
        num_gt = gt_boxes.shape[0] # c1 当前帧gt中当前类别的数量
        # 初始化anchor对应的label和gt_id ，并置为 -1，-1表示loss计算时候不会被考虑，背景的类别被设置为0
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1 #(107136,) anchor对应的label
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1 #(107136,) anchor对应的gt_id
        
        # ---------------------2.计算该类别中anchor的前景和背景------------------------#
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # 1.计算该帧中某一个类别gt和对应anchors之间的iou（jaccard index）
            # anchor_by_gt_overlap    shape : (107136, c1)
            # anchor_by_gt_overlap代表当前类别的所有anchor和当前类别中所有GT的iou
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7]) #False 以bev视角进行iou匹配 (107136,c1)值为每个anchor与对应c1个当前帧当前类别的iou值

            # NOTE: The speed of these two versions depends the environment and the number of anchors
            # anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()

            # 2.得到每一个anchor与哪个的GT的的iou最大
            # anchor_to_gt_argmax表示数据维度是anchor的长度，索引是gt
            # (107136，）找到每个anchor最匹配的gt的索引
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1) #(107136,)索引值
            # anchor_to_gt_max得到每一个anchor最匹配的gt的iou数值
            # （107136，）找到每个anchor最匹配的gt的iou
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]#(107136,)与c1个当前帧当前类别gt iou最大值

            # gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            # 3.找到每个gt最匹配anchor的索引和iou
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0) #(1,c1) 索引值 哪个anchor与c1个当前帧当前类别gt最匹配
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)] #(c1,1) 对应着与gt最匹配的anchor的iou值
            # 4.标记没有匹配的gt并将iou置为-1
            empty_gt_mask = gt_to_anchor_max == 0 # 没有匹配anchor的gt的mask
            gt_to_anchor_max[empty_gt_mask] = -1 # 让没有匹配anchor的gt的iou值为-1

            # 5.找到anchor中和gt存在最大iou的anchor索引，即前景anchor
            """
                        由于在前面的实现中，仅仅找出来每个GT和anchor的最大iou索引，但是argmax返回的是索引最小的那个，
                        在匹配的过程中可能一个GT和多个anchor拥有相同的iou大小，
                        所以此处要找出这个GT与所有anchors拥有相同最大iou的anchor
            """
            # 以gt为基础，逐个anchor对应，比如第一个gt的最大iou为0.9，则在所有anchor中找iou为0.9的anchor
            # nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
            """
            矩阵比较例子 :
                        anchors_with_max_overlap = torch.tensor([[0.78, 0.1, 0.9, 0],
                                                                [0.0, 0.5, 0, 0],
                                                                [0.0, 0, 0.9, 0.8],
                                                                [0.78, 0.1, 0.0, 0]])
                        gt_to_anchor_max = torch.tensor([0.78, 0.5, 0.9,0.8])
                        anchors_with_max_overlap = anchor_by_gt_overlap == gt_to_anchor_max

                        # 返回的结果中包含了在anchor中与该GT拥有相同最大iou的所有anchor
                        anchors_with_max_overlap = tensor([[ True, False,  True, False],
                                                            [False,  True, False, False],
                                                            [False, False,  True,  True],
                                                            [ True, False, False, False]])
                        在torch中nonzero返回的是tensor中非0元素的位置，此函数在numpy中返回的是非零元素的行列表和列列表。
                        torch返回结果tensor([[0, 0],
                                            [0, 2],
                                            [1, 1],
                                            [2, 2],
                                            [2, 3],
                                            [3, 0]])
                        numpy返回结果(array([0, 0, 1, 2, 2, 3]), array([0, 2, 1, 2, 3, 0]))
                        所以可以得到第一个GT同时与第一个anchor和最后一个anchor最为匹配
            """
            """所以在实际的一批数据中可以到得到结果为
            tensor([[33382,     9],
                    [43852,    10],
                    [47284,     5],
                    [50370,     4],
                    [58498,     8],
                    [58500,     8],
                    [58502,     8],
                    [59139,     2],
                    [60751,     1],
                    [61183,     1],
                    [61420,    11],
                    [62389,     0],
                    [63216,    13],
                    [63218,    13],
                    [65046,    12],
                    [65048,    12],
                    [65478,    12],
                    [65480,    12],
                    [71924,     3],
                    [78046,     7],
                    [80150,     6]], device='cuda:0')
            在第0维度拥有相同gt索引的项，在该类所有anchor中同时拥有多个与之最为匹配的anchor
            """
            # (num_of_multiple_best_matching_for_per_GT,) anchor的索引
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0] # (35,)
            # 找到anchor中和gt存在最大iou的gt索引
            # 其实和(anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 1]的结果一样
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] # （35，）
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] # 将gt的类别赋值到对应的anchor的label中
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int() # 将gt的索引赋值到对应的anchor的gt_id中

            # 6.根据matched_threshold和unmatched_threshold以及anchor_to_gt_max计算前景和背景索引，并更新labels和gt_ids  阈值匹配的！
            # 这里应该对labels和gt_ids的操作应该包含了上面的anchors_with_max_overlap
            pos_inds = anchor_to_gt_max >= matched_threshold # 找到最匹配的anchor中iou大于给定阈值的mask #anchor_to_gt_max:(107136,)与c1个当前帧当前类别gt iou最大值
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds] # 找到最匹配的anchor中iou大于给定阈值的gt的索引 #(105,) anchor_to_gt_argmax: (107136，）找到每个anchor最匹配的gt的索引
            labels[pos_inds] = gt_classes[gt_inds_over_thresh] # 将pos anchor对应gt的类别赋值到对应的anchor的label中
            gt_ids[pos_inds] = gt_inds_over_thresh.int() # 将pos anchor对应gt的索引赋值到对应的anchor的gt_id中

            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0] # 找到背景anchor索引 (106879，)
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        # 找到前景anchor的索引--> (num_of_foreground_anchor,)
        # 106879 + 119 = 106998 < 107136 说明有一些anchor既不是背景也不是前景，
        # iou介于unmatched_threshold和matched_threshold之间
        fg_inds = (labels > 0).nonzero()[:, 0] 
        # 到目前为止得到哪些anchor是前景和哪些anchor是背景

          #------------------3.对anchor的前景和背景进行筛选和赋值--------------------#
        # 如果存在前景采样比例，则分别采样前景和背景anchor
        if self.pos_fraction is not None: # anchor_target_cfg.POS_FRACTION = -1 < 0 --> None
            num_fg = int(self.pos_fraction * self.sample_size) # self.sample_size=512
            # 如果前景anchor大于采样前景数
            if len(fg_inds) > num_fg:
                # 计算要丢弃的前景anchor数目
                num_disabled = len(fg_inds) - num_fg
                # 在前景数目中随机产生索引值，并取前num_disabled个关闭索引
                # 比如：torch.randperm(4)
                # 输出：tensor([ 2,  1,  0,  3])
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                # 将被丢弃的anchor的iou设置为-1
                labels[disable_inds] = -1
                # 更新前景索引
                fg_inds = (labels > 0).nonzero()[:, 0]

            # 计算所需背景数
            num_bg = self.sample_size - (labels > 0).sum()
            # 如果当前背景数大于所需背景数
            if len(bg_inds) > num_bg:
                # torch.randint在0到len(bg_inds)之间，随机产生size为(num_bg,)的数组
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                # 将enable_inds的标签设置为0
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            # 如果该类别没有GT的话，将该类别的全部label置0，即所有anchor都是背景类别
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                # anchor与GT的iou小于unmatched_threshold的anchor的类别设置类背景类别
                labels[bg_inds] = 0
                # 将前景赋对应类别
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        #------------------4.计算bbox_targets和reg_weights--------------------#
        # 初始化bbox_targets
        # 初始化每个anchor的7个回归参数，并设置为0数值
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size)) # (107136,7)
        # 如果该帧中有该类别的GT时候，就需要对这些设置为正样本类别的anchor进行编码操作了
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # 使用anchor_to_gt_argmax[fg_inds]来重复索引每个anchor对应前景的GT_box
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            # 提取所有属于前景的anchor
            fg_anchors = anchors[fg_inds, :]
            """
                        PointPillar编码gt和前景anchor，并赋值到bbox_targets的对应位置
                        7个参数的编码的方式为
                        ∆x = (x^gt − xa^da)/d^a , ∆y = (y^gt − ya^da)/d^a , ∆z = (z^gt − za^ha)/h^a
                        ∆w = log (w^gt / w^a) ∆l = log (l^gt / l^a) , ∆h = log (h^gt / h^a)
                        ∆θ = sin(θ^gt - θ^a)
            """
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors) # 编码gt和前景anchor，并赋值到bbox_targets的对应位置
        

        
        # 初始化回归权重
        reg_weights = anchors.new_zeros((num_anchors,)) # (107136,) new_zeros()可以方便的复制原来tensor的所有类型，比如数据类型和数据所在设备

        if self.norm_by_num_examples: # False
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples 
        else:
            reg_weights[labels > 0] = 1.0 # 将前景anchor的回归权重设置为1 背景的便为0

        ret_dict = {
            'box_cls_labels': labels, # (107136,)  每个anchor的类别
            'box_reg_targets': bbox_targets, # (107136,7)编码后的结果 每个anchor的回归残差  (∆x, ∆y, ∆z, ∆l, ∆w, ∆h, ∆θ）
            'reg_weights': reg_weights, #(107136,) 每个box的回归权重
        }
        return ret_dict
         