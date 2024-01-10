import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .vfe_template import VFETemplate

################################################ CenterNet ###############################################
###
from spconv.utils import VoxelGeneratorV2
import os
import argparse
from easydict import EasyDict as edict
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
from skimage.feature import peak_local_max
import numba

import ctypes
# fangchange
# from . import libselect
from ctypes import string_at

BN_MOMENTUM = 0.1


class kitti_config():
    # Car and Van ==> Car class
    # Pedestrian and Person_Sitting ==> Pedestrian Class
    CLASS_NAME_TO_ID = {
        'Pedestrian': -99,
        'Car': 0,
        'Cyclist': -99,
        'Van': 0,
        'Truck': 0,
        'Person_sitting': -99,
        'Tram': -99,
        'Misc': -99,
        'DontCare': -1
    }

    colors = [[0, 0, 255], [0, 255, 255], [255, 0, 0], [255, 120, 0],
              [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

    #####################################################################################
    # boundary = {
    #     "minX": 0,
    #     "maxX": 69.12,
    #     "minY": -34.56,
    #     "maxY": 34.56,
    #     "minZ": -2.73,
    #     "maxZ": 1.27
    # }
    # 修改
    boundary = {
        "minX": -25.6,
        "maxX": 25.6,
        "minY": -10,
        "maxY": 41.2,
        "minZ": -3,
        "maxZ": 1
    }


    bound_size_x = boundary['maxX'] - boundary['minX']
    bound_size_y = boundary['maxY'] - boundary['minY']
    bound_size_z = boundary['maxZ'] - boundary['minZ']

    boundary_back = {
        "minX": -50,
        "maxX": 0,
        "minY": -25,
        "maxY": 25,
        "minZ": -2.73,
        "maxZ": 1.27
    }
    # BEV_WIDTH = 432  # across y axis -25m ~ 25m
    # BEV_HEIGHT = 432  # across x axis 0m ~ 50m
    # BEV_WIDTH = 480  # across y axis -25m ~ 25m
    # BEV_HEIGHT = 480  # across x axis 0m ~ 50m
    # 修改
    BEV_WIDTH = 160  # across y axis -25m ~ 25m
    BEV_HEIGHT = 160  # across x axis 0m ~ 50m
    DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

    # maximum number of points per voxel
    T = 35

    # voxel size
    vd = 0.1  # z
    vh = 0.05  # y
    vw = 0.05  # x

    # voxel grid
    W = math.ceil(bound_size_x / vw)
    H = math.ceil(bound_size_y / vh)
    D = math.ceil(bound_size_z / vd)

    # Following parameters are calculated as an average from KITTI dataset for simplicity
    #####################################################################################
    Tr_velo_to_cam = np.array([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ])

    # cal mean from train set
    R0 = np.array([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ])

    P2 = np.array([[719.787081, 0., 608.463003, 44.9538775],
                   [0., 719.787081, 174.545111, 0.1066855],
                   [0., 0., 1., 3.0106472e-03],
                   [0., 0., 0., 0]
                   ])

    R0_inv = np.linalg.inv(R0)
    Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
    P2_inv = np.linalg.pinv(P2)
    #####################################################################################


def get_paddings_indicator(actual_num, max_num, axis=1):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    # print('actual: {}'.format(actual_num.shape))
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNetOld'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device
        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features
        print('ouput: {}'.format(self.nchannels))

    def forward(self, voxel_features, coords, batch_size):
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv,
                 output_shape,
                 voxel_generator,
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 #  vfe_num_filters=[32, 128],
                 vfe_num_filters=[32, 64],
                 num_input_features=4,
                 with_distance=False,
                 use_norm=True,
                 **kwargs):

        middle_class_name = "PointPillarsScatter",
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads
        self.voxel_generator = voxel_generator
        #修改
        self.input_h = 160
        self.input_w = 160
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #修改
        # self.conv_up_level1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        # self.conv_up_level2 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        self.conv_up_level3 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)

        self.voxel_feature_extractor = PillarFeatureNet(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range,
        )

        self.middle_feature_extractor = PointPillarsScatter(output_shape=output_shape,
                                                            num_input_features=vfe_num_filters[-1])
        #修改
        # fpn_channels = [256, 128, 64]
        fpn_channels = [64]
        #head_conv 256
        for fpn_idx, fpn_c in enumerate(fpn_channels):
            for head in sorted(self.heads):
                num_output = self.heads[head]
                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(fpn_c, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
                else:
                    fc = nn.Conv2d(in_channels=fpn_c, out_channels=num_output, kernel_size=1, stride=1, padding=0)

                self.__setattr__('fpn{}_{}'.format(fpn_idx, head), fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    # @profile
    def forward(self, voxels, coors, num_points):
        # print("PoseResNet")
        # _, _, input_h, input_w = x.size()
        hm_h, hm_w = self.input_h // 4, self.input_w // 4
        # hm_h, hm_w = self.input_h, self.input_w

        # batch_size_dev = 1
        #fangchange
        batch_size_dev = 1
        # batch_size_dev = 4

        if len(num_points.shape) == 2:  # multi-gpu
            num_voxel_per_batch = voxels.cpu().numpy().reshape(
                -1)
            voxel_list = []
            num_points_list = []
            coors_list = []
            for i, num_voxel in enumerate(num_voxel_per_batch):
                # print('{}, {}'.format(i, num_voxel))
                voxel_list.append(voxels[i, :int(num_voxel)])
                num_points_list.append(num_points[i, :num_voxel])
                coors_list.append(coors[i, :num_voxel])
            voxels = torch.cat(voxel_list, dim=0)
            num_points = torch.cat(num_points_list, dim=0)
            coors = torch.cat(coors_list, dim=0)

        voxel_features = self.voxel_feature_extractor(voxels, num_points,
                                                      coors)
        spatial_features = self.middle_feature_extractor(
            voxel_features, coors, batch_size_dev)
        afdet_spatial_features = spatial_features

        spatial_features = self.conv1(spatial_features)
        spatial_features = self.bn1(spatial_features)
        spatial_features = self.relu(spatial_features)
        spatial_features = self.maxpool(spatial_features)

   
        out_layer1 = self.layer1(spatial_features) #[1,64,40,40]
        
        out_layer2 = self.layer2(out_layer1) #[1,128,20,20]
        up_level3 = F.interpolate(out_layer2, scale_factor=2, mode='bilinear',
                                  align_corners=True)#[1,128,40,40]
        
        up_level4 = self.conv_up_level3(torch.cat((up_level3, out_layer1), dim=1))#[1,64,40,40]

        from collections import namedtuple

        ret = {}
        for head in self.heads:
            temp_outs = []
            #修改
            for fpn_idx, fdn_input in enumerate([up_level4]):
                fpn_out = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))(fdn_input)
                _, _, fpn_out_h, fpn_out_w = fpn_out.size()
                # Make sure the added features having same size of heatmap output
                if (fpn_out_w != hm_w) or (fpn_out_h != hm_h):
                    fpn_out = F.interpolate(fpn_out, size=(hm_h, hm_w))
                temp_outs.append(fpn_out)
            # Take the softmax in the keypoint feature pyramid network
            final_out = self.apply_kfpn(temp_outs)

            ret[head] = final_out

        data_name_tuple = namedtuple('data_name_tuple', ret)
        ret = data_name_tuple(**ret)

        return ret, afdet_spatial_features

    def apply_kfpn(self, outs):
        outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
        softmax_outs = F.softmax(outs, dim=-1)
        ret_outs = (outs * softmax_outs).sum(dim=-1)
        return ret_outs

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # TODO: Check initial weights for head later
            for fpn_idx in [0, 1, 2]:  # 3 FPN layers
                for head in self.heads:
                    final_layer = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))
                    for i, m in enumerate(final_layer.modules()):
                        if isinstance(m, nn.Conv2d):
                            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                            # print('=> init {}.bias as 0'.format(name))
                            if m.weight.shape[0] == self.heads[head]:
                                if 'hm' in head:
                                    nn.init.constant_(m.bias, -2.19)
                                else:
                                    nn.init.normal_(m.weight, std=0.001)
                                    nn.init.constant_(m.bias, 0)
            # pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            model_dict = self.state_dict()
            conv = nn.Conv2d(128, 64, kernel_size=7, stride=1,
                             padding=3, bias=False)
            torch.nn.init.xavier_uniform(conv.weight)
            pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if
                               k in model_dict and k != 'base_layer.0.weight'}
            pretrained_dict['conv1.weight'] = conv.weight
            self.load_state_dict(pretrained_dict, strict=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1  # expansion被设置为1，表示残差块的通道数不发生改变

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#修改
resnet_spec = {18: (BasicBlock, [1, 1, 1, 1]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


#####################################################################################


def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation using PyTorch')
    parser.add_argument('--seed', type=int, default=2020,
                        help='re-produce the results with seed random')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')

    parser.add_argument('--root-dir', type=str, default='../', metavar='PATH',
                        help='The ROOT working directory')
    ####################################################################
    ##############     Model configs            ########################
    ####################################################################
    parser.add_argument('--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--hflip_prob', type=float, default=0.5,
                        help='The probability of horizontal flip')
    parser.add_argument('--no-val', action='store_true',
                        help='If true, dont evaluate the model on the val set')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='mini-batch size (default: 16), this is the total'
                             'batch size of all GPUs on the current node when using'
                             'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--print_freq', type=int, default=50, metavar='N',
                        help='print frequency (default: 50)')
    parser.add_argument('--tensorboard_freq', type=int, default=50, metavar='N',
                        help='frequency of saving tensorboard (default: 50)')
    parser.add_argument('--checkpoint_freq', type=int, default=2, metavar='N',
                        help='frequency of saving checkpoints (default: 5)')
    ####################################################################
    ##############     Training strategy            ####################
    ####################################################################

    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=300, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr_type', type=str, default='cosin',
                        help='the type of learning rate scheduler (cosin or multi_step or one_cycle)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--minimum_lr', type=float, default=1e-7, metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('--momentum', type=float, default=0.949, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0., metavar='WD',
                        help='weight decay (default: 0.)')
    parser.add_argument('--optimizer_type', type=str, default='adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam')
    parser.add_argument('--steps', nargs='*', default=[150, 180],
                        help='number of burn in step')

    ####################################################################
    ##############     Loss weight            ##########################
    ####################################################################

    ####################################################################
    ##############     Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--world-size', default=-1, type=int, metavar='N',
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='N',
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    parser.add_argument('--evaluate', action='store_true',
                        help='only evaluate the model, not training')
    parser.add_argument('--resume_path', type=str, default=None, metavar='PATH',
                        help='the path of the resumed checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')

    configs = edict(vars(parser.parse_args(args=[])))

    ####################################################################
    ############## Hardware configurations #############################
    ####################################################################
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    configs.ngpus_per_node = torch.cuda.device_count()

    cnf = kitti_config

    configs.pin_memory = True
    configs.input_size = (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)
    configs.down_ratio = 2
    configs.hm_size = (cnf.BEV_WIDTH / configs.down_ratio, cnf.BEV_HEIGHT / configs.down_ratio)
    configs.max_objects = 50

    configs.imagenet_pretrained = True
    #修改
    configs.head_conv = 128
    configs.num_classes = 1
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos 8 for bin cos sin
    # 修改
    # configs.voxel_size = [0.16, 0.16, 4]
    configs.voxel_size = [0.32, 0.32, 4]
    # configs.point_cloud_range =[0, -34.56, -2.73, 69.12, 34.56, 1.27]
    # 修改
    configs.point_cloud_range = [-25.6, -10, -3, 25.6, 41.2,  1]
    # configs.point_cloud_range = [-38.4, -10, -3, 38.4, 66.8,  1]
    configs.max_number_of_points_per_voxel = 32

    configs.heads = {
        'hm_cen': configs.num_classes, # 1
        'cen_offset': configs.num_center_offset, # 2
        'direction': configs.num_direction, # 2
        'z_coor': configs.num_z, # 1
        'dim': configs.num_dim # 3
    }

    configs.num_input_features = 4

    ####################################################################
    ############## Dataset, logs, Checkpoints dir ######################
    ####################################################################
    configs.dataset_dir = '/media/wx/File/data/kittidata'
    configs.checkpoints_dir = os.path.join(configs.root_dir, 'checkpoints', configs.saved_fn)
    configs.logs_dir = os.path.join(configs.root_dir, 'logs', configs.saved_fn)

    if not os.path.isdir(configs.checkpoints_dir):
        os.makedirs(configs.checkpoints_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    return configs


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind.long())
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.div(topk_inds, width)).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # topk_clses = (torch.div(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    # return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
    return topk_score, topk_inds, topk_ys, topk_xs


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _neg_loss(pred, gt, alpha=2, beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float() #找到正样本的位置
    neg_inds = gt.lt(1).float() #找到负样本的位置

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0
    # print(pred.shape) #[2,1,240,240]
    # print(pos_inds.shape) #[2,1,240,240]
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss_Balanced(nn.Module):
    """Balanced L1 Loss
    paper: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    Code refer from: https://github.com/OceanPang/Libra_R-CNN
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0):
        super(L1Loss_Balanced, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        assert beta > 0
        self.beta = beta

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = self.balanced_l1_loss(pred * mask, target * mask)
        loss = loss.sum() / (mask.sum() + 1e-4)

        return loss

    def balanced_l1_loss(self, pred, target):
        assert pred.size() == target.size() and target.numel() > 0

        diff = torch.abs(pred - target)
        b = math.exp(self.gamma / self.alpha) - 1
        loss = torch.where(diff < self.beta,
                           self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff,
                           self.gamma * diff + self.gamma / b - self.alpha * self.beta)

        return loss


def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target.long(), reduction='elementwise_mean')


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 50, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 50, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 50, 2) [bin1_res, bin2_res]
    # mask: (B, 50, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)


def to_cpu(tensor):
    return tensor.detach().cpu()


class Compute_Loss(nn.Module):
    def __init__(self, device):
        super(Compute_Loss, self).__init__()
        self.device = device
        self.focal_loss = FocalLoss()
        self.l1_loss = L1Loss()
        self.l1_loss_balanced = L1Loss_Balanced(alpha=0.5, gamma=1.5, beta=1.0)
        self.weight_hm_cen = 1.
        self.weight_z_coor, self.weight_cenoff, self.weight_dim, self.weight_direction = 1., 1., 1., 1.
        self.rot_loss = BinRotLoss()

    def forward(self, outputs, tg):
        # tg: targets

        outputs = outputs._asdict()
        # print("out hm_cen: ", outputs['hm_cen'], ", ", outputs['hm_cen'].shape)
        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
        # print("afdet_features:",outputs['hm_cen'].shape)
        # print("targets:",tg['hm_cen'].shape)

        l_hm_cen = self.focal_loss(outputs['hm_cen'], tg['hm_cen'])
        l_cen_offset = self.l1_loss(outputs['cen_offset'], tg['obj_mask'], tg['indices_center'], tg['cen_offset'])
        l_direction = self.l1_loss(outputs['direction'], tg['obj_mask'], tg['indices_center'], tg['direction'])
        # l_direction = self.rot_loss(outputs['direction'], tg['obj_mask'], tg['indices_center'], tg['anglebin'], tg['angleoffset'])
        # Apply the L1_loss balanced for z coor and dimension regression
        l_z_coor = self.l1_loss_balanced(outputs['z_coor'], tg['obj_mask'], tg['indices_center'], tg['z_coor'])
        l_dim = self.l1_loss_balanced(outputs['dim'], tg['obj_mask'], tg['indices_center'], tg['dim'])

        total_loss = l_hm_cen * self.weight_hm_cen + l_cen_offset * self.weight_cenoff + \
                     l_dim * self.weight_dim + l_direction * self.weight_direction + \
                     l_z_coor * self.weight_z_coor

        # loss_stats = {
        #     'total_loss': to_cpu(total_loss).item(),
        #     'hm_cen_loss': to_cpu(l_hm_cen).item(),
        #     'cen_offset_loss': to_cpu(l_cen_offset).item(),
        #     'dim_loss': to_cpu(l_dim).item(),
        #     'direction_loss': to_cpu(l_direction).item(),
        #     'z_coor_loss': to_cpu(l_z_coor).item(),
        # }

        # return total_loss, loss_stats
        return total_loss


###
################################################ CenterNet ###############################################

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


configs = parse_train_configs()

class interRAL(nn.Module):
    def __init__(self, channels):
        super(interRAL, self).__init__()
        self.linear = nn.Linear(10, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)

        self.trans_conv_up = nn.Linear(channels // 4, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool1d(4, 4)

    def forward(self, x, y):
        x = self.linear(x).permute(0, 2, 1)
        # x = x.permute(1,2,0)
        # x = self.maxpool(x)
        # x = x.permute(2,0,1)

        y = self.linear(y).permute(0, 2, 1)
        y_t = y.permute(1, 2, 0)

        # print("y",y.shape)
        if y.shape[2] > 100:
            y = self.maxpool(y_t)
            y = y.permute(2, 0, 1)

        x_q = self.q_conv(x).permute(1, 0, 2)  # b, n, c
        y_k = self.k_conv(y).permute(1, 2, 0)  # b, c, n
        y_v = self.v_conv(y).permute(1, 0, 2)
        # print("my version-----------")
        # print("x",x_q.shape)
        # print("y",y_k.shape)

        energy = torch.bmm(y_k, y_v)  # 16, n1,n2
        # print("energy",energy.shape)
        attention = self.softmax(energy)  # 16,b1,b2
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        # y_r = torch.bmm(y_v,attention).permute(1, 2, 0) # b, c, n
        # print(attention.shape)
        # print(x_q.shape)
        y_r = torch.bmm(x_q, attention).permute(1, 2, 0)  # b, c, n
        y_r = self.trans_conv_up(y_r).permute(0, 2, 1)
        # print("yr",y_r.shape)  # b,32,16 -> 2115,64,32
        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        ###
        configs = parse_train_configs()

        configs.device = torch.device('cuda:1')
        arch_parts = configs.arch.split('_')
        num_layers = int(arch_parts[-1])
        heads = configs.heads
        head_conv = configs.head_conv
        vfe_num_filters = list([64])

        voxel_generator = VoxelGeneratorV2(
            voxel_size=list(configs.voxel_size),
            point_cloud_range=list(configs.point_cloud_range),
            max_num_points=configs.max_number_of_points_per_voxel,
            max_voxels=20000
        )

        grid_size = voxel_generator.grid_size
        dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]

        block_class, layers = resnet_spec[num_layers]
        self.AFDet = PoseResNet(block_class, layers, heads, head_conv=head_conv, output_shape=dense_shape,
                                voxel_generator=voxel_generator)

        self.interral = interRAL(64)
        ###

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    # @profile
    def forward(self, batch_dict, **kwargs):
        ######################################### Center部分 ############################################
        # optimizer
        # optimizer = create_optimizer(configs, model)
        
        #6.7%
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        #18.4
        afdet_features, afdet_spatial_features = self.AFDet(batch_dict['lidar_voxels'],
                                                            batch_dict['lidar_voxel_coords'],
                                                            batch_dict['lidar_voxel_num_points'])
        # batch_dict['afdet_spatial_features'] = afdet_spatial_features
        # loss 训练时记得解注释
        targets = {
            'hm_cen': batch_dict['hm_cen'],
            'cen_offset': batch_dict['cen_offset'],
            'direction': batch_dict['direction'],
            'z_coor': batch_dict['z_coor'],
            'dim': batch_dict['dim'],
            'indices_center': batch_dict['indices_center'],
            'obj_mask': batch_dict['obj_mask']
        }
        #修改
        criterion = Compute_Loss(device=configs.device)
        total_loss = criterion(afdet_features, targets)
        configs.distributed = False
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)
        cen_loss = total_loss
        batch_dict['cen_loss'] = cen_loss
        
        total_loss.backward()
        print("t_loss", t2 - t1)
        for k in targets.keys():
            targets[k] = targets[k].to(configs.device, non_blocking=True)
        total_loss, loss_stats = criterion(outputs, targets)

        heatmap_center = afdet_features.hm_cen
        heatmap_center = _nms(heatmap_center)
        K = 20
        cen_offset = afdet_features.cen_offset
        # 获取分数及关键点坐标
        batch_size, num_classes, height, width = heatmap_center.size()
        # heatmap_center = _nms(heatmap_center)
        scores, inds, ys, xs = _topk(heatmap_center, K=K)
        if cen_offset is not None:
            cen_offset = _transpose_and_gather_feat(cen_offset, inds)
            cen_offset = cen_offset.view(batch_size, K, 2)
            xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
            ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
        else:
            xs = xs.view(batch_size, K, 1) + 0.5
            ys = ys.view(batch_size, K, 1) + 0.5

        scores = scores.view(batch_size, K, 1)

        xs_008 = xs * 8
        ys_008 = ys * 8

        for batch in range(batch_size):
            #找关键点周围的点
            # index matrix method:
            radar_batch_0_len = batch_dict['radar_batch_0_len']
            lidar_batch_0_len = batch_dict['lidar_batch_0_len']
            radar_pillars_num = 0
            lidar_pillars_num = 0

            radar_batch_0_len = 0

            lidar_batch_0_len = 0

            radar_coords = np.array(batch_dict['radar_voxel_coords_008'].cpu())
            lidar_coords = np.array(batch_dict['lidar_voxel_coords_008'].cpu())



            # 对于雷达坐标
            radar_class_ids = radar_coords[:, 0].astype(int)
            radar_batch_0_len = np.sum(radar_class_ids == 0)


            # 对于激光雷达坐标
            lidar_class_ids = lidar_coords[:, 0].astype(int)
            lidar_batch_0_len = np.sum(lidar_class_ids == 0)
          

            for i in range(len(xs_008[batch])):
                xx = int(xs_008[batch][i]+0.5)
                yy = int(ys_008[batch][i]+0.5)
                x_min = max(xx - 24, 0)
                x_max = min(xx + 24, 2000)
                y_min = max(yy - 24, 0)
                y_max = min(yy + 24, 2000)

                radar_pillars_matrix_index = batch_dict['radar_pillars_matrix_index'][batch]
                lidar_pillars_matrix_index = batch_dict['lidar_pillars_matrix_index'][batch]

                # 创建布尔掩码来表示雷达和激光有效位置
                radar_pillar_valid = radar_pillars_matrix_index[x_min:x_max, y_min:y_max] != -1
                lidar_pillar_valid = lidar_pillars_matrix_index[x_min:x_max, y_min:y_max] != -1

                # 计算雷达和激光有效数量
                radar_pillars_num += np.sum(radar_pillar_valid)
                lidar_pillars_num += np.sum(lidar_pillar_valid)

                if radar_pillars_num != 0:
                    # 处理雷达数据
                    if batch == 1:
                        radar_pillars_matrix_index[x_min:x_max, y_min:y_max][radar_pillar_valid] += radar_batch_0_len

                    radar_center_indices = radar_pillars_matrix_index[x_min:x_max, y_min:y_max]
                    radar_center_indices_valid = radar_center_indices[radar_pillar_valid]
                    # print("len",len(radar_center_indices_valid))

                    if len(radar_center_indices_valid) != 0:
                    # 使用向量化操作收集雷达数据
                        radar_center_coords_test = batch_dict['radar_voxel_coords_008'][radar_center_indices_valid].view(-1,4)
                        radar_center_num_point_test = batch_dict['radar_voxel_num_points_008'][radar_center_indices_valid].view(-1)
                        radar_center_voxels_test = batch_dict['radar_voxels_008'][radar_center_indices_valid].view(-1,32, 4)
                    # print("r",radar_pillars_num_1)
                    # print(radar_center_voxels.shape)

                if lidar_pillars_num != 0:
                # 处理激光数据
                    if batch == 1:
                        lidar_pillars_matrix_index[x_min:x_max, y_min:y_max][lidar_pillar_valid] += lidar_batch_0_len

                    lidar_center_indices = lidar_pillars_matrix_index[x_min:x_max, y_min:y_max]
                    lidar_center_indices_valid = lidar_center_indices[lidar_pillar_valid]
                    if len(lidar_center_indices_valid) != 0:
                    # 使用向量化操作收集激光数据
                        lidar_center_coords_test = batch_dict['lidar_voxel_coords_008'][lidar_center_indices_valid].view(-1,4)
                        lidar_center_num_point_test = batch_dict['lidar_voxel_num_points_008'][lidar_center_indices_valid].view(-1)
                        lidar_center_voxels_test = batch_dict['lidar_voxels_008'][lidar_center_indices_valid].view(-1, 32, 4)
                    
            #不同batch cat
            if batch == range(batch_size)[0]:
                if radar_pillars_num == 0:  #
                    radar_mask_c = batch_dict['radar_voxel_coords_008'][:, 0] == batch
                    radar_batch_coords_t = batch_dict['radar_voxel_coords_008'][radar_mask_c, :]
                    radar_batch_num_points_t = batch_dict['radar_voxel_num_points_008'][radar_mask_c]
                    radar_batch_voxels_t = batch_dict['radar_voxels_008'][radar_mask_c, :]
                else:
                   radar_batch_coords_t = radar_center_coords_test
                   radar_batch_num_points_t = radar_center_num_point_test
                   radar_batch_voxels_t = radar_center_voxels_test

                if lidar_pillars_num == 0:  #
                    # control_bit[batch] == False
                    lidar_mask_c = batch_dict['lidar_voxel_coords_008'][:, 0] == batch
                    lidar_batch_coords_t = batch_dict['lidar_voxel_coords_008'][lidar_mask_c, :]
                    lidar_batch_num_points_t = batch_dict['lidar_voxel_num_points_008'][lidar_mask_c]
                    lidar_batch_voxels_t = batch_dict['lidar_voxels_008'][lidar_mask_c, :]
                else:
                   lidar_batch_coords_t = lidar_center_coords_test
                   lidar_batch_num_points_t = lidar_center_num_point_test
                   lidar_batch_voxels_t = lidar_center_voxels_test
            else:
                if radar_pillars_num == 0:
                    # control_bit[batch] == False
                    radar_mask_cc = batch_dict['radar_voxel_coords_008'][:, 0] == batch
                    radar_batch_coords_t = torch.cat((radar_batch_coords_t, batch_dict['radar_voxel_coords_008'][radar_mask_cc, :]), dim=0)
                    radar_batch_num_points_t = torch.cat((radar_batch_num_points_t, batch_dict['radar_voxel_num_points_008'][radar_mask_cc]), dim=0)
                    radar_batch_voxels_t = torch.cat((radar_batch_voxels_t, batch_dict['radar_voxels_008'][radar_mask_cc]), dim=0)
                else:
                   radar_batch_coords_t = torch.cat((radar_batch_coords_t, radar_center_coords_test), dim=0)
                   radar_batch_num_points_t = torch.cat((radar_batch_num_points_t, radar_center_num_point_test), dim=0)
                   radar_batch_voxels_t = torch.cat((radar_batch_voxels_t, radar_center_voxels_test), dim=0)

                if lidar_pillars_num == 0:
                    # control_bit[batch] == False
                    lidar_mask_cc = batch_dict['lidar_voxel_coords_008'][:, 0] == batch
                    lidar_batch_coords_t = torch.cat((lidar_batch_coords_t, batch_dict['lidar_voxel_coords_008'][lidar_mask_cc, :]), dim=0)
                    lidar_batch_num_points_t = torch.cat((lidar_batch_num_points_t, batch_dict['lidar_voxel_num_points_008'][lidar_mask_cc]), dim=0)
                    lidar_batch_voxels_t = torch.cat((lidar_batch_voxels_t, batch_dict['lidar_voxels_008'][lidar_mask_cc]), dim=0)
                else:
                   lidar_batch_coords_t = torch.cat((lidar_batch_coords_t, lidar_center_coords_test), dim=0)
                   lidar_batch_num_points_t = torch.cat((lidar_batch_num_points_t, lidar_center_num_point_test), dim=0)
                   lidar_batch_voxels_t = torch.cat((lidar_batch_voxels_t, lidar_center_voxels_test), dim=0)

        radar_batch_coords = radar_batch_coords_t
        radar_batch_num_points = radar_batch_num_points_t
        radar_batch_voxels = radar_batch_voxels_t
        lidar_batch_coords = lidar_batch_coords_t
        lidar_batch_num_points = lidar_batch_num_points_t
        lidar_batch_voxels = lidar_batch_voxels_t

        # print("result len: ", lidar_batch_coords.shape)
        ############################################### radar CenterPillar参与训练 ########################################
        radar_cen_points_mean = radar_batch_voxels[:, :, :3].sum(dim=1, keepdim=True) / radar_batch_num_points.type_as(
            radar_batch_voxels).view(-1, 1, 1)
        radar_cen_f_cluster = radar_batch_voxels[:, :, :3] - radar_cen_points_mean

        radar_cen_f_center = torch.zeros_like(radar_batch_voxels[:, :, :3])
        radar_cen_f_center[:, :, 0] = radar_batch_voxels[:, :, 0] - (
                    radar_batch_coords[:, 3].to(radar_batch_voxels.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        radar_cen_f_center[:, :, 1] = radar_batch_voxels[:, :, 1] - (
                    radar_batch_coords[:, 2].to(radar_batch_voxels.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        radar_cen_f_center[:, :, 2] = radar_batch_voxels[:, :, 2] - (
                    radar_batch_coords[:, 1].to(radar_batch_voxels.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            radar_cen_features = [radar_batch_voxels, radar_cen_f_cluster, radar_cen_f_center]
        else:
            radar_cen_features = [radar_batch_voxels[..., 3:], radar_cen_f_cluster, radar_cen_f_center]

        if self.with_distance:
            radar_cen_points_dist = torch.norm(radar_batch_voxels[:, :, :3], 2, 2, keepdim=True)
            radar_cen_features.append(radar_cen_points_dist)
        radar_cen_features = torch.cat(radar_cen_features, dim=-1)

        radar_cen_voxel_count = radar_cen_features.shape[1]
        radar_cen_mask = self.get_paddings_indicator(radar_batch_num_points, radar_cen_voxel_count, axis=0)
        radar_cen_mask = torch.unsqueeze(radar_cen_mask, -1).type_as(radar_batch_voxels)
        radar_cen_features *= radar_cen_mask
        ################################################# radar CenterPillar参与训练 ###############################################

        ############################################### lidar CenterPillar参与训练 ########################################
        lidar_cen_points_mean = lidar_batch_voxels[:, :, :3].sum(dim=1, keepdim=True) / lidar_batch_num_points.type_as(
            lidar_batch_voxels).view(-1, 1, 1)
        lidar_cen_f_cluster = lidar_batch_voxels[:, :, :3] - lidar_cen_points_mean

        lidar_cen_f_center = torch.zeros_like(lidar_batch_voxels[:, :, :3])
        lidar_cen_f_center[:, :, 0] = lidar_batch_voxels[:, :, 0] - (
                    lidar_batch_coords[:, 3].to(lidar_batch_voxels.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        lidar_cen_f_center[:, :, 1] = lidar_batch_voxels[:, :, 1] - (
                    lidar_batch_coords[:, 2].to(lidar_batch_voxels.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        lidar_cen_f_center[:, :, 2] = lidar_batch_voxels[:, :, 2] - (
                    lidar_batch_coords[:, 1].to(lidar_batch_voxels.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            lidar_cen_features = [lidar_batch_voxels, lidar_cen_f_cluster, lidar_cen_f_center]
        else:
            lidar_cen_features = [lidar_batch_voxels[..., 3:], lidar_cen_f_cluster, lidar_cen_f_center]

        if self.with_distance:
            lidar_cen_points_dist = torch.norm(lidar_batch_voxels[:, :, :3], 2, 2, keepdim=True)
            lidar_cen_features.append(lidar_cen_points_dist)
        lidar_cen_features = torch.cat(lidar_cen_features, dim=-1)

        lidar_cen_voxel_count = lidar_cen_features.shape[1]
        lidar_cen_mask = self.get_paddings_indicator(lidar_batch_num_points, lidar_cen_voxel_count, axis=0)
        lidar_cen_mask = torch.unsqueeze(lidar_cen_mask, -1).type_as(lidar_batch_voxels)
        lidar_cen_features *= lidar_cen_mask
        ################################################# lidar CenterPillar参与训练 ###############################################

        ###################################### transformer ##########################################
        lidar_cen_features_output = self.interral(lidar_cen_features, radar_cen_features)

        radar_cen_features_output = self.interral(radar_cen_features, lidar_cen_features)


        lidar_cen_features = lidar_cen_features_output.view(
            [lidar_cen_features_output.size()[0], lidar_cen_features_output.size()[1]])
        radar_cen_features = radar_cen_features_output.view(
            [radar_cen_features_output.size()[0], radar_cen_features_output.size()[1]])
        batch_dict['radar_cen_pillar_features'] = radar_cen_features
        batch_dict['radar_cen_voxel_coords'] = radar_batch_coords
        batch_dict['lidar_cen_pillar_features'] = lidar_cen_features
        batch_dict['lidar_cen_voxel_coords'] = lidar_batch_coords
        ###################################### transformer ##########################################

        ################################################# lidar和radar参与训练 ###############################################
        lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict[
            'lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
        radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict[
            'radar_voxel_num_points'], batch_dict['radar_voxel_coords']
        lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(
            lidar_voxel_features).view(-1, 1, 1)
        radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(
            radar_voxel_features).view(-1, 1, 1)
        lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
        radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

        lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
        radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
        lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (
                    lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (
                    lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (
                    lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (
                    radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (
                    radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (
                    radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
        else:
            lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
        if self.use_absolute_xyz:
            radar_features = [radar_voxel_features, radar_f_cluster, radar_f_center]
        else:
            radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]

        if self.with_distance:
            lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
            lidar_features.append(lidar_points_dist)
        lidar_features = torch.cat(lidar_features, dim=-1)
        if self.with_distance:
            radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
            radar_features.append(radar_points_dist)
        radar_features = torch.cat(radar_features, dim=-1)

        lidar_voxel_count = lidar_features.shape[1]
        radar_voxel_count = radar_features.shape[1]
        lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
        radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
        lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
        radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
        lidar_features *= lidar_mask
        radar_features *= radar_mask


        ###################################### transformer ##########################################
        lidar_features_output = self.interral(lidar_features, radar_features)
        radar_features_output = self.interral(radar_features, lidar_features)
        lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
        radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])
        ###################################### transformer ##########################################

        batch_dict['lidar_pillar_features'] = lidar_features
        batch_dict['radar_pillar_features'] = radar_features

        return batch_dict