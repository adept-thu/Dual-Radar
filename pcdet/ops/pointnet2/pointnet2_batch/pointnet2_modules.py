from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import Conv1dNormRelu, Conv2dNormRelu 

from . import pointnet2_utils


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.farthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method

class PointnetSAModuleMSG_WithSampling(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 npoint_list: List[int],
                 sample_range_list: List[int],
                 sample_type_list: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_class):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling type
        :param sample_type_list: list of str, list of used sampling type, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_group: whether to use dilated group
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        super().__init__()
        self.sample_type_list = sample_type_list
        self.sample_range_list = sample_range_list
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                self.groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None


    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, cls_features: torch.Tensor = None, new_xyz=None, ctr_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param cls_features: (B, N, num_class) tensor of the descriptors of the the confidence (classification) features 
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers 
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(self.sample_type_list)):
                sample_type = self.sample_type_list[i]
                sample_range = self.sample_range_list[i]
                npoint = self.npoint_list[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) 
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()

        else:
            new_xyz = ctr_xyz

        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        if self.confidence_layers is not None:
            cls_features = self.confidence_layers(new_features).transpose(1, 2)
            
        else:
            cls_features = None

        return new_xyz, new_features, cls_features

class Vote_layer(nn.Module):
    """ Light voting module with limitation"""
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        self.mlp_list = mlp_list
        if len(mlp_list) > 0:
            for i in range(len(mlp_list)):
                shared_mlps = []

                shared_mlps.extend([
                    nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_list[i]),
                    nn.ReLU()
                ])
                pre_channel = mlp_list[i]
            self.mlp_modules = nn.Sequential(*shared_mlps)
        else:
            self.mlp_modules = None

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.max_offset_limit = torch.tensor(max_translate_range).float() if max_translate_range is not None else None
       

    def forward(self, xyz, features):
        xyz_select = xyz
        features_select = features

        if self.mlp_modules is not None: 
            new_features = self.mlp_modules(features_select) #([4, 256, 256]) ->([4, 128, 256])            
        else:
            new_features = new_features
        
        ctr_offsets = self.ctr_reg(new_features) #[4, 128, 256]) -> ([4, 3, 256])

        ctr_offsets = ctr_offsets.transpose(1, 2)#([4, 256, 3])
        feat_offets = ctr_offsets[..., 3:]
        new_features = feat_offets
        ctr_offsets = ctr_offsets[..., :3]
        
        if self.max_offset_limit is not None:
            max_offset_limit = self.max_offset_limit.view(1, 1, 3)            
            max_offset_limit = self.max_offset_limit.repeat((xyz_select.shape[0], xyz_select.shape[1], 1)).to(xyz_select.device) #([4, 256, 3])
      
            limited_ctr_offsets = torch.where(ctr_offsets > max_offset_limit, max_offset_limit, ctr_offsets)
            min_offset_limit = -1 * max_offset_limit
            limited_ctr_offsets = torch.where(limited_ctr_offsets < min_offset_limit, min_offset_limit, limited_ctr_offsets)
            vote_xyz = xyz_select + limited_ctr_offsets
        else:
            vote_xyz = xyz_select + ctr_offsets

        return vote_xyz, new_features, xyz_select, ctr_offsets

class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)

class _PointnetSAModuleFSBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.groupers = None
        self.mlps = None
        self.npoint_list = []
        self.sample_range_list = [[0, -1]]
        self.sample_method_list = ['d-fps']
        self.radii = []

        self.pool_method = 'max_pool'
        self.dilated_radius_group = False
        self.weight_gamma = 1.0
        self.skip_connection = False

        self.aggregation_mlp = None
        self.confidence_mlp = None

    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz=None,
                scores=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            assert len(self.npoint_list) == len(self.sample_range_list) == len(self.sample_method_list)
            sample_idx_list = []
            for i in range(len(self.sample_method_list)):
                xyz_slice = xyz[:, self.sample_range_list[i][0]:self.sample_range_list[i][1], :].contiguous()
                if self.sample_method_list[i] == 'd-fps':
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                elif self.sample_method_list[i] == 'f-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]]
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i])
                elif self.sample_method_list[i] == 's-fps':
                    assert scores is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma
                    sample_idx = pointnet2_utils.furthest_point_sample_weights(
                        xyz_slice,
                        scores_slice,
                        self.npoint_list[i]
                    )
                else:
                    raise NotImplementedError

                sample_idx_list.append(sample_idx + self.sample_range_list[i][0])

            sample_idx = torch.cat(sample_idx_list, dim=-1)
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3)
            
            if self.skip_connection: 
                old_features = pointnet2_utils.gather_operation(
                    features,
                    sample_idx
                ) if features is not None else None  # (B, C, npoint)

        for i in range(len(self.groupers)):
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            idx_cnt_mask = (idx_cnt > 0).float()  # (B, npoint)
            idx_cnt_mask = idx_cnt_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)
            new_features = new_features * idx_cnt_mask

            if self.pool_method == 'max_pool':
                pooled_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError
            
            new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)

        if self.skip_connection and old_features is not None:
            new_features_list.append(old_features)

        new_features = torch.cat(new_features_list, dim=1)
        if self.aggregation_mlp is not None:
            new_features = self.aggregation_mlp(new_features)

        if self.confidence_mlp is not None:
            new_scores = self.confidence_mlp(new_features)
            new_scores = new_scores.squeeze(1)  # (B, npoint)
            return new_xyz, new_features, new_scores

        return new_xyz, new_features, None

class PointnetSAModuleFSMSG(_PointnetSAModuleFSBase):
    """Pointnet set abstraction layer with fusion sampling and multiscale grouping"""

    def __init__(self, *,
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None):
        """
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__()

        assert npoint_list is None or len(npoint_list) == len(sample_range_list) == len(sample_method_list)
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.sample_range_list = sample_range_list
        self.sample_method_list = sample_method_list
        self.radii = radii
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        former_radius = 0.0
        in_channels, out_channels = 0, 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if dilated_radius_group:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroupDilated(former_radius, radius, nsample, use_xyz=use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                )
            former_radius = radius
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlp = []
            for k in range(len(mlp_spec) - 1):
                shared_mlp.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlp))
            in_channels = mlp_spec[0] - 3 if use_xyz else mlp_spec[0]
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method
        self.dilated_radius_group = dilated_radius_group
        self.skip_connection = skip_connection
        self.weight_gamma = weight_gamma

        if skip_connection:
            out_channels += in_channels

        if aggregation_mlp is not None:
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_mlp = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_mlp = None

        if confidence_mlp is not None:
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, 1, kernel_size=1, bias=True),
            )
            self.confidence_mlp = nn.Sequential(*shared_mlp)
        else:
            self.confidence_mlp = None


class PointnetSAModuleFSMSGWithImage(PointnetSAModuleFSMSG):
    "Fuse point cloud features with image features."
    def __init__(self, *,
                npoint_list: List[int] = None,
                sample_range_list: List[List[int]] = None,
                sample_method_list: List[str] = None,
                radii: List[float],
                nsamples: List[int],
                mlps: List[List[int]],
                bn: bool = True,
                use_xyz: bool = True,
                pool_method='max_pool',
                dilated_radius_group: bool = False,
                skip_connection: bool = False,
                weight_gamma: float = 1.0,
                aggregation_mlp: List[int] = None,
                confidence_mlp: List[int] = None,
                image_feature_channel: int=0):
        super().__init__(
            mlps=mlps, npoint_list=npoint_list, sample_range_list=sample_range_list,
            sample_method_list=sample_method_list, radii=radii, nsamples=nsamples,
            bn=bn, use_xyz=use_xyz, pool_method=pool_method, dilated_radius_group=dilated_radius_group,
            skip_connection=skip_connection, weight_gamma=weight_gamma,
            aggregation_mlp=aggregation_mlp, confidence_mlp=confidence_mlp
        )
        in_channels_2d = image_feature_channel
        in_channels_3d = aggregation_mlp[-1]

        # image to points
        self.fuse3d_before_mlps = nn.Sequential(
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm='batch_norm'),
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm='batch_norm'),
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm='batch_norm'),
        )
        self.fuse3d_mlp = nn.Sequential(nn.Conv1d(in_channels_2d + in_channels_3d, in_channels_3d, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(in_channels_3d),
                                    nn.ReLU())

        # points to image
        self.fuse2d_mlp = nn.Sequential(
            Conv2dNormRelu(in_channels_3d, in_channels_3d, norm='batch_norm'),
            Conv2dNormRelu(in_channels_3d, in_channels_3d, norm='batch_norm'),
            Conv2dNormRelu(in_channels_3d, in_channels_3d, norm='batch_norm'),
        )
        self.fuse2d_conv = Conv2dNormRelu(in_channels_2d + in_channels_3d, in_channels_2d)


    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                vis: torch.Tensor = None,
                image_features: torch.Tensor = None,
                new_xyz=None,
                scores=None,
                world_scale=None,
                world_rotation=None,
                flip_along_x=None,
                V2R=None,
                P2=None,
                image_shape=None,
                images=None
                ):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the features
        :param new_xyz:
        :param scores: (B, N) tensor of confidence scores of points, required when using s-fps
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        vis_flipped = vis.transpose(1, 2).contiguous()
        
        if new_xyz is None:
            assert len(self.npoint_list) == len(self.sample_range_list) == len(self.sample_method_list)
            sample_idx_list = []
            for i in range(len(self.sample_method_list)):
                xyz_slice = xyz[:, self.sample_range_list[i][0]:self.sample_range_list[i][1], :].contiguous()
                if self.sample_method_list[i] == 'd-fps':
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_slice, self.npoint_list[i])
                elif self.sample_method_list[i] == 'f-fps':
                    features_slice = features[:, :, self.sample_range_list[i][0]:self.sample_range_list[i][1]]
                    dist_matrix = pointnet2_utils.calc_dist_matrix_for_sampling(xyz_slice,
                                                                                features_slice.permute(0, 2, 1),
                                                                                self.weight_gamma)
                    sample_idx = pointnet2_utils.furthest_point_sample_matrix(dist_matrix, self.npoint_list[i])
                elif self.sample_method_list[i] == 's-fps':
                    assert scores is not None
                    scores_slice = \
                        scores[:, self.sample_range_list[i][0]:self.sample_range_list[i][1]].contiguous()
                    scores_slice = scores_slice.sigmoid() ** self.weight_gamma
                    sample_idx = pointnet2_utils.furthest_point_sample_weights(
                        xyz_slice,
                        scores_slice,
                        self.npoint_list[i]
                    )
                else:
                    raise NotImplementedError

                sample_idx_list.append(sample_idx + self.sample_range_list[i][0])

            sample_idx = torch.cat(sample_idx_list, dim=-1)
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3)
            # import pdb;pdb.set_trace()
            new_vis = pointnet2_utils.gather_operation(
                vis_flipped,
                sample_idx
            ).transpose(1, 2).contiguous()  # (B, npoint, 3)
            
            if self.skip_connection: 
                old_features = pointnet2_utils.gather_operation(
                    features,
                    sample_idx
                ) if features is not None else None  # (B, C, npoint)

        for i in range(len(self.groupers)):
            idx_cnt, new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            idx_cnt_mask = (idx_cnt > 0).float()  # (B, npoint)
            idx_cnt_mask = idx_cnt_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, npoint, 1)
            new_features = new_features * idx_cnt_mask

            if self.pool_method == 'max_pool':
                pooled_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                pooled_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError
            
            new_features_list.append(pooled_features.squeeze(-1))  # (B, mlp[-1], npoint)

        if self.skip_connection and old_features is not None:
            new_features_list.append(old_features)

        new_features = torch.cat(new_features_list, dim=1)
        if self.aggregation_mlp is not None:
            new_features = self.aggregation_mlp(new_features)

        # fuse point features and image features
        if image_features is not None:
            batch_size = new_features.shape[0]
            point_image_features_list = []
            image_point_features_list = []
            uv_list = []

            batch_size, _, image_features_h, image_features_w = image_features.shape
            
            # import pdb;pdb.set_trace()
            for bs_idx in range(batch_size):
                keypoints_b = new_xyz[bs_idx].clone()
                # if self.training:
                # restore the raw positions of keypoints, seq: scale->rotate->flip?
                if world_scale is not None:
                    world_scale_b = world_scale[bs_idx]
                    keypoints_b /= world_scale_b
                if world_rotation is not None:
                    world_rotation_b = world_rotation[bs_idx]
                    keypoints_b = common_utils.rotate_points_along_z_single(keypoints_b, -world_rotation_b)
                if flip_along_x is not None:
                    flip_along_x_b = flip_along_x[bs_idx] # ad hoc, only process flip_x
                    if flip_along_x_b:
                        keypoints_b[:, 1] = -keypoints_b[:, 1]

                # project keypoint to image
                keypoints_b_hom = torch.cat([keypoints_b, keypoints_b.new_ones(len(keypoints_b),1)], dim=-1)
                scan_C0 = torch.mm(keypoints_b_hom, V2R[bs_idx].T)
                scan_C2 = torch.mm(scan_C0, P2[bs_idx].T) # [N, 3]
                scan_C2_depth = scan_C2[:, 2]
                scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T

                # save debug images
                # import matplotlib.pyplot as plt
                # import numpy as np
                # _scan_C2 = scan_C2.cpu().numpy();_scan_C2_depth = scan_C2_depth.cpu().numpy()
                # img = images[bs_idx][:3].permute(1,2,0).cpu().numpy()
                # fig = plt.figure(figsize=(12, 6));plt.imshow(img)
                # inds = _scan_C2[:, 0] > 0;inds = np.logical_and(inds, _scan_C2[:, 0] < img.shape[1]);inds = np.logical_and(inds, _scan_C2[:, 1] > 0);inds = np.logical_and(inds, _scan_C2[:, 1] < img.shape[0]);inds = np.logical_and(inds, _scan_C2_depth > 0)
                # plt.scatter(_scan_C2[inds, 0], _scan_C2[inds, 1], c=-_scan_C2_depth[inds], alpha=1, s=2, cmap='viridis')
                # plt.axis('off');plt.tight_layout();plt.savefig('output.png', bbox_inches='tight')
                # import pdb;pdb.set_trace()

                scan_C2[:, 0] *= (image_features[bs_idx].shape[2]/image_shape[1]) # w
                scan_C2[:, 1] *= (image_features[bs_idx].shape[1]/image_shape[0]) # h
                uv_list.append(scan_C2)

                # cur_point_image_features = common_utils.bilinear_interpolate_torch(cur_image_features, scan_C2[:, 0], scan_C2[:, 1])
                cur_image_features = image_features[bs_idx].permute(1, 2, 0)  # (C,H,W) -> (H, W, C)
                valid_mask = new_vis[bs_idx].flatten()
                cur_point_image_features = torch.zeros((len(valid_mask), cur_image_features.shape[-1]), dtype=cur_image_features.dtype, device=cur_image_features.device)
                valid_cur_point_image_features = common_utils.bilinear_interpolate_torch(cur_image_features, scan_C2[valid_mask==1, 0], scan_C2[valid_mask==1, 1])
                cur_point_image_features[valid_mask==1, :] = valid_cur_point_image_features
                point_image_features_list.append(cur_point_image_features)

            feat_3d_to_2d_list = []
            for bs_idx in range(batch_size):
                bs_uv = uv_list[bs_idx]
                bs_uv_grid = torch.floor(bs_uv).int()
                valid_point_mask = (bs_uv[:, 0]>=0) & (bs_uv[:, 0]<image_features[0].shape[2]) & (bs_uv[:, 1]>=0) & (bs_uv[:, 1]<image_features[0].shape[1])

                # import pdb;pdb.set_trace()
                valid_point_mask = valid_point_mask & (new_vis[bs_idx]>0).flatten()

                bs_uv_grid = bs_uv_grid[valid_point_mask].transpose(0,1)
                bs_new_features = new_features[bs_idx].transpose(0,1)[valid_point_mask, :]
                mask_value = torch.ones([bs_new_features.shape[0],1], device=bs_new_features.device)

                gather_3d_features = torch.sparse_coo_tensor(bs_uv_grid, bs_new_features, (image_features_w, image_features_h, new_features.shape[1]))
                gather_3d_features = gather_3d_features.to_dense()
                gather_counts = torch.sparse_coo_tensor(bs_uv_grid, mask_value, (image_features_w, image_features_h, 1))
                gather_counts = gather_counts.to_dense()
                bs_feat_3d_to_2d = gather_3d_features / (gather_counts + 1e-6) # (w,h,c)
                feat_3d_to_2d_list.append(bs_feat_3d_to_2d)

            feat_3d_to_2d = torch.stack(feat_3d_to_2d_list).permute(0,3,2,1)            

            feat_3d_to_2d = self.fuse2d_mlp(feat_3d_to_2d)
            new_image_features = self.fuse2d_conv(torch.cat([feat_3d_to_2d, image_features], dim=1))
            
            # 2d to 3d
            feat_2d_to_3d = torch.stack(point_image_features_list).permute(0,2,1)
            feat_2d_to_3d = self.fuse3d_before_mlps(feat_2d_to_3d)

            new_features = torch.cat([new_features, feat_2d_to_3d], dim=1)
            new_features = self.fuse3d_mlp(new_features)

        if self.confidence_mlp is not None:
            new_scores = self.confidence_mlp(new_features)
            new_scores = new_scores.squeeze(1)  # (B, npoint)
            return new_xyz, new_features, new_vis, new_scores, new_image_features

        return new_xyz, new_features, new_vis, None, new_image_features


class PointnetSAModuleFS(PointnetSAModuleFSMSG):
    """Pointnet set abstraction layer with fusion sampling"""

    def __init__(self, *,
                 mlp: List[int],
                 npoint_list: List[int] = None,
                 sample_range_list: List[List[int]] = None,
                 sample_method_list: List[str] = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 dilated_radius_group: bool = False,
                 skip_connection: bool = False,
                 weight_gamma: float = 1.0,
                 aggregation_mlp: List[int] = None,
                 confidence_mlp: List[int] = None):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint_list: list of int, number of samples for every sampling method
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling method
        :param sample_method_list: list of str, list of used sampling method, d-fps, f-fps or c-fps
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_radius_group: whether to use radius dilated group
        :param skip_connection: whether to add skip connection
        :param weight_gamma: gamma for s-fps, default: 1.0
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        """
        super().__init__(
            mlps=[mlp], npoint_list=npoint_list, sample_range_list=sample_range_list,
            sample_method_list=sample_method_list, radii=[radius], nsamples=[nsample],
            bn=bn, use_xyz=use_xyz, pool_method=pool_method, dilated_radius_group=dilated_radius_group,
            skip_connection=skip_connection, weight_gamma=weight_gamma,
            aggregation_mlp=aggregation_mlp, confidence_mlp=confidence_mlp
        )


class PointnetFPModuleWithImage(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], image_feature_channel: int=0, bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv1d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

        in_channels_2d = image_feature_channel
        in_channels_3d = mlp[-1]

        # image to points
        self.fuse3d_before_mlps = nn.Sequential(
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm='batch_norm'),
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm='batch_norm'),
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm='batch_norm'),
        )
        self.fuse3d_mlp = nn.Sequential(nn.Conv1d(in_channels_2d + in_channels_3d, in_channels_3d, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(in_channels_3d),
                                    nn.ReLU())

        # points to image
        self.fuse2d_mlp = nn.Sequential(
            # Conv2dNormRelu(in_channels_3d + 3, in_channels_3d, norm='batch_norm'),
            Conv2dNormRelu(in_channels_3d, in_channels_3d, norm='batch_norm'),
            Conv2dNormRelu(in_channels_3d, in_channels_3d, norm='batch_norm'),
            Conv2dNormRelu(in_channels_3d, in_channels_3d, norm='batch_norm'),
        )
        self.fuse2d_conv = Conv2dNormRelu(in_channels_2d + in_channels_3d, in_channels_2d)


    def forward(self,
                unknown: torch.Tensor,
                known: torch.Tensor,
                unknow_feats: torch.Tensor,
                known_feats: torch.Tensor,
                image_features: torch.Tensor,
                new_vis: torch.Tensor,
                world_scale=None,
                world_rotation=None,
                flip_along_x=None,
                V2R=None,
                P2=None,
                image_shape=None
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        # new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        # fuse point features and image features
        new_xyz = unknown
        xyz = known
        if image_features is not None:
            batch_size = new_features.shape[0]
            point_image_features_list = []
            image_point_features_list = []
            uv_list = []

            batch_size, _, image_features_h, image_features_w = image_features.shape
            
            # import pdb;pdb.set_trace()
            for bs_idx in range(batch_size):
                keypoints_b = new_xyz[bs_idx].clone()
                # if self.training:
                # restore the raw positions of keypoints, seq: scale->rotate->flip?
                if world_scale is not None:
                    world_scale_b = world_scale[bs_idx]
                    keypoints_b /= world_scale_b
                if world_rotation is not None:
                    world_rotation_b = world_rotation[bs_idx]
                    keypoints_b = common_utils.rotate_points_along_z_single(keypoints_b, -world_rotation_b)
                if flip_along_x is not None:
                    flip_along_x_b = flip_along_x[bs_idx] # ad hoc, only process flip_x
                    if flip_along_x_b:
                        keypoints_b[:, 1] = -keypoints_b[:, 1]

                # project keypoint to image
                keypoints_b_hom = torch.cat([keypoints_b, keypoints_b.new_ones(len(keypoints_b),1)], dim=-1)
                scan_C0 = torch.mm(keypoints_b_hom, V2R[bs_idx].T)
                scan_C2 = torch.mm(scan_C0, P2[bs_idx].T) # [N, 3]
                scan_C2_depth = scan_C2[:, 2]
                scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T

                # save debug images
                # import matplotlib.pyplot as plt
                # import numpy as np
                # _scan_C2 = scan_C2.cpu().numpy();_scan_C2_depth = scan_C2_depth.cpu().numpy()
                # img = images[bs_idx][:3].permute(1,2,0).cpu().numpy()
                # fig = plt.figure(figsize=(12, 6));plt.imshow(img)
                # inds = _scan_C2[:, 0] > 0;inds = np.logical_and(inds, _scan_C2[:, 0] < img.shape[1]);inds = np.logical_and(inds, _scan_C2[:, 1] > 0);inds = np.logical_and(inds, _scan_C2[:, 1] < img.shape[0]);inds = np.logical_and(inds, _scan_C2_depth > 0)
                # plt.scatter(_scan_C2[inds, 0], _scan_C2[inds, 1], c=-_scan_C2_depth[inds], alpha=1, s=2, cmap='viridis')
                # plt.axis('off');plt.tight_layout();plt.savefig('output.png', bbox_inches='tight')
                # import pdb;pdb.set_trace()

                scan_C2[:, 0] *= (image_features[bs_idx].shape[2]/image_shape[1]) # w
                scan_C2[:, 1] *= (image_features[bs_idx].shape[1]/image_shape[0]) # h
                uv_list.append(scan_C2)

                cur_image_features = image_features[bs_idx].permute(1, 2, 0)  # (C,H,W) -> (H, W, C)
                valid_mask = new_vis[bs_idx].flatten() 
                cur_point_image_features = torch.zeros((len(valid_mask), cur_image_features.shape[-1]), dtype=cur_image_features.dtype, device=cur_image_features.device)
                valid_cur_point_image_features = common_utils.bilinear_interpolate_torch(cur_image_features, scan_C2[valid_mask==1, 0], scan_C2[valid_mask==1, 1])
                cur_point_image_features[valid_mask==1, :] = valid_cur_point_image_features
                point_image_features_list.append(cur_point_image_features)

            feat_3d_to_2d_list = []
            for bs_idx in range(batch_size):
                bs_uv = uv_list[bs_idx]
                bs_uv_grid = torch.floor(bs_uv).int()
                valid_point_mask = (bs_uv[:, 0]>=0) & (bs_uv[:, 0]<image_features[0].shape[2]) & (bs_uv[:, 1]>=0) & (bs_uv[:, 1]<image_features[0].shape[1])

                # import pdb;pdb.set_trace()
                valid_point_mask = valid_point_mask & (new_vis[bs_idx]>0).flatten()

                bs_uv_grid = bs_uv_grid[valid_point_mask].transpose(0,1)
                bs_new_features = new_features[bs_idx].transpose(0,1)[valid_point_mask, :]
                mask_value = torch.ones([bs_new_features.shape[0],1], device=bs_new_features.device)

                gather_3d_features = torch.sparse_coo_tensor(bs_uv_grid, bs_new_features, (image_features_w, image_features_h, new_features.shape[1]))
                gather_3d_features = gather_3d_features.to_dense()
                gather_counts = torch.sparse_coo_tensor(bs_uv_grid, mask_value, (image_features_w, image_features_h, 1))
                gather_counts = gather_counts.to_dense()
                bs_feat_3d_to_2d = gather_3d_features / (gather_counts + 1e-6) # (w,h,c)
                feat_3d_to_2d_list.append(bs_feat_3d_to_2d)

            feat_3d_to_2d = torch.stack(feat_3d_to_2d_list).permute(0,3,2,1)

            feat_3d_to_2d = self.fuse2d_mlp(feat_3d_to_2d)
            # import pdb;pdb.set_trace()
            new_image_features = self.fuse2d_conv(torch.cat([feat_3d_to_2d, image_features], dim=1))
            
            # 2d to 3d
            feat_2d_to_3d = torch.stack(point_image_features_list).permute(0,2,1)
            feat_2d_to_3d = self.fuse3d_before_mlps(feat_2d_to_3d)

            new_features = torch.cat([new_features, feat_2d_to_3d], dim=1)
            new_features = self.fuse3d_mlp(new_features)


        return new_features, new_image_features
        
if __name__ == "__main__":
    pass
