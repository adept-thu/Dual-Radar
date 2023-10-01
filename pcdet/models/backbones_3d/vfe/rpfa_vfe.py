import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

class PCT(nn.Module):
    def __init__(self, channels):
        super(PCT, self).__init__()

        self.act = nn.ReLU()
        # self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.fc_relation = nn.Linear(channels+channels+1,1)
    def forward(self, x1,x2,x1_2):
       """
       x1 : 64,32
       x2 : 64,11
       x3 : 64,32

       output: 64,32
       """
       # print(x1.size()) # 64,1
       energy = torch.bmm(x1, x2.permute(0,2,1)) # 64,64
       energy = self.softmax(energy)
       # print("attention 1",energy.size()) #64 ,64

       attention = energy / (1e-9 + energy.sum(dim=1, keepdims=True))
       x_r = torch.bmm(attention, x1_2) # 64,11
       x_r = self.act(self.after_norm(x1 - self.trans_conv(x_r)))

       energy_2 = energy.permute(0,2,1)
       energy_total = torch.cat([energy,energy_2,x1],dim=2)
       energy_total = self.sigmoid(self.fc_relation(energy_total))


       x = energy_total * x1 + x_r
       return x

class PCT_start(nn.Module):
    def __init__(self, channels):
        super(PCT_start, self).__init__()

        self.act = nn.ReLU()
        # self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.sigmoid = nn.Sigmoid()
        self.fc_relation = nn.Linear(channels + channels + 32, 32)
    def forward(self, x1,x2,x1_2):
       """
       x1 : 11,32
       x2 : 32,11
       x3 : 11,32

       output: 64,32
       """
       # print(x1.size())  # 11,32

       energy = torch.bmm(x1, x2) # 64,64
       energy = self.softmax(energy)
       # print("attention",energy.size()) # 11, 11
       attention = energy / (1e-9 + energy.sum(dim=1, keepdims=True))
       x_r = torch.bmm(attention, x1_2) # 64,11
       x_r = self.act(self.after_norm(x1 - self.trans_conv(x_r)))

       energy_2 = energy.permute(0, 2, 1)
       energy_total = torch.cat([energy, energy_2, x1], dim=2)
       energy_total = self.sigmoid(self.fc_relation(energy_total))

       x = energy_total * x1 + x_r
       return x

class CCT(nn.Module):
    def __init__(self, inc,channels,step=4):
        super(CCT, self).__init__()
        self.linear = nn.Linear(inc, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // step, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // step, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels//step, 1)
        self.trans_conv = nn.Conv1d(channels//step, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,choice='nomax'):
        x = self.linear(x).permute(0, 2, 1) # b, 64, 32
        x_q = self.q_conv(x).permute(2, 0, 1)  # b, 16, 32-> 32,b,16
        x_k = self.k_conv(x).permute(2, 1, 0)  # b, 16, 32-> 32,16,b
        x_v = self.v_conv(x).permute(2, 0, 1)  # b, n, c
        energy = torch.bmm(x_k,x_q )  # b, c,c
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True)) # b, c,c
        # print(attention.size(),x_v.size())
        x_r = torch.bmm(x_v,attention).permute(1, 2, 0)  # b, c, n
        # print(x_r.size())
        x_r = self.act(self.after_norm(x - self.trans_conv(x_r)))
        x = x + x_r
        if choice=="nomax":
            # print('no-max----------------')
            return x
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x

class PCTrans(nn.Module):
    def __init__(self):
        super(PCTrans, self).__init__()

        self.pct = PCT(64)
        self.cct1 = CCT(inc=11, channels=64,step=4)
        self.cct1_2 = CCT(inc=11, channels=64,step=4)
        self.cct2 = CCT(inc=32, channels=64,step=4)

    def forward(self, features):
        # pctrans, input (b,11,32)

        features1 = self.cct1(features,choice="yes")  # b, 64,1
        features1_2 = self.cct1_2(features,choice="yes")  # b, 64,1
        # print("feature1",features1.size()) # 64,32 no-max
        features = features.permute(0, 2, 1)
        # print(features1_2.size())  #b, 11, 32  # 64,32 no-max
        features2 = self.cct2(features,choice="yes")  # b 64 , 1
        # print("feature2",features2.size())  # 64,11 no-max
        features = self.pct(features1, features2, features1_2)
        # print('outputpct',features.size())
        return features

class PCTrans_start(nn.Module):
    def __init__(self):
        super(PCTrans_start, self).__init__()

        self.pct_start = PCT_start(11)

        self.cct1 = CCT(inc=11, channels=11,step=4)
        self.cct1_2 = CCT(inc=11, channels=11,step=4)
        self.cct2 = CCT(inc=32, channels=32,step=4)

    def forward(self, features):
        # pctrans, input (b,11,32)

        features1 = self.cct1(features,choice="nomax")  # b, 64,1
        features1_2 = self.cct1_2(features,choice="nomax")  # b, 64,1
        # print("feature1",features1.size()) # 11,32 no-max
        features = features.permute(0, 2, 1)
        # print(features1_2.size())  #b, 11, 32  # 11,32 no-max
        features2 = self.cct2(features,choice="nomax")  # b 64 , 1
        # print("feature2",features2.size())  # 32,11 no-max
        features = self.pct_start(features1, features2, features1_2).permute(0,2,1)
        # print('outputpct',features.size())
        return features
        
class RpfaVFE(VFETemplate):
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

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.pctrans_start = PCTrans_start()
        # self.pctrans_start1 = PCTrans_start()
        self.pctrans = PCTrans()
        # self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.w1.data.fill_(0.5)
        # self.w2.data.fill_(0.5)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask #(M,32,13)

#        features1 = self.pct(features) * self.w1
#        features2 = self.cct(features) * self.w2
#        features = features1 + features2
        print("start input ",features.size()) # 32,11
        features = self.pctrans_start(features)
        # features = self.pctrans_start1(features)
        features = self.pctrans(features)
        # print(features.size(),'--------------------------------')
        features = features.view([features.size()[0], features.size()[1]])

        batch_dict['pillar_features'] = features
        return batch_dict
