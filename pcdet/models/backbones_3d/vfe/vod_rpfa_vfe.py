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
        # self.sigmoid = nn.Sigmoid()
        # self.fc_relation = nn.Linear(channels+channels+1,1)
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

    #    energy_2 = energy.permute(0,2,1)
    #    energy_total = torch.cat([energy,energy_2,x1],dim=2)
    #    energy_total = self.sigmoid(self.fc_relation(energy_total))


    #    x = energy_total * x1 + x_r
       x = x1 + x_r
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

        # self.sigmoid = nn.Sigmoid()
        # self.fc_relation = nn.Linear(channels + channels + 32, 32)
        # self.fc_relation = nn.Linear(channels + channels + 10, 10)
    def forward(self, x1,x2,x1_2):
       """
       x1 :   b 12 32
       x2 :   b 32 12
       x1_2 : b 12 32

       output: 64,32
       """
    
       #又一次注意力机制
       energy = torch.bmm(x1, x2) # (b,12,12)
       energy = self.softmax(energy)
       attention = energy / (1e-9 + energy.sum(dim=1, keepdims=True))
       x_r = torch.bmm(attention, x1_2) # (b,12,32)
       x_r = self.act(self.after_norm(x1 - self.trans_conv(x_r)))# (b,12,32) 相减后BN + RELU

    #    energy_2 = energy.permute(0, 2, 1)# (b,12,12)
    #    energy_total = torch.cat([energy, energy_2, x1], dim=2)#(b,12,12 + 12 + 32)->(b,12,56)
    #    energy_total = self.sigmoid(self.fc_relation(energy_total))# (b,12,32)

    #    x = energy_total * x1 + x_r # (b,12,32)
       x =  x1 + x_r
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
        x_v = self.v_conv(x).permute(2, 0, 1)  # b, 16, 32->32,b,16
        energy = torch.bmm(x_k,x_q )  # 32,16,16
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True)) # 32,16,16
        x_r = torch.bmm(x_v,attention).permute(1, 2, 0)  # 32,b,16->b,16,32
        x_r = self.act(self.after_norm(x - self.trans_conv(x_r)))#(b,64,32)
        x = x + x_r #(b,64,32)
        if choice=="nomax":
            # print('no-max----------------')
            return x
        x = torch.max(x, dim=2, keepdim=True)[0]#(b,64,1)
        return x

class PCTrans(nn.Module):
    def __init__(self):
        super(PCTrans, self).__init__()

        self.pct = PCT(64)
        self.cct1 = CCT(inc=13, channels=64,step=4)
        # self.cct1_2 = CCT(inc=13, channels=64,step=4)
        # # self.cct2 = CCT(inc=32, channels=64,step=4)
        # self.cct2 = CCT(inc=32, channels=64,step=4)

    def forward(self, features):
        # pctrans, input (b,12,32)

        features1 = self.cct1(features,choice="yes")  # b, 64,1
        features1_2 = features1
        features2 = features1
        
        # features1_2 = self.cct1_2(features,choice="yes")  # b, 64,1
        # features = features.permute(0, 2, 1)#(b,32,12)
        # features2 = self.cct2(features,choice="yes")  # b, 64 , 1
        features = self.pct(features1, features2, features1_2) #(b,12,32)
        return features

class PCTrans_start(nn.Module):
    def __init__(self):
        super(PCTrans_start, self).__init__()

        self.pct_start = PCT_start(13)

        self.cct1 = CCT(inc=13, channels=13,step=4)
        # self.cct1_2 = CCT(inc=13, channels=13,step=4)
        # self.cct2 = CCT(inc=32, channels=32,step=4)
        # self.cct2 = CCT(inc=32, channels=32,step=4)

        # self.pct_start = PCT_start(11)

        # self.cct1 = CCT(inc=11, channels=11,step=4)
        # self.cct1_2 = CCT(inc=11, channels=11,step=4)
        # self.cct2 = CCT(inc=32, channels=32,step=4)

    def forward(self, features):
        # pctrans, input (b,12,32)

        features1 = self.cct1(features,choice="nomax")  #  (b,12,32)   
        features1_2 = features1
        features2 = features1.permute(0, 2, 1)
        # features1_2 = self.cct1_2(features,choice="nomax")# (b,12,32)     
        # features = features.permute(0, 2, 1)   #features(b,32,12)         
        # features2 = self.cct2(features,choice="nomax")  #(b,32,12)  
        features = self.pct_start(features1, features2, features1_2).permute(0,2,1) #(b,12,32)
        # print('outputpct',features.size())
        return features
     



class VodRpfaVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)
        """
        VFE:
        NAME: VodRpfaVFE
        WITH_DISTANCE: False
        USE_ABSLOTE_XYZ: True
        USE_RCS: True
        USE_ELEVATION: True
        USE_VR_COMP: True
        USE_VR: True
        USE_VXYZ: False
        USE_TIME: False
        USE_NORM: True
        NUM_FILTERS: [64]
        """
        

        num_point_features = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION"]

        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            self.use_RCS = self.model_cfg.USE_RCS
            self.use_vr = self.model_cfg.USE_VR
            self.use_vr_comp = self.model_cfg.USE_VR_COMP
            self.use_time = self.model_cfg.USE_TIME
            self.use_elevation = self.model_cfg.USE_ELEVATION

        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']

        num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.rcs_ind = self.available_features.index('rcs')
        self.vr_ind = self.available_features.index('v_r')
        self.vr_comp_ind = self.available_features.index('v_r_comp')
        self.time_ind = self.available_features.index('time')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_RCS:  # add 1 if RCS is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.vr_comp_ind)

        if self.use_time:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

        ## LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)


        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
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

        # self.pct = PCT(64)

    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_cluster, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask #[4404, 32, 12]
        ###

        #n,32,10(kitti)

        # print('@@@@faeture size @@@@@@')
        # print(features.size())
        # rpfa
        features = self.pctrans_start(features)#(b,12,32)
        # features = self.pctrans_start1(features)
        features = self.pctrans(features)#(b,64,1)
        # print(features.size(),'--------------------------------')
        features = features.view([features.size()[0], features.size()[1]])#(b,64)



        # pct
        # features = self.pct(features)
        # features = features.view([features.size()[0], features.size()[1]])

        batch_dict['pillar_features'] = features#(b,64)
        return batch_dict


