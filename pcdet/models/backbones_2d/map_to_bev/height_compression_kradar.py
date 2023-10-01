import torch.nn as nn
import torch


class HeightCompression_kradar(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # 高度的特征数
        self.bn1 = nn.BatchNorm2d(256,eps=1e-3, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(128,eps=1e-3, momentum=0.01)
        self.bn3 = nn.BatchNorm2d(512,eps=1e-3, momentum=0.01)
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(832,512,3,2,1)
        # self.conv1 = nn.Conv2d(672,256,3,2,1)
        self.conv2 = nn.Conv2d(832,512,3,2,1)
        # self.conv2 = nn.Conv2d(704,256,3,2,1)
        self.conv3 = nn.Conv2d(384,256,3,1,1)
        # self.conv3 = nn.Conv2d(320,256,3,1,1)
        self.conv4 = nn.Conv2d(512,256,3,2,1)
        self.conv5 = nn.Conv2d(512,256,3,1,1)
        self.conv6 = nn.Conv2d(256,128,3,1,1)
        self.conv7 = nn.Conv2d(256,128,3,1,1)
        self.conv8 = nn.Conv2d(256,128,3,1,1)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # 得到VoxelBackBone8x的输出特征
        # encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # # 将稀疏的tensor转化为密集tensor,[bacth_size, 128, 2, 200, 176]
        # # 结合batch，spatial_shape、indice和feature将特征还原到密集tensor中对应位置
        # spatial_features = encoded_spconv_tensor.dense()
        # # batch_size，128，2，200，176
        # N, C, D, H, W = spatial_features.shape
        # """
        # 将密集的3D tensor reshape为2D鸟瞰图特征    
        # 将两个深度方向内的voxel特征拼接成一个 shape : (batch_size, 256, 200, 176)
        # z轴方向上没有物体会堆叠在一起，这样做可以增大Z轴的感受野，
        # 同时加快网络的速度，减小后期检测头的设计难度
        # """
        # spatial_features = spatial_features.view(N, C * D, H, W)
        # # 将特征和采样尺度加入batch_dict
        # batch_dict['spatial_features'] = spatial_features
        # # 特征图的下采样倍数 8倍
        # batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        # ---------------------------------------------
        x_conv2 = batch_dict['multi_scale_3d_features']['x_conv2'] #[batch_size, 32, [21, 800, 704]]
        x_conv3 = batch_dict['multi_scale_3d_features']['x_conv3'] #[batch_size, 64, [11, 400, 352]]
        x_conv4 = batch_dict['multi_scale_3d_features']['x_conv4'] #[batch_size, 64, [5, 200, 176]]
        spatial_feature1 = x_conv2.dense()
        spatial_feature2 = x_conv3.dense()
        spatial_feature3 = x_conv4.dense()
        N1, C1, D1, H1, W1 = spatial_feature1.shape
        N2, C2, D2, H2, W2 = spatial_feature2.shape
        N3, C3, D3, H3, W3 = spatial_feature3.shape
        print(spatial_feature1.shape)
        print(spatial_feature2.shape)
        print(spatial_feature3.shape)
        spatial_feature1 = spatial_feature1.view(N1,C1 * D1,H1,W1)
        spatial_feature2 = spatial_feature2.view(N2,C2 * D2,H2,W2)
        spatial_feature3 = spatial_feature3.view(N3,C3 * D3,H3,W3)
        # conv1 = nn.Conv2d(C1 * D1,256,3,2,1).cuda() #卷积向下取整
        # conv2 = nn.Conv2d(C2 * D2,256,3,2,1).cuda()
        # conv3 = nn.Conv2d(C3 * D3,256,3,1,1).cuda()
        # conv4 = nn.Conv2d(256,128,3,2,1).cuda()
        # conv5 = nn.Conv2d(256,128,3,1,1).cuda()
        # conv6 = nn.Conv2d(256,128,3,1,1).cuda()
        x1 = self.relu(self.bn3(self.conv1(spatial_feature1)))
        x1 = self.relu(self.bn1(self.conv4(x1)))
        x1 = self.relu(self.bn2(self.conv7(x1)))
        # print("----------------------------")
        # print(x1.shape)
        x2 = self.relu(self.bn3(self.conv2(spatial_feature2)))
        x2 = self.relu(self.bn1(self.conv5(x2)))
        x2 = self.relu(self.bn2(self.conv8(x2)))
        # print(x2.shape)
        x3 = self.relu(self.bn1(self.conv3(spatial_feature3)))
        x3 = self.relu(self.bn2(self.conv6(x3)))
        # print(x3.shape)
        spatial_features = torch.cat([x1,x2,x3],dim=1)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = 8


        batch_dict['spatial_feature1'] = spatial_feature1
        batch_dict['spatial_feature2'] = spatial_feature2
        batch_dict['spatial_feature3'] = spatial_feature3
        batch_dict['spatial_features_stride1'] = batch_dict['multi_scale_3d_strides']['x_conv2']
        batch_dict['spatial_features_stride2'] = batch_dict['multi_scale_3d_strides']['x_conv3']
        batch_dict['spatial_features_stride3'] = batch_dict['multi_scale_3d_strides']['x_conv4']


        return batch_dict
