import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # 高度的特征数
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

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
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # 将稀疏的tensor转化为密集tensor,[bacth_size, 128, 2, 200, 176]
        # 结合batch，spatial_shape、indice和feature将特征还原到密集tensor中对应位置
        spatial_features = encoded_spconv_tensor.dense()
        # batch_size，128，2，200，176
        N, C, D, H, W = spatial_features.shape
        #print("1111111111111111111111111",spatial_features.shape)
        """
        将密集的3D tensor reshape为2D鸟瞰图特征    
        将两个深度方向内的voxel特征拼接成一个 shape : (batch_size, 256, 200, 176)
        z轴方向上没有物体会堆叠在一起，这样做可以增大Z轴的感受野，
        同时加快网络的速度，减小后期检测头的设计难度
        """
        spatial_features = spatial_features.view(N, C * D, H, W)
        # 将特征和采样尺度加入batch_dict
        #print("1111111111111111111111111",spatial_features.shape)
        batch_dict['spatial_features'] = spatial_features
        # 特征图的下采样倍数 8倍
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
