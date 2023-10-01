import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone_kradar(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        # 读取下采样层参数 LAYER_NUMS: [5, 5]  LAYER_STRIDES: [1, 2]  NUM_FILTERS: [128, 256]
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS #[5, 5]
            layer_strides = self.model_cfg.LAYER_STRIDES #[1, 2]
            num_filters = self.model_cfg.NUM_FILTERS #[128, 256]
        else:
            layer_nums = layer_strides = num_filters = []
        # 读取上采样层参数 UPSAMPLE_STRIDES: [1, 2]  NUM_UPSAMPLE_FILTERS: [256, 256]
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS #[1, 2]
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES #[256, 256]
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums) #2
        c_in_list = [input_channels, *num_filters[:-1]] #[256,128]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]): # 根据layer_nums堆叠卷积层 堆5层
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            # 在block中添加该层
            # *作用是：将列表解开成几个独立的参数，传入函数 # 类似的运算符还有两个星号(**)，是将字典解开成独立的元素作为形参
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features :(batch_size,256,200,176)
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        # 对不同的分支部分分别进行conv和deconv的操作
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            """
            SECOND中一共存在两个下采样分支，
            分支一: (batch,128,200,176)
            分支二: (batch,256,100,88)
            """

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

             # 如果存在deconv，则对经过conv的结果进行反卷积操作
            """
            SECOND中存在两个下采样，则分别对两个下采样分支进行反卷积操作
            分支一: (batch,128,200,176)-->(batch,256,200,176)
            分支二: (batch,256,100,88)-->(batch,256,200,176)
            """
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        # 将上采样结果在通道维度拼接
        if len(ups) > 1:
            """
            最终经过所有上采样层得到的2个尺度的的信息
            每个尺度的 shape 都是 (batch,256,200,176)
            在第一个维度上进行拼接得到x  维度是 (batch,512,200,176)
            """
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        # Fasle
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        # 将结果存储在spatial_features_2d中并返回
        data_dict['spatial_features_2d'] = x #(batch,512,200,176)

        return data_dict
