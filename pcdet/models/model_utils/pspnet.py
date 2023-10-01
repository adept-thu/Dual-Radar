import torch
from torch import nn
from torch.nn import functional as F
import pcdet.models.model_utils.resnet_utils as extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False, ratio=1):
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained,ratio)
        self.psp = PSPModule(int(psp_size*ratio), int(1024*ratio), sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(int(1024*ratio), 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.final = nn.Sequential(
        #     nn.Conv2d(64, n_classes, kernel_size=1),
        #     nn.LogSoftmax()
        # )

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )

    def forward(self, x):
        f, feature_dict = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)
        # feature_dict['deconv_s8'] = p
        p = self.up_1(p)
        p = self.drop_2(p)
        # feature_dict['deconv_s4'] = p

        p = self.up_2(p)
        p = self.drop_2(p)
        # feature_dict['deconv_s2'] = p

        p = self.up_3(p)
        # feature_dict['deconv_s1'] = p
        
        # return self.final(p), feature_dict
        return self.final(p)

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class PSPModel(nn.Module):

    def __init__(self, n_classes=2, input_channels=3, ratio=1):
        super(PSPModel, self).__init__()
        assert input_channels==3
        # assert model_cfg.BACKBONE.depth==18
        # self.model = psp_models['resnet{}'.format(str(model_cfg.BACKBONE.depth))](ratio=model_cfg.BACKBONE.ratio)
        self.model = PSPNet(n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', ratio=ratio)

    def forward(self, x):
        x, feature_dict = self.model(x)
        return x, feature_dict
    
    def get_output_feature_dim(self):
        # ad hoc
        return 32

if __name__ == '__main__':
    import torch
    from pcdet.config import cfg
    from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file

    cfg_from_yaml_file("/home/yzhang3362/git/OpenPCDet/tools/cfgs/kitti_models/bfdet.yaml", cfg)
    model = PSPModel(cfg.MODEL.BACKBONE_IMAGE, input_channels=3).eval()

    images = torch.rand(1,3,375, 1242)
    data_dict={}
    data_dict['images'] = images
    outputs = model(data_dict)
    print(f"outputs.shape = {outputs['image_features'].shape}")
