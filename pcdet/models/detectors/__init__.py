from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .rpfa import  RpfaNet
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .pfa import pfaNet
from .IASSD import IASSD
from .point_3dssd import Point3DSSD
from .rdiou_net import RDIoUNet
from .CT3D import CT3D
from .CT3D_3CAT import CT3D_3CAT
from .voxel_rcnn_fusion import VoxelRCNNFusion

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'RpfaNet': RpfaNet,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'pfaNet': pfaNet,
    'IASSD': IASSD,
    'BiProDet': Point3DSSD,
    'RDIoUNet': RDIoUNet,
    'CT3D': CT3D,
    'CT3D_3CAT': CT3D_3CAT,
    'VoxelRCNNFusion': VoxelRCNNFusion
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
