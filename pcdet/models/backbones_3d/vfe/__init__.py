import imp
from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE,RadarPillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .rpfa_vfe import RpfaVFE
from .rpfa_maxpool_vfe import RpfaMaxpoolVFE
from .rpfa_avgpool_vfe import RpfaAvgpoolVFE
from .vod_rpfa_vfe import VodRpfaVFE
from .pfa_vfe import pfaVFE
from .image_point_vfe import ImagePointVFE

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'VodRpfaVFE': VodRpfaVFE,
    'RadarPillarVFE': RadarPillarVFE,
    'RpfaVFE': RpfaVFE,
    'RpfaMaxpoolVFE': RpfaMaxpoolVFE,
    'RpfaAvgpoolVFE': RpfaAvgpoolVFE,
    'pfaVFE': pfaVFE,
    'ImagePointVFE': ImagePointVFE
}
