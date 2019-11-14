from .tenons.backbones import *
from .tenons.spatial_temporal_modules import *
from .tenons.segmental_consensuses import *
from .tenons.cls_heads import * 
from .recognizers import *
from .tenons.necks import *
from .tenons.roi_extractors import *
from .tenons.anchor_heads import *
from .tenons.shared_heads import *
from .tenons.bbox_heads import *
from .detectors import *
from .localizers import *


from .registry import (BACKBONES, BACKBONES1, BACKBONES2, SPATIAL_TEMPORAL_MODULES, SPATIAL_TEMPORAL_MODULES1, SPATIAL_TEMPORAL_MODULES2, SEGMENTAL_CONSENSUSES, SEGMENTAL_CONSENSUSES1, SEGMENTAL_CONSENSUSES2, HEADS,HEADS1,HEADS2,
                       RECOGNIZERS, LOCALIZERS, DETECTORS, ARCHITECTURES,
                       NECKS, ROI_EXTRACTORS)
from .builder import (build_backbone, build_backbone1, build_backbone2, build_spatial_temporal_module, build_spatial_temporal_module1, build_spatial_temporal_module2, build_segmental_consensus, build_segmental_consensus1, build_segmental_consensus2, 
                      build_head, build_head1, build_head2, build_recognizer, build_detector,
                      build_localizer, build_architecture,
                      build_neck, build_roi_extractor)

__all__ = [
    'BACKBONES', 'BACKBONES1', 'BACKBONES2', 'SPATIAL_TEMPORAL_MODULES', 'SPATIAL_TEMPORAL_MODULES1','SPATIAL_TEMPORAL_MODULES2', 'SEGMENTAL_CONSENSUSES','SEGMENTAL_CONSENSUSES1','SEGMENTAL_CONSENSUSES2', 'HEADS', 'HEADS1', 'HEADS2',
    'RECOGNIZERS', 'LOCALIZERS', 'DETECTORS', 'ARCHITECTURES',
    'NECKS', 'ROI_EXTRACTORS',
    'build_backbone', 'build_spatial_temporal_module', 'build_spatial_temporal_module1' 'build_segmental_consensus',
    'build_head', 'build_recognizer', 'build_detector',
    'build_localizer', 'build_architecture',
    'build_neck', 'build_roi_extractor'
]
