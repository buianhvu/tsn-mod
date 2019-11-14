import mmcv
from torch import nn

from .registry import (BACKBONES, BACKBONES1, BACKBONES2, FLOWNETS, SPATIAL_TEMPORAL_MODULES, SPATIAL_TEMPORAL_MODULES1, SPATIAL_TEMPORAL_MODULES2,
                       SEGMENTAL_CONSENSUSES, SEGMENTAL_CONSENSUSES1, SEGMENTAL_CONSENSUSES2,HEADS, HEADS1, HEADS2,
                       RECOGNIZERS, DETECTORS, LOCALIZERS, ARCHITECTURES,
                       NECKS, ROI_EXTRACTORS)


def _build_module(cfg, registry, default_args):
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        print("\n\n\n")
        print("==========================================\n")
        print("OBJ type: ", obj_type)
        print("Registry dict: ", registry.module_dict)
        if obj_type not in registry.module_dict:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
        obj_type = registry.module_dict[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)

def build_backbone1(cfg):
    return build(cfg, BACKBONES1)

def build_backbone2(cfg):
    return build(cfg, BACKBONES2)


def build_flownet(cfg):
    return build(cfg, FLOWNETS)


def build_spatial_temporal_module(cfg):
    return build(cfg, SPATIAL_TEMPORAL_MODULES)

def build_spatial_temporal_module1(cfg):
    return build(cfg, SPATIAL_TEMPORAL_MODULES1)

def build_spatial_temporal_module2(cfg):
    return build(cfg, SPATIAL_TEMPORAL_MODULES2)

def build_segmental_consensus(cfg):
    return build(cfg, SEGMENTAL_CONSENSUSES)

def build_segmental_consensus1(cfg):
    return build(cfg, SEGMENTAL_CONSENSUSES1)

def build_segmental_consensus2(cfg):
    return build(cfg, SEGMENTAL_CONSENSUSES2)

def build_head(cfg):
    return build(cfg, HEADS)

def build_head1(cfg):
    return build(cfg, HEADS1)

def build_head2(cfg):
    return build(cfg, HEADS2)

def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, RECOGNIZERS,
                 dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_localizer(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, LOCALIZERS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_architecture(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, ARCHITECTURES,
                 dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)
