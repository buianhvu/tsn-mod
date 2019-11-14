import torch.nn as nn


class Registry(object):

    def __init__(self, name):
        # print('asdasdas Name', name)
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class):
        # print('-------- module', module_class.__name__)
        """Register a module

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not issubclass(module_class, nn.Module):
            raise TypeError(
                'module must be a child of nn.Module, but got {}'.format(
                    module_class))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        print('------asdkashdasd----', cls)
        self._register_module(cls)
        return cls


BACKBONES = Registry('backbone')
BACKBONES1 = Registry('backbone1')
BACKBONES2 = Registry('backbone2')

FLOWNETS = Registry('flownet')
SPATIAL_TEMPORAL_MODULES = Registry('spatial_temporal_module')
SPATIAL_TEMPORAL_MODULES1 = Registry('spatial_temporal_module')
SPATIAL_TEMPORAL_MODULES2 = Registry('spatial_temporal_module')

SEGMENTAL_CONSENSUSES = Registry('segmental_consensus')
SEGMENTAL_CONSENSUSES1 = Registry('segmental_consensus1')
SEGMENTAL_CONSENSUSES2 = Registry('segmental_consensus2')


HEADS = Registry('head')
HEADS1 = Registry('head1')
HEADS2 = Registry('head2')

RECOGNIZERS = Registry('recognizer')
LOCALIZERS = Registry('localizer')
DETECTORS = Registry('detector')
ARCHITECTURES = Registry('architecture')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
