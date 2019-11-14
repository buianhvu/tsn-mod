import torch.nn as nn
from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS
import torch

@RECOGNIZERS.register_module
class TSN2D(BaseRecognizer):

    def __init__(self,
                 backbone,
                 backbone1,
                 backbone2,
                 modality='RGB',
                 in_channels=3,
                 spatial_temporal_module=None,
                 spatial_temporal_module1=None,
                 spatial_temporal_module2=None,
                 segmental_consensus=None,
                 segmental_consensus1=None,
                 segmental_consensus2=None,
                 cls_head=None,
                 cls_head1=None,
                 cls_head2=None,
                 train_cfg=None,
                 test_cfg=None):

        super(TSN2D, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.backbone1 = builder.build_backbone1(backbone1)
        self.backbone2 = builder.build_backbone2(backbone2)
        self.modality = modality
        self.in_channels = in_channels

        if spatial_temporal_module is not None:
            self.spatial_temporal_module = builder.build_spatial_temporal_module(
                spatial_temporal_module)
        else:
            raise NotImplementedError
        #add spatial module 1
        if spatial_temporal_module1 is not None:
            self.spatial_temporal_module1 = builder.build_spatial_temporal_module1(
                spatial_temporal_module1)
        else:
            raise NotImplementedError
        
        #add spatial module 2
        if spatial_temporal_module2 is not None:
            self.spatial_temporal_module2 = builder.build_spatial_temporal_module2(
                spatial_temporal_module2)
        else:
            raise NotImplementedError
        
        if segmental_consensus is not None:
            self.segmental_consensus = builder.build_segmental_consensus(
                segmental_consensus)
        else:
            raise NotImplementedError

        #add segmental_consensus1
        if segmental_consensus1 is not None:
            self.segmental_consensus1 = builder.build_segmental_consensus1(
                segmental_consensus1)
        else:
            raise NotImplementedError

        #add segmental_consensus2
        if segmental_consensus1 is not None:
            self.segmental_consensus2 = builder.build_segmental_consensus2(
                segmental_consensus2)
        else:
            raise NotImplementedError


        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        else:
            raise NotImplementedError

        if cls_head1 is not None:
            self.cls_head1 = builder.build_head1(cls_head1)
        else:
            raise NotImplementedError

        if cls_head2 is not None:
            self.cls_head2 = builder.build_head2(cls_head2)
        else:
            raise NotImplementedError

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert modality in ['RGB', 'Flow', 'RGBDiff']

        self.init_weights()

        if modality == 'Flow' or modality == 'RGBDiff':
            self._construct_2d_backbone_conv1(in_channels)

    @property
    def with_spatial_temporal_module(self):
        return hasattr(self, 'spatial_temporal_module') and self.spatial_temporal_module is not None

    @property
    def with_spatial_temporal_module1(self):
        return hasattr(self, 'spatial_temporal_module1') and self.spatial_temporal_module1 is not None

    @property
    def with_spatial_temporal_module2(self):
        return hasattr(self, 'spatial_temporal_module2') and self.spatial_temporal_module2 is not None


    @property
    def with_segmental_consensus(self):
        return hasattr(self, 'segmental_consensus') and self.segmental_consensus is not None

    @property
    def with_segmental_consensus1(self):
        return hasattr(self, 'segmental_consensus1') and self.segmental_consensus1 is not None

    @property
    def with_segmental_consensus2(self):
        return hasattr(self, 'segmental_consensus2') and self.segmental_consensus2 is not None


    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @property
    def with_cls_head1(self):
        return hasattr(self, 'cls_head1') and self.cls_head1 is not None

    @property
    def with_cls_head2(self):
        return hasattr(self, 'cls_head2') and self.cls_head2 is not None

    def _construct_2d_backbone_conv1(self, in_channels):
        modules = list(self.backbone.modules())
        first_conv_idx = list(filter(lambda x: isinstance(
            modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (in_channels, ) + kernel_size[2:]
        new_kernel_data = params[0].data.mean(dim=1, keepdim=True).expand(
            new_kernel_size).contiguous()  # make contiguous!

        new_conv_layer = nn.Conv2d(in_channels, conv_layer.out_channels,
                                   conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                                   bias=True if len(params) == 2 else False)
        new_conv_layer.weight.data = new_kernel_data
        if len(params) == 2:
            new_conv_layer.bias.data = params[1].data
        # remove ".weight" suffix to get the layer layer_name
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv_layer)

    def init_weights(self):
        super(TSN2D, self).init_weights()
        self.backbone.init_weights()

        if self.with_spatial_temporal_module:
            self.spatial_temporal_module.init_weights()

        if self.with_spatial_temporal_module1:
            self.spatial_temporal_module1.init_weights()

        if self.with_spatial_temporal_module2:
            self.spatial_temporal_module2.init_weights()

        if self.with_segmental_consensus:
            self.segmental_consensus.init_weights()

        if self.with_segmental_consensus1:
            self.segmental_consensus1.init_weights()

        if self.with_segmental_consensus2:
            self.segmental_consensus2.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

        if self.with_cls_head1:
            self.cls_head1.init_weights()

        if self.with_cls_head2:
            self.cls_head2.init_weights()

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x

    def extract_feat1(self, img_group):
        x = self.backbone1(img_group)
        return x

    def extract_feat2(self, img_group):
        x = self.backbone2(img_group)
        return x

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_label,
                      **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']
        list_sub_tensors = torch.chunk(img_group, 3, dim=1)
        img_group, img_group1, img_group2 = list_sub_tensors[1], list_sub_tensors[1], list_sub_tensors[2]
        print("list of subs type: ", type(list_sub_tensors))
        print("shape0: ", img_group.shape)
        print("shape01 ", img_group1.shape)
        print("shape02: ", img_group2.shape)

        print("\n\n\n")
        print("type of img_group: ", type(img_group))
        print("shape of img_group: ", img_group.shape)
        print("\n\n\n")
        # exit(1)

        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[3:])
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)
        x1 = self.extract_feat(img_group)
        x2 = self.extract_feat(img_group)

        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)

        if self.with_spatial_temporal_module1:
            x1 = self.spatial_temporal_module1(x1)

        if self.with_spatial_temporal_module2:
            x2 = self.spatial_temporal_module2(x2)
        
        x = x.reshape((-1, num_seg) + x.shape[1:])
        x1 = x1.reshape((-1, num_seg) + x1.shape[1:])
        x2 = x2.reshape((-1, num_seg) + x2.shape[1:])

        if self.with_segmental_consensus:
            x = self.segmental_consensus(x)
            x = x.squeeze(1)

        if self.with_segmental_consensus1:
            x1 = self.segmental_consensus1(x1)
            x1 = x1.squeeze(1)

        if self.with_segmental_consensus2:
            x2 = self.segmental_consensus2(x2)
            x2 = x2.squeeze(1)

        print("shape of x after consensus: ", x.shape)
        # exit(1)
        #we may have 3 losses for 3 streams
        losses = dict()
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            # print("SHAPE OF CLASS SCORE: ", cls_score.shape)
            # exit(1)
            gt_label = gt_label.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            # losses.update(loss_cls)
        if self.with_cls_head1:
            cls_score1 = self.cls_head1(x1)
            gt_label = gt_label.squeeze()
            loss_cls1 = self.cls_head1.loss(cls_score1, gt_label)
            # losses.update(loss_cls)
        if self.with_cls_head2:
            cls_score2 = self.cls_head2(x2)
            gt_label = gt_label.squeeze()
            loss_cls2 = self.cls_head2.loss(cls_score2, gt_label)
            # losses.update(loss_cls)
        print("Type of loss: ", type(loss_cls))
        print("Print a loss1: ", loss_cls)
        print("Print a loss1: ", loss_cls1)
        print("Print a loss2: ", loss_cls2)
        tens = loss_cls['loss_cls']
        tens1 = loss_cls1['loss_cls']
        tens2 = loss_cls2['loss_cls']
        dict_to_update = {}
        # print("Tens 1: ", tens1)
        # print("Tens 2: ", tens2)
        ten = tens + tens1 + tens2
        dict_to_update['loss_cls'] = ten
        print("Ten : ", ten)
        # exit(1)
        # sum_loss = loss_cls + loss_cls1 + loss_cls2
        losses.update(dict_to_update)
        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']
        list_sub_tensors = torch.chunk(img_group, 3, dim=1)
        img_group, img_group1, img_group2 = list_sub_tensors[1], list_sub_tensors[1], list_sub_tensors[2]
        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[3:])
        img_group1 = img_group1.reshape(
            (-1, self.in_channels) + img_group1.shape[3:])
        img_group2 = img_group2.reshape(
            (-1, self.in_channels) + img_group2.shape[3:])
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)
        x1 = self.extract_feat1(img_group1)
        x2 = self.extract_feat2(img_group2)

        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        x = x.reshape((-1, num_seg) + x.shape[1:])

        if self.with_spatial_temporal_module1:
            x1 = self.spatial_temporal_module1(x1)
        x1 = x1.reshape((-1, num_seg) + x1.shape[1:])

        if self.with_spatial_temporal_module2:
            x2 = self.spatial_temporal_module2(x2)
        x2 = x2.reshape((-1, num_seg) + x2.shape[1:])

        if self.with_segmental_consensus:
            x = self.segmental_consensus(x)
            x = x.squeeze(1)

        if self.with_segmental_consensus1:
            x1 = self.segmental_consensus1(x1)
            x1 = x1.squeeze(1)

        if self.with_segmental_consensus2:
            x2 = self.segmental_consensus2(x2)
            x2 = x2.squeeze(1)

        if self.with_cls_head:
            print("x shape: ", x.shape)
            x = self.cls_head(x)
            print("SHAPE OF x in test: ", x.shape)

        if self.with_cls_head1:
            print("x1 shape: ", x1.shape)
            x1 = self.cls_head1(x1)
            print("SHAPE OF x1 in test: ", x1.shape)


        if self.with_cls_head2:
            print("x2 shape: ", x2.shape)
            x2 = self.cls_head2(x2)
            print("SHAPE OF x2 in test: ", x2.shape)

        print("TYPE of x result: ", type(x))
        print("X shape: ", x.shape)
        print("X1 Shape: ", x1.shape)

        out_x = torch.cat((x, x1, x2), 0)
        out_x, locations = torch.max(out_x, 0)
        print("Shape of out_x: ", out_x.shape)
        print("OUT_X: ", out_x)
        # exit(1)
        #add a max-concesus pooling for 3 streams of 3 views before returning
        return out_x.cpu().numpy()
