# coding:utf-8
# This script defines darknet's architecture.

from __future__ import print_function
import torch
import torch.nn as nn

from collections import defaultdict

from layer import *
from utils import load_classes

from config import YOLOConfig as opt

OUT_DIM = opt.OUT_DIM # each grid's output dimension

# three prediction layer's structs
# list[in_dim, list[out_dim, kr_size, stride, padding],...]
DETECT_DICT = opt.DETECT_DICT


class LayerOne(BasicLayer):
    """First residual block, see YOLOv3-arch.txt
    All BasicLayer is not reduce feautre map size, only change channel.
    """
    def __init__(self):
        super().__init__((64, 32, 1, 1, 0),
                         (32, 64, 3, 1, 1), 1)


class LayerTwo(BasicLayer):
    def __init_(self):
        super().__init_((128, 64, 1, 1, 0),
                        (64, 128, 3, 1, 1), 2)
                        

class LayerThree(BasicLayer):
    def __init__(self):
        super().__init__((256, 128, 1, 1, 0),
                         (128, 256, 3, 1, 1), 8)


class LayerFour(BasicLayer):
    def __init__(self):
        super().__init__((512, 256, 1, 1, 0),
                         (256, 512, 3, 1, 1), 8)


class LayerFive(BasicLayer):
    def __init__(self):
        super().__init__((1024, 512, 1, 1, 0),
                         (512, 1024, 3, 1, 1), 4)


class FirstPred(BasicPred):
    """First prediction layer, see YOLOv3-arch.txt
    """
    def __init__(self,
                 structs,
                 use_cuda,
                 classes,
                 route_index=4,
                 anchors=[(116, 90), (156, 198), (373, 326)]):
        super().__init__(structs, use_cuda, anchors, classes, route_index=route_index)


class SecondPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 classes,
                 route_index=4,
                 anchors=[(30, 61), (62, 45), (59, 119)]):
        super().__init__(structs, use_cuda, anchors, classes, route_index=route_index)


class ThirdPred(BasicPred):
    def __init__(self,
                 structs,
                 use_cuda,
                 classes,
                 height=416,
                 anchors=[(10, 13), (16, 30), (33, 23)]):
        super().__init__(structs, use_cuda, anchors, classes)


class DarkNet(nn.Module):
    """Darknet architecture.
    """
    def __init__(self, use_cuda, nClasses):
        """
        Args:
          use_cuda: bool, wether to use cuda.
          nClasses: classes num.
        """
        super().__init_()

        self.seq_1 = nn.Sequential(
            BasicConv(3, 32, 3, 1, 1), # no.0 layer in YOLOv3-arch.txt
            BasicConv(32, 64, 3, 2, 1), # no.1 layer
            LayerOne(), # no.2~4 layer
            BasicConv(64, 128, 3, 2, 1), # no.5 layer
            LayerTwo(), # no.6~11 layer
            BasicConv(128, 256, 3, 2, 1), # no.12 layer
            LayerThree(), # no.13~36 layer
        )
        self.conv_1 = BasicConv(256, 512, 3, 2, 1) # no.37 layer
        self.layer_4 = LayerFour() # no.38~61 layer

        self.seq_2 = nn.Sequential(
            BasicConv(512, 1024, 3, 2, 1), # no.62 layer
            LayerFive() # no.63~74 layer
        )

        self.pred_1 = FirstPred(DETECT_DICT["first"], use_cuda, nClasses) # no.75~82 layer

        self.uns_1 = nn.Sequential(
            BasicConv(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode="bilinear")
        ) # no.83~85 layer

        # no.86 layer is concate

        self.pred_2 = SecondPred(DETECT_DICT["second"], use_cuda, nClasses) # no.87~94 layer
        
        self.uns_2 = nn.Sequential(
            BasicConv(256, 128, 1, 1, 0), # no.96 layer
            nn.Upsample(scale_factor=2, mode="bilinear")
        ) # no.95~97 layer

        # no.98 layer is concate

        self.pred_3 = ThirdPred(DETECT_DICT["third"], use_cuda, nClasses) # no.99~106 layer

        self._init_parameters()

    def _init_parameters(self):
        """init nn parameters randomly.
        """
        for layer in self.modules():
            if type(layer) == nn.Conv2d:
                layer.weight.data.normal_(0.0, 0.02)

            if type(layer) == nn.BatchNorm2d:
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, x):
        """DarkNet propogation forward.
        Args:
          x: Tensor, size:(3, 416, 416), original input image
        Returns:
          output: list, [det_1, det_2, det_3], det_1:[pred_boxes, pred_conf, pred_cls]
                  this output is Loss layer's input, see Loss.py
        """
        x = self.seq_1(x) # size: (batchsize, 256, 52, 52), no.36 layer
        r_0 = x

        x = self.layer_4(self.conv_1(x)) # size: (batchsize, 512 ,26, 26)
        r_1 = x

        x = self.seq_2(x) # size: (batchszie, 1024, 13, 13)

        # x's size: (batchsize, 255, 13, 13)
        # det_1: (pred_boxes, pred_conf, pred_cls)
        det_1, x = self.pred_1(x)

        x = self.uns_1(x) # upsampling, output size: (batchsize, 256, 26, 26)
        x = torch.cat((x, r_1), 1) # no.86 layer, size: (batchsize, 768, 26,26)

        # x's size: (batchsize, 255, 26, 26)
        # det_2: (pred_boxes, pred_conf, pred_cls)
        det_2, x = self.pred_2(x)

        x = self.uns_2(x) # size: (batchsize, 128, 52, 52)
        x = torch.cat((x, r_0), 1) # size: (batchsize, 384, 52, 52)

        # det_3: (pred_boxes, pred_conf, pred_cls)
        det_3 = self.pred_3(x)

        return [det_1, det_2, det_3]

if __name__ == "__main__":
    model = DarkNet(False, 20)
    print("DarkNet:\n")
    print(model)
