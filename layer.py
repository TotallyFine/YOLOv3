# coding:utf-8
# This file contains following basic nn layers for darknet.
# 1. BasicConv: Conv2d+BN+LeakyReLU.
# 2. BasicLayer: residual connection among BasicConvs.
# 3. BasicPred: predicts detection result from feature map.
 

import math

import torch
import torch.nn as nn
import numpy as np

from utils import bbox_iou


class BasicConv(nn.Module):
    """Used for basic convolution. Conv2d+BN+LeakyReLU.
    """
    def __init__(self, ind, outd, kr_size, stride, padding, lr=0.1, bais=False):
        super().__init_()

        self.layers = nn.Sequential(
            nn.Conv2d(ind, outd, kr_size, stride, padding, bais=bais),
            nn.BatchNorm2d(outd),
            nn.LeakyReLU(lr)
        )

    def forward(self, x):
        return self.layers(x)


class BasicLayer(nn.Module):
    """Residual connect between conv_1 and conv_2 iteratively.
    """
    def __init_(self, conv_1, conv_2, times):
        """
        Args:
          conv_1: list, configurations of BasicConv.
          conv_2: list, configurations of BasicConv.
          times: int, iterations num, every iteration has one conv_1 and one conv_2.
        """
        super().__init_()

        self.layers = nn.ModuleList()
        for _ in range(times):
            self.layers.append(BasicConv(*conv_1))
            self.layers.append(BasicConv(*conv_2))

    def forward(self, x):
        residual = x
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index % 2 == 1: # get into conv_2, add residual
                x += residual
                residual = x
        return x


class BasicPred(nn.Module):
    """Prediction layers: predicts detection result from feature map.
    """
    def __init__(self, structs, use_cuda, anchors, classes, height=416, route_index=0):
        """
        Args:
          structs: list[in_dim, list[out_dim, kr_size, stride, padding],...], contains prediction layers' architecture.
          use_cuda: bool, whether to use cuda.
          anchors: list[(x, y),...], anchors on input image.
          classes: int, class num.
          height: int, height of input image which is square.
          route_index: int, if 0 means norml else output route_index layer's feature map.
        """
        super().__init__()

        self.ri = route_index
        self.classes = classes
        self.height = height
        self.anchors = anchors
        self.torch = torch.cuda if use_cuda else torch

        # for training
        self.mse_loss = nn.MSELoss() # corrdinate loss
        self.bce_loss = nn.BECLoss() # confidence loss
        self.ce_loss = nn.CrossEntropyLoss() # class loss

        in_dim = structs[0] # dimension of BasicPred's input
        self.layers = nn.ModuleList()
        for s in structs[1:]:
            if len(s) == 4:
                out_dim, kr_size, stride, padding = s
                layer = BasicConv(in_dim, out_dim, kr_size, stride, padding)
            else: # the last element used as a flag, means this layer is just Conv2d
                out_dim, kr_size, stride, padding, _ = s
                layer = nn.Conv2d(in_dim, out_dim, kr_size, stride, padding)
            
            in_dim = out_dim
            self.layers.append(layer)

    def forward(self, x):
        """Propogation forward.
        Args:
          x: Tensor, BasicPred's input.
        Returns:
          detections: Tensor, detection result.
          output: Tensor, size:(batchsize, 3*(num_classes+5), 13|26|52, 13|26|52)
        """
        for index, layer in enumerate(self.layers):
            x = layer(x)
            # besides detection result, also output previous layer's feautre map
            if self.ri != 0 and index == self.ri:
                output = x
        
        detections = self.predict_transform(x)

        if self.ri != 0:
            return detections, output
        else: # last detection layer doesn't need to route.
            return detections

    def predict_transform(self, inp):
        """Apply sigmoid and other operations to get final result.
        Args:
          inp: Tensor, size:(batchsize, 3*(num_classes+5), 13|26|52, 13|26|52)
               3 means 3 anchors, last two parameters are feature's size
        Returns:
          pred_boxes: Tensor, size:(batchsize, num_anchors, grid_size, grid_size, 4)
          pred_conf: Tensor, size:(batchsize, gride_size, grid_size)
          pred_cls: Tensor, size:(batchsize, grid_size, grid_size, classes)
        """
        bsz = inp.size(0) # batchsize
        grid_size = inp.size(2)
        stride = self.height // grid_size # is 32 generally
        bbox_attrs = 5 + self.classes # every box has these attributes
        num_anchors = len(self.anchors)
        
        # get prediction (batchsize, num_anchors, grid_size, grid_size, 5+classes)
        prediction = inp.view(bsz, num_anchors, bbox_attrs, grid_size, 
                              grid_size).permute(0, 1, 3, 4, 2).contiguous()
        
        # get anchors on feature map(prediction).
        anchors = self.torch.FloatTensor(
            [(a[0]/stride, a[1]/stride) for a in self.anchors]
        )

        x = torch.sigmoid(prediction[..., 0]) # apply sigmoid to tx, size: (batchsize, num_anchors, grid_size, grid_size)
        y = torch.sigmoid(prediction[..., 1]) # apply sigmoid to ty, size: (batchsize, num_anchors, grid_size, grid_size)
        w = prediction[..., 2] # tw, size: (barchsize, num_anchors, grid_size, grid_size)
        h = prediction[..., 3] # th, size: (batchsize, num_anchors, grid_size, grid_size)
        pred_conf = torch.sigmoid(prediction[..., 4]) # to, objectness, size: (batchsize, num_anchors, grid_size, grid_size)
        pred_cls = torch.sigmoid(prediction[..., 5]) # every class's confidence, size: (batchsize, num_anchors, grid_size, grid_size)

        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size]).type(self.torch.FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size]).type(self.torch.FloatTensor)

        anchor_w = anchors[:, 0].view((1, num_anchors, 1, 1))
        anchor_h = anchors[:, 1].view((1, num_anchors, 1, 1))

        pred_boxes = self.torch.FloatTensor(prediction[..., :4].shape) # (batchsize, num_anchors, grid_size, grid_size, 4)
        pred_boxes[..., 0] = x.data + grid_x # add distence which is each grid to left top corner.
        pred_boxes[..., 1] = y.data + grid_y # same as upper line
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w # implemention for first formulations in paper
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h # get box's width and height

        """
        output = torch.cat(
            (
                pred_boxes.view(bsz, -1, 4) * stride, # recover to input image's scale
                pred_conf.view(bsz, -1, 1),
                pred_cls.view(bsz, -1, self.classes),
            ),
            -1
        )
        """
        detections = [pred_boxes, pred_conf, pred_cls, anchors, x, y, w, h]
        return detections