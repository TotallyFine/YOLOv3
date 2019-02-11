# coding:utf-8
# YOLo v3 loss.

from __future__ import print_function
import math

import torch
import torch.nn as nn

import numpy as np
from collections import defaultdict

from config import YOLOConfig as opt
from utils import bbox_iou

LOSS_NAMES = opt.LOSS_NAMES

class Loss(nn.Module):
    def __init__(self):
        """
        Args:
          opt: YOLOConfig: configuration for YOLOv3.
        """
        super().__init__()

        self.mse_loss = nn.MSELoss()  # Coordinate loss
        self.bce_loss = nn.BCELoss()  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, input, targets):
        """
        Args:
          x: Tensor, darknet's output, [[pred_boxes, pred_conf, pred_cls],...], standard YOLov3 has three prediction layer, so len(x)=3.
          tarets: Tensor, [batchsize, max_obj, 5]
                  max_obj means num of objects in this image,
                  5 means tx, ty, tw, th and to.
        Return:
          losses:(loss:Tensor;
                  loss_x:float;
                  loss_y:float;
                  loss_w:float;
                  loss_h:float;
                  loss_conf:float;
                  loss_cls:float;
                  recall:float;
                  precision:float;
                 )
        """
        # {loss:, loss_x:, loss_y:, loss_w:, loss_h:, loss_conf:, loss_cls:, recall:, precision:}
        gather_losses = defaultdict(float) # corresponding to LOSS_NAMES.

        for pred_boxes, pred_conf, pred_cls, anchors, x, y, w, h in input:
            num_anchors = 3
            grid_size = pred_boxes.size(2)

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=anchors.cpu().data,
                num_anchors=num_anchors,
                num_classes=self.classes,
                grid_size=grid_size,
                ignore_thres=0.5,
                img_dim=self.height)

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = mask.type(self.torch.ByteTensor)
            conf_mask = conf_mask.type(self.torch.ByteTensor)

            # Handle target variables
            with torch.no_grad():
                tx = tx.type(self.torch.FloatTensor)
                ty = ty.type(self.torch.FloatTensor)
                tw = tw.type(self.torch.FloatTensor)
                th = th.type(self.torch.FloatTensor)
                tconf = tconf.type(self.torch.FloatTensor)
                tcls = tcls.type(self.torch.LongTensor)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            loss_cls = self.ce_loss(
                pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            losses = (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )
            
            for name, loss in zip(LOSS_NAMES, losses):
                gather_losses[name] += loss

        gather_losses["recall"] /= 3 # get average recall
        gather_losses["precision"] /= 3 # get average precision
            
        return losses

    def build_target(self, pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
        """
        Args:
          pred_boxes: Tensor, size:(batchsize, num_anchors, grid_size, grid_size, 4)
          pred_conf: Tensor, size:(batchsize, grid_size, grid_size)
          pred_cls: Tensor, size:(batchsize, grid_size, grid_size, classes)
          target: Tensor, size:(batchsize, max_obj, 5), ground truth.
          anchors: int, anchor num.
          num_classes: int, classes num.
          grid_size: width/height of feature map which predict layer apply on.
          ignore_thres: threshold of pred_conf to ingore background.
          img_dim: int, input image's width/height.
        Return:
          
        """
        nB = target.size(0) # batchsize
        nA = num_anchors
        nC = num_classes
        nG = grid_size
        mask = torch.zeros(nB, nA,)
        conf_mask = torch.ones(nB, nA, nG, nG)
        tx = torch.zeros(nB, nA, nG, nG)
        ty = torch.zeros(nB, nA, nG, nG)
        tw = torch.zeros(nB, nA, nG, nG)
        th = torch.zeros(nB, nA, nG, nG)
        tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
        tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

        nGT = 0
        nCorrect = 0
        for b in range(nB):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    # pad
                    continue
                nGT += 1
                # Convert to position relative to box
                gx = target[b, t, 1] * nG
                gy = target[b, t, 2] * nG
                gw = target[b, t, 3] * nG
                gh = target[b, t, 4] * nG
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor(
                    np.array([0, 0, gw, gh])).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate(
                    (np.zeros((len(anchors), 2)), np.array(anchors)), 1))

                # Calculate iou between gt and anchor shapes
                # 1 on 3
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
                # Find the best matching anchor box

                best_n = np.argmax(anch_ious)
                # Get ground truth box
                gt_box = torch.FloatTensor(
                    np.array([gx, gy, gw, gh])).unsqueeze(0)
                # Get the best prediction
                pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
                # Masks
                mask[b, best_n, gj, gi] = 1
                conf_mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = math.log(
                    gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(
                    gh / anchors[best_n][1] + 1e-16)
                # One-hot encoding of label
                target_label = int(target[b, t, 0])
                tcls[b, best_n, gj, gi, target_label] = 1
                tconf[b, best_n, gj, gi] = 1

                # Calculate iou between ground truth and best matching prediction
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
                score = pred_conf[b, best_n, gj, gi]
                if iou > 0.5 and pred_label == target_label and score > 0.5:
                    nCorrect += 1

        return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls