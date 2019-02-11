# coding:utf-8
# this file contain following utils:
# 1. load_classes: read class string name from data/dataset.names
# 2. bbox_iou: calc iou.


import torch
import numpy as np

def load_classes(inp="data/coco.names"):
    """Load classes names from file as string list
    Args:
      inp: str, file which contains classes name
    Return:
      list[str], list contains classes name
    """
    return [c.strip() for c in open(inp)]

def bbox_iou(box1, box2, x1y1x2y2=True):
    """Calc IOU between bboxs.
    Args:
      box1: Tensor, size:(box_num, 4)
      box2: Tensor, size:(box_num, 4)
      x1y1x2y2: bool, if true box contain x1y1x2y2 else cx, cy w, h
                (x2, y2) is right bottom corner's position, away from image left top point.
                (x1, y1) is left top corner's position.
    Return:
      iou: Tensor, size: (box_num,)
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.max(b1_y2, b2_y2)

    inter_area = torch.clmap(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clmap(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou