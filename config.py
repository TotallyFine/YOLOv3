# coding:utf-8

class YOLOConfig(object):
    use_cuda = False

    # MODEL RELATED #
    img_size = 416 # input image's size
    
    num_anchors = 3 # YOLOv3 only use 3 anchor on every grid
    num_classes = 80 # coco's class num is 80, voc is

    ingore_thres = 1 #range from 0 ~ 1, for objectness

    OUT_DIM = 3 * (num_classes + 5) # prediction layer's output dimension

    # three prediction layer's struct
    # 3 list[in_dim, list[out_dim, kr_size, stride, padding],...]
    DETECT_DICT = {
        'first': [1024, (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1), (512, 1, 1, 0), (1024, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)],
        'second': [768, (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), (256, 1, 1, 0), (512, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)],
        'third': [384, (128, 1, 1, 0), (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), (128, 1, 1, 0), (256, 3, 1, 1), (OUT_DIM, 1, 1, 0, 0)],
    }

    # calc floowing loss in loss layer.
    LOSS_NAMES = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    # TRAIN RELATED #
    epoch = 30
    batch_size = 16
