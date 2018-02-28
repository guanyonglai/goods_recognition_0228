import os
import cv2
from darknet import *

# working dir
wd = '/home/tfl/workspace/project/YI/goods_recognition/'
# classes
classes = ['beer','beverage','instantnoodle','redwine','snack','springwater','yogurt']

def init_detector(model_cfg_name='yolo-voc',model_weights_name='yolo-voc',meta_name='goodid.data'):
    model_cfg_path = os.path.join(wd, 'cfg', '%s.cfg' % model_cfg_name)
    model_weights_path = os.path.join(wd, 'model', '%s.weights' % model_weights_name)
    meta_path = os.path.join(wd, 'cfg', '%s' % meta_name)

    if not os.path.exists(model_cfg_path):
        print('Model cfg file missing!')
        exit()
    if not os.path.exists(model_weights_path):
        print('Model weights file missing!')
        exit()
    if not os.path.exists(meta_path):
        print('Data meta file missing!')
        exit()

    # --load data
    net = load_net(model_cfg_path, model_weights_path, 0)
    meta = load_meta(meta_path)

    return net,meta


def det(im_path,net,meta,conf_thres=0.001,nms=0.45):

    # --goods detection
    res = detect(net, meta, im_path, thresh=conf_thres,nms=nms)

    # --parse results
    cls = -1
    conf = -1
    x = -1
    y = -1
    w = -1
    h = -1

    im = cv2.imread(im_path)
    (im_h, im_w, im_c) = im.shape

    results = []

    if len(res) == 0:
        return results

    for line in res:
        cls_name = line[0]
        cls = classes.index(cls_name)
        conf = line[1]
        bb = line[2]

        # convert bb
        x = bb[0] / im_w
        y = bb[1] / im_h
        w = bb[2] / im_w
        h = bb[3] / im_h

        results.append([cls,conf,x,y,w,h])

    return results
