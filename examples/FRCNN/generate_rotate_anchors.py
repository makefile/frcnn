# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
from math import radians
# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# anchor:(w,h,angle)
def generate_anchors(base_size=16, ratios=[1.0/2, 1.0/5, 1.0/8],
                     scales=2**np.arange(3, 6),
		     angles=[radians(-60),radians(-30),radians(0),radians(30),radians(60),radians(90)]):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales X angles wrt a reference (15, 15, 0) window.
    """

    base_anchor = np.array([base_size, base_size, 1.0]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    scale_anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    anchors = np.vstack([_angle_enum(scale_anchors[i, :], angles) for i in range(scale_anchors.shape[0])])
 
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, angles):
    """
    Given a vector of widths (ws) and heights (hs) , output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    angles = angles[:, np.newaxis]
    anchors = np.hstack((ws,hs,angles))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    #w, h, x_ctr, y_ctr = _whctrs(anchor)
    w, h, angle = anchor
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    angles = np.array([angle] * len(ratios))
    anchors = _mkanchors(ws, hs, angles)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, angle = anchor
    ws = w * scales
    hs = h * scales
    angles = np.array([angle] * len(scales))
    return _mkanchors(ws, hs, angles)

def _angle_enum(anchor, angles):
    """
    Enumerate a set of anchors for each angle wrt an anchor.
    """
    w, h, angle = anchor
    ws = np.array([w] * len(angles))
    hs = np.array([h] * len(angles))
    return _mkanchors(ws, hs, np.array(angles))

if __name__ == '__main__':
    #import time
    #t = time.time()
    a = generate_anchors(base_size=8, ratios=[0.25,0.4])
    #print time.time() - t
    #print a
#    from IPython import embed; embed()
    for i in range(len(a)-1):
    	print("%6d, %6d, %10.3f,"%(a[i,0],a[i,1],a[i,2]))
    print("%6d, %6d, %10.3f"%(a[i+1,0],a[i+1,1],a[i+1,2]))
