import numpy as np
import sys, os, argparse
from base_function import parse_args
from base_function import prepare_data
from base_function import check_data
from base_function import voc_ap
from base_function import cal_ap
from base_function import change_addition2attr
from base_function import Get_CLASSES
from base_function import convert_allbox

"""
Grount Truth File Format:
# id
image_name
number of boxes
label x1 y1 x2 y2 diff
....

Results File Format:
# id
image_name
number of boxes
label x1 y1 x2 y2 confidence
....
"""

if __name__ == '__main__':
    # For 2/7/10 classes
    num_class = 7 # Does not include background
    CLASSES = Get_CLASSES(num_class)
    num_class = num_class + 1
    assert (num_class == len(CLASSES)), 'num_class == len(CLASSES)'

    args = parse_args()
    ovthresh = args.overlap
    images_gt, results_gt = prepare_data(args.gt_file)
    GT_num_class = np.zeros(num_class, dtype=int)
    change_addition2attr(results_gt, 'difficult', type=int)
    total_image = len(images_gt)
    for index in range(total_image):
        for cls in range(num_class):
            R = [obj for obj in results_gt[index] if obj['cls'] == cls]
            GT_num_class[cls] += len(R)
        if index % 2000 == 0:
            print('[GT] Count {} / {} images').format(index, total_image)

    ans_fils = args.ans_file
    for file_ in ans_fils:
        print '\n************************************** : {} vs Ground Truth {}'.format(file_, args.gt_file)
        images_res, results_res = prepare_data(file_)
        check_data(images_gt, images_res, results_gt, results_res)
        change_addition2attr(results_res, 'confidence', type=float)
        RES_num_class, all_boxes = convert_allbox(results_res, num_class)

        for index in range(num_class):
            print '{} has {} Ground Truth, {} Predicted Results'.format(CLASSES[index], GT_num_class[index], RES_num_class[index])

        AP = np.zeros(num_class)

        for cls in range(1,num_class,1): # Ingore __background__
            rec, prec, AP[cls] = cal_ap(all_boxes[cls], results_gt, cls, ovthresh)
            print '%9s : AP = %.5f , Recall = %.4f' %(CLASSES[cls], float(AP[cls]), float(rec[-1]))

        print 'Mean AP = {}'.format(np.mean(AP[1:]))
