import numpy as np
import sys, os, argparse

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
voc_classes = [
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor' ]
NWPU_classes = ['plane','ship','storage','harbor','bridge']
CLASSES = ['__background__'] + NWPU_classes

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Calculate Recall For Rpn Detector Results')
    parser.add_argument('--gt', dest='gt_file',
                        help='Ground Truth File',
                        default=None, type=str)
    parser.add_argument('--answer', dest='ans_file',
                        help='Rpn Results File',
                        default=None, type=str)
    parser.add_argument('--overlap', dest='overlap',
                        help='overlap value',
                        default=0.5, type=float)
    args = parser.parse_args()

    if args.gt_file is None or args.ans_file is None:
        parser.print_help()
        sys.exit(1)

    return args


# images ground_truth results = prepare_data()
def prepare_data(args):
    print 'Called with args:\n{}'.format(args)
    if not os.path.isfile(args.gt_file):
        print 'Ground Truth File({}) does not exist'.format(args.gt_file)
        sys.exit(1)
    if not os.path.isfile(args.ans_file):
        print 'Results File({}) does not exist'.format(args.ans_file)
        sys.exit(1)
    gt_file = open(args.gt_file)
    ans_file = open(args.ans_file)
    # Handle

    recs = []
    results = []
    images = []

    total = 0
    boxes_num_gt = 0
    boxes_num_res = 0
    while True:
        gt_line, ans_line = gt_file.readline(), ans_file.readline()
        if gt_line == '':
            assert (gt_line == ans_line), 'GT File Done But Results File unfinished, {}'.format(ans_line)
            break
        gt_line, ans_line = gt_line.strip('\r\n').split(' '), ans_line.strip('\r\n').split(' ')
        assert (len(gt_line) == 2 and gt_line[0] == '#'), 'First Line of GT File Must Be \'# ID\', not {}'.format(gt_line)
        assert (len(ans_line) == 2 and ans_line[0] == '#'), 'First Line of Result File Must Be \'# ID\''
        assert (ans_line[1] == gt_line[1]), 'ID of GT and Result Must Be Same: ID=%s -> "%s" != "%s"'%(total,gt_line[1],ans_line[1])
        gt_line, ans_line = gt_file.readline().strip('\r\n'), ans_file.readline().strip('\r\n')
        assert (gt_line == ans_line), 'Image Does not Euqal : {} and {}'.format(gt_line, ans_line)
        images.append(gt_line)

        gt_line, ans_line = gt_file.readline().strip('\r\n'), ans_file.readline().strip('\r\n')
        num_gt, num_ans = int(gt_line), int(ans_line)

        boxes_num_gt += num_gt
        boxes_num_res += num_ans
        # Get Ground Truth Boxes
        # gt_current_box = np.zeros((num_gt, 5))
        objects = []
        for index in range(num_gt):
            gt_line = gt_file.readline().strip('\r\n').split(' ')
            for jj in range(gt_line.count('')):
                gt_line.remove('')
            assert (len(gt_line) == 6), 'Ground Truth : label x1 y1 x2 y2 diff, not {}'.format(gt_line)
            obj_struct = {}
            obj_struct['bbox'] = [float(gt_line[1]),float(gt_line[2]), float(gt_line[3]), float(gt_line[4])]
            obj_struct['cls']  = int(gt_line[0])
            obj_struct['difficult'] = int(gt_line[5])
            objects.append(obj_struct)

        recs.append(objects)

        # Get Results Boxes
        res_current_box = np.zeros((num_ans, 6))
        for index in range(num_ans):
            ans_line = ans_file.readline().strip('\r\n').split(' ')
            for jj in range(ans_line.count('')):
                ans_line.remove('')
            assert (len(ans_line) == 6), 'Ground Truth : label x1 y1 x2 y2 confidence, not {}'.format(gt_line)
            res_current_box[index, :] = np.array([float(ans_line[0]), float(ans_line[1]), float(ans_line[2]),
                                                  float(ans_line[3]), float(ans_line[4]), float(ans_line[5])])
        results.append(res_current_box)

        total += 1
        if total % 1000 == 0:
            print('Process {} images').format(total)

    print('Total {} images load done').format(total)
    print('Total {} ground truth boxes load done').format(boxes_num_gt)
    print('Total {} result boxes load done').format(boxes_num_res)
    gt_file.close()
    ans_file.close()
    return images, recs, results

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def cal_ap(RES, gts, ccls):

    assert len(gts) == len(RES)
    # extract gt objects for this class
    class_recs = []
    npos = 0
    for index in xrange(len(gts)):
        R = [obj for obj in gts[index] if obj['cls'] == ccls]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs.append({'bbox': bbox,
                            'difficult': difficult,
                            'det': det})

    # read dets
    image_ids = []
    confidence = []
    BB = []
    for index in xrange(len(RES)):
        for obj in xrange(len(RES[index])):
            image_ids.append(index)
            confidence.append(RES[index][obj,4])
            BB.append(RES[index][obj,:4])

    confidence = np.array(confidence)
    BB = np.array(BB)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    #sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]


     # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
            #if True:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print 'npos : {}'.format(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True)

    return rec, prec, ap

if __name__ == '__main__':
    args = parse_args()
    images, gts, results = prepare_data(args)

    print('Prepare Data Done')
    ovthresh = args.overlap

    total_image = len(images)
    assert (total_image == len(gts))

    #num_class = 21
    #assert (num_class == len(CLASSES)), 'num_class == len(CLASSES)'
    num_class = len(CLASSES)

    GT_num_class = np.zeros(num_class, dtype=int)
    RES_num_class = np.zeros(num_class, dtype=int)
    all_boxes = [ [[] for _ in xrange(total_image)] for _ in xrange(num_class) ]

    for index in range(total_image):
        for cls in range(num_class):
            R = [obj for obj in gts[index] if obj['cls'] == cls]
            GT_num_class[cls] += len(R)
            RES_num_class[cls] += np.sum(results[index][:,0]==cls)
            inds = np.where(results[index][:,0]==cls)[0]
            all_boxes[cls][index] = results[index][inds , 1:6] # box confidence
        if index % 2000 == 0:
            print('Count {} / {} images').format(index,total_image)

    for index in range(num_class):
        print '{} has {} Ground Truth, {} Predicted Results'.format(CLASSES[index],GT_num_class[index],RES_num_class[index])

    AP = np.zeros(num_class)

    for cls in range(1,num_class,1): # Ingore __background__
        rec, prec, AP[cls] = cal_ap(all_boxes[cls], gts, cls)
        print 'AP for {} = {}'.format(CLASSES[cls],AP[cls])

    print 'Mean AP = {}'.format(np.mean(AP[1:]))
