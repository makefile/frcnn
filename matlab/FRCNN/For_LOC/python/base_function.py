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

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Calculate Recall For Rpn Detector Results')
    parser.add_argument('--gt', dest='gt_file',
                        help='Ground Truth File',
                        default=None, type=str)
    parser.add_argument('--answer', dest='ans_file',
                        help='Results Files, Split By \';\'',
                        default=None, type=str)
    parser.add_argument('--overlap', dest='overlap',
                        help='overlap value',
                        default=0.5, type=float)
    args = parser.parse_args()

    if args.ans_file is None or args.gt_file is None:
        parser.print_help()
        sys.exit(1)
    args.ans_file = args.ans_file.split(';')

    for files in args.ans_file:
        if not os.path.isfile(files):
            print 'Answer File : {} is not file'.format(files)
            parser.print_help()
            sys.exit(1)
        else:
            print 'Answer File : {}'.format(files)

    if not os.path.isfile(args.gt_file):
        print 'Ground Truth File : {}'.format(args.gt_file)
        sys.exit(1)

    print 'Called with args:\n{}'.format(args)
    return args


# image results = prepare_data()
def prepare_data(file_):
    if not os.path.isfile(file_):
        print 'Results File({}) does not exist'.format(ans_file)
        sys.exit(1)
    file_ = open(file_)
    # Handle

    results = []
    images = []

    total = 0
    boxes_num = 0
    while True:
        ans_line = file_.readline()
        if ans_line == '':
            break
        ans_line = ans_line.strip('\n').split(' ')
        assert (len(ans_line) == 2 and ans_line[0] == '#'), 'First Line of Result File Must Be \'# ID\''
        assert (int(ans_line[1]) == total), 'ID og GT and Result Must Be Same'
        ans_line = file_.readline().strip('\n')
        images.append(ans_line)

        ans_line = file_.readline().strip('\n')
        num_ans = int(ans_line)

        boxes_num += num_ans
        # Get Ground Truth Boxes
        # gt_current_box = np.zeros((num_gt, 5))
        objects = []
        for index in range(num_ans):
            ans_line = file_.readline().strip('\n').split(' ')
            for jj in range(ans_line.count('')):
                ans_line.remove('')
            assert (len(ans_line) == 6), 'Ground Truth : label x1 y1 x2 y2 diff, not {}'.format(gt_line)
            #assert (len(ans_line) == 6), 'Result File : label x1 y1 x2 y2 confidence, not {}'.format(ans_line)
            obj_struct = {}
            obj_struct['bbox'] = np.array([float(ans_line[1]),float(ans_line[2]), float(ans_line[3]), float(ans_line[4])])
            obj_struct['cls']  = int(ans_line[0])
            obj_struct['additional'] = ans_line[5]
            objects.append(obj_struct)

        results.append(objects)

        total += 1
        #if total % 1000 == 0:
        #    print('Process {} images').format(total)

    print('Total {} images and {} boxes load done').format(total, boxes_num)
    file_.close()
    return images, results

def check_data(images_gt, images_res, res_gt, res_res):
    assert len(images_gt) == len(images_res)
    assert len(res_gt) == len(res_res)
    assert len(images_gt) == len(res_gt)
    for index in range(len(images_gt)):
        assert images_gt[index] == images_res[index]
    
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

def cal_ap(RES, gts, ccls, ovthresh):

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
        for obj in RES[index]:
            image_ids.append(index)
            confidence.append(obj['confidence'])
            BB.append(obj['bbox'])

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

def convert_allbox(results_res, num_class):
    total_image = len(results_res)
    RES_num_class = np.zeros(num_class, dtype=int)
    all_boxes = [ [[] for _ in xrange(total_image)] for _ in xrange(num_class) ]
    for index in range(total_image):
        for cls in range(num_class):
            R = [obj for obj in results_res[index] if obj['cls'] == cls]
            RES_num_class[cls] += len(R)
            all_boxes[cls][index] = R
    return RES_num_class, all_boxes

def change_addition2attr(results, attr, type):
    for objs in results:
        for obj in objs:
            obj[attr] = type(obj['additional'])

def Get_CLASSES(cls_num):
    if cls_num == 2:
        CLASSES = ('__background__', 'Animal', 'Music')
    elif cls_num == 7:
        CLASSES = ('__background__', 'Animal', 'Music', 'Bike', 'Baby', 'Boy', 'Fire', 'Skier')
    elif cls_num == 10:
        CLASSES = ('__background__', 'Animal', 'Music', 'Bike', 'Baby', 'Boy', 'Fire', 'Skier', 'Running', 'Dancing', 'Sit')
    else:
        assert False, 'Un support class number : {}'.format(cls_num)

    assert len(CLASSES) == cls_num + 1
    return CLASSES

