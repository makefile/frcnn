import numpy as np
import sys, os, argparse

"""
Grount Truth File Format:
# id
image_name
number of boxes
label x1 y1 x2 y2 difficult
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

    images = []
    ground_truth = []
    results = []

    total = 0
    while True:
        gt_line, ans_line = gt_file.readline(), ans_file.readline()
        if gt_line == '':
            assert (gt_line == ans_line), 'GT File Done But Results File unfinished, {}'.format(ans_line)
            break
        gt_line, ans_line = gt_line.strip('\n').split(' '), ans_line.strip('\n').split(' ')
        assert (len(gt_line) == 2 and gt_line[0] == '#'), 'First Line of GT File Must Be \'# ID\', not {}'.format(
            gt_line)
        assert (len(ans_line) == 2 and ans_line[0] == '#'), 'First Line of Result File Must Be \'# ID\''
        assert (ans_line[1] == gt_line[1]), 'ID og GT and Result Must Be Same'
        gt_line, ans_line = gt_file.readline().strip('\n'), ans_file.readline().strip('\n')
        assert (gt_line == ans_line), 'Image Does not Euqal : {} and {}'.format(gt_line, ans_line)
        images.append(gt_line)

        gt_line, ans_line = gt_file.readline().strip('\n'), ans_file.readline().strip('\n')
        num_gt, num_ans = int(gt_line), int(ans_line)

        # Get Ground Truth Boxes
        gt_current_box = []#np.zeros((num_gt, 5))
        for index in range(num_gt):
            gt_line = gt_file.readline().strip('\n').split(' ')
            for jj in range(gt_line.count('')):
                gt_line.remove('')
            assert (len(gt_line) == 6), 'Ground Truth : label x1 y1 x2 y2 diff, not {}'.format(gt_line)
            diff = int(gt_line[5])
            if diff == 0:
                gt_current_box.append( np.array([float(gt_line[0]), float(gt_line[1]),
                    float(gt_line[2]), float(gt_line[3]), float(gt_line[4])]) )

        num_gt = len(gt_current_box)
        gt_current_box_ = np.zeros((num_gt, 5))
        for index in range(num_gt):
            gt_current_box_[index, :] = gt_current_box[index]

        ground_truth.append(gt_current_box_)

        # Get Results Boxes
        res_current_box = np.zeros((num_ans, 6))
        for index in range(num_ans):
            ans_line = ans_file.readline().strip('\n').split(' ')
            for jj in range(ans_line.count('')):
                ans_line.remove('')
            assert (len(ans_line) == 6), 'Result : label x1 y1 x2 y2 confidence, not {}'.format(gt_line)
            res_current_box[index, :] = np.array([float(ans_line[0]), float(ans_line[1]), float(ans_line[2]),
                                                  float(ans_line[3]), float(ans_line[4]), float(ans_line[5])])
        results.append(res_current_box)

        total += 1
        if total % 1000 == 0:
            print('Process {} images').format(total)
    print('Total {} images boxes load done').format(total)
    gt_file.close()
    ans_file.close()

    return images, ground_truth, results


if __name__ == '__main__':
    args = parse_args()
    images, gts, results = prepare_data(args)

    print('Prepare Data Done')
    ovthresh = args.overlap

    total_image = len(images)

    total = float(0)
    TP = float(0)
    for index in range(total_image):
        score = results[index][:, -1]
        order = score.ravel().argsort()[::-1]
        boxes = results[index][order, 1:5]
        BBGT = gts[index][:, 1:]
        det = np.zeros([BBGT.shape[0]])
        for obj in range(boxes.shape[0]):
            bb = boxes[obj, :]
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
                if det[jmax] == 0:
                    det[jmax] = 1
                    TP += 1
        total += BBGT.shape[0]
        if index % 1000 == 0:
            print 'Process {}/{} images '.format(index+1, total_image)

    print 'Recall %.4f , %5d / %5d' % (TP / total, TP, total)
