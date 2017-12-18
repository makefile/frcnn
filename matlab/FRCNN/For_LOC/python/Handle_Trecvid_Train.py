import xml.etree.ElementTree as ET
import scipy.sparse
import scipy.io as sio
import cv2
import os
import random
#[6] Animal
#[13] Bicycling
#[16] Boy
#[38] Dancing
#[49] Explosion_fire
#[71] Instrumental_Musician ( localize both player and instrument)
#[100] Running
#[107] Sitting_Down
#[434] Skier 
#[163] Baby
#            Animal  Music    Bike    Baby   Boy     Fire    Skier   Running Dancing  Sit
NEG_CLASS = ['1006', '1071', '1013', '1163', '1016', '1049', '1434', '1100', '1038', '1107']
TOTAL_CLASS = 10
def generate_pos():
    tlist = '../dataset/10.train_val'
    ori_DIR = '/home/dongxuanyi/data/LOC/IMG/'
    pos_DIR = '/home/dongxuanyi/data/LOC/pos_clips_extract/'
    pos_file = open(tlist, 'r')
    total = 0
    IDX = 0
    while True:
        line = pos_file.readline()
        if line == '':
            break
        line = line.strip('\n').split(' ')
        assert len(line) == 2, '{}'.format(line)
        assert line[0] == '#' and int(line[1]) == IDX, '{}  {}'.format(line[0], line[1])
        IDX = IDX + 1
        pos_name = pos_file.readline().strip('\n')
        image_name = os.path.join(ori_DIR, pos_name);
        img = cv2.imread(image_name, cv2.CV_LOAD_IMAGE_COLOR)

        num = int(pos_file.readline().strip('\n'))
        for index in range(num):
            line = pos_file.readline().strip('\n').split(' ')
            label = int(line[0])
            x1, y1, x2, y2 = int(line[1]), int(line[2]), int(line[3]), int(line[4])
            crop_name = '{}{}__{}.JPEG'.format(pos_DIR, pos_name[:-5], index)
            crop_img = img[y1:y2, x1:x2, :]
            cv2.imwrite(crop_name, crop_img)
            print '{} {}'.format(crop_name, 2*label-1)
            
        total = total + num

    print 'Total {0:^5d} Pos Images'.format(total)
    
def generate_neg():
    assert len(NEG_CLASS) == TOTAL_CLASS
    _neg_to_ind = {}
    for index in range(TOTAL_CLASS):
        _neg_to_ind[ NEG_CLASS[index] ] = index + 1

    tlist = '/home/dongxuanyi/data/LOC/neg_sample.cid'
    neg_DIR = '/home/dongxuanyi/data/LOC/neg_clips_extract'
    if not os.path.isfile(tlist):
        print 'File({}) does not exist'.format(tlist)
        sys.exit(-1)
    else:
        print 'Open Neg List : {}'.format(tlist)
    neg_file = open(tlist, 'r')
    total = 0
    for line in neg_file:
        line = line.split(' ')
        assert len(line) == 2
        video_name = line[0].strip('\n').split(',')[0]
        neg_cls = line[1].strip('\n')
        video_name = video_name[:-4]
        neg_cls = _neg_to_ind[neg_cls]
        neg_dir = os.path.join(neg_DIR, video_name)
        image_files = os.listdir(neg_dir)
        jpg = []
        for image in image_files:
            if image[-4:] == '.jpg':
                jpg.append(image)
                print '{0:20s} {1:2d}'.format(os.path.join(neg_DIR, video_name, image), 2*neg_cls-2)
            else:
                print '{} Unsupport Suffix : {}'.format(video_name, image)

        total = total + len(jpg)
    neg_file.close()
    print 'Total {0:^5d} Neg Images'.format(total)

def get_cls_train_val():
    pos_cls = './dataset/pos_cls_data.txt'
    neg_cls = './dataset/neg_cls_data.txt'
    f = open(pos_cls, 'r')
    pos_lines = f.readlines()
    f.close()
    f = open(neg_cls, 'r')
    neg_lines = f.readlines()
    f.close()
    basename = './dataset/CLS'
    random.shuffle(pos_lines)
    random.shuffle(neg_lines)

    pos_num = len(pos_lines)
    neg_num = min(10*pos_num, len(neg_lines))
    lines = pos_lines[:pos_num]
    lines.extend(neg_lines[:neg_num])
    print 'POS : {},  NEG : {}'.format(pos_num, neg_num)
    random.shuffle(lines)

    total_lines = len(lines)
    val_num = int(total_lines * 0.2)
    #train_num = total_lines - val_num
    train_file = '{}.train'.format(basename)
    train_file = open(train_file, 'w')
    for line in lines[:val_num]:
        train_file.write('{}'.format(line))
    train_file.close()

    val_file = '{}.val'.format(basename)
    val_file = open(val_file, 'w')
    for line in lines[val_num:]:
        val_file.write('{}'.format(line))
    val_file.close()
    

if __name__ == '__main__':

    #generate_pos()
    #generate_neg()
    get_cls_train_val()
