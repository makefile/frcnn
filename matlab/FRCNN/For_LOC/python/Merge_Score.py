import numpy as np
import sys, os, argparse

def Load_SCORE(file_name):
    file = open(file_name, 'r')
    shot_dir = []
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip('\n').split('\t')
        assert len(line) == 4 and line[0] == '#' and int(line[1]) == len(shot_dir)
        shot = {}
        shot['name'] = line[2]
        num = int(line[3])
        IMAGE = []
        for i in range(num):
            images = {}
            line = file.readline().strip('\n').split('\t')
            assert len(line) == 3 and line[0] == '&'
            images['jpeg'] = line[1]
            box_num = int(line[2])
            #cls = np.zeros(box_num, dtype=np.int32)
            scores = np.zeros(box_num)
            for j in range(box_num):
                line = file.readline().strip('\n').split(' ')
                assert len(line) == 1
              #  cls[j] = int(line[0])
                scores[j] = float(line[0])

            #images['cls'] = cls
            images['score'] = scores
            IMAGE.append(images)
        shot['images'] = IMAGE
        shot_dir.append(shot)
    file.close()
    print 'Load Score File : {}, Total : {}'.format(file_name, len(shot_dir))
    return shot_dir

def Load_ALL(file_name):
    file = open(file_name, 'r')
    shot_dir = []
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip('\n').split(' ')
        assert len(line) == 4 and line[0] == '#' and int(line[1]) == len(shot_dir)
        shot = {}
        shot['name'] = line[2]
        num = int(line[3])
        IMAGE = []
        for i in range(num):
            images = {}
            line = file.readline().strip('\n').split(' ')
            assert len(line) == 3 and line[0] == '&'
            images['jpeg'] = line[1]
            box_num = int(line[2])
            cls = np.zeros(box_num, dtype=np.int32)
            boxes = np.zeros((box_num, 5))
            for j in range(box_num):
                line = file.readline().strip('\n').split(' ')
                assert len(line) == 6
                cls[j] = int(line[0])
                boxes[j,:] = np.array([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5])])

            images['cls'] = cls
            images['boxes'] = boxes
            IMAGE.append(images)
        shot['images'] = IMAGE
        shot_dir.append(shot)
    file.close()
    print 'Load All File : {}, Total'.format(file_name, len(shot_dir))
    return shot_dir

def MERGE(base, scores):
    shot_merge = []
    for score in scores:
        assert len(base) == len(score)
    num_merge = scores
    for index in range(len(base)):
        base_shot = base[index]
        shot = {}
        shot['name'] = base_shot['name']
        IMAGE = []
        for score in scores:
            current_shot = score[index]
            assert base_shot['name'] == current_shot['name']
            assert len(base_shot['images']) == len(current_shot['images'])
        for j in range(len(base_shot['images'])):
            image = {}
            image['jpeg'] = base_shot['images'][j]['jpeg']
            image['cls'] = base_shot['images'][j]['cls']
            image['boxes'] = base_shot['images'][j]['boxes']
            for score in scores:
                base_image = base_shot['images'][j]
                curr_image = score[index]['images'][j]
                assert base_image['cls'].shape[0] == curr_image['score'].shape[0]
                assert base_image['cls'].shape[0] == image['boxes'].shape[0]
                assert curr_image['score'].shape[0] == base_image['cls'].shape[0]
                assert base_image['jpeg'] == curr_image['jpeg']
                for k in range(base_image['cls'].shape[0]):
                    image['boxes'][k,-1] = image['boxes'][k,-1] + curr_image['score'][k]
            IMAGE.append(image)
        shot['images'] = IMAGE
        shot_merge.append(shot)
    print 'MERGE : {} '.format(len(shot_merge))
    return shot_merge
                    
if __name__ == '__main__':
    BASE_SCORE = '../LOC/merge_loc/TWO_1_9958_res152.frcnn';
    ADDO_SCORE = ['../two/res152_merge_other/TWO_1_9958_googlenet_v1.score',
                    '../two/res152_merge_other/TWO_1_9958_res101.score', 
                    '../two/res152_merge_other/TWO_1_9958_vgg19.score']
    OUT_PATH = '../TWO_1_9958_res152.merge'
    """
    BASE_SCORE = '../LOC/merge_loc/EIGHT_1_9958_res152.frcnn';
    ADDO_SCORE = ['../eight/res152_merge_other/EIGHT_1_9958_googlenet_v1.score',
                    '../eight/res152_merge_other/EIGHT_1_9958_res101.score', 
                    '../eight/res152_merge_other/EIGHT_1_9958_vgg19.score']
    OUT_PATH = '../EIGHT_1_9958_res152.merge'
    """
    all_shot = Load_ALL(BASE_SCORE)
    score_shot = []
    for file in ADDO_SCORE:
        temp = Load_SCORE(file)
        score_shot.append(temp)
    
    shot_merge = MERGE(all_shot, score_shot)
    file = open(OUT_PATH, 'w')
    total = 0
    for shot in shot_merge:
        IMAGES = shot['images']
        shot_name = shot['name']
        file.write('# {} {} {}\n'.format(total, shot_name, len(IMAGES)))
        total = total + 1
        for image in IMAGES:
            image_name = image['jpeg']
            cls = image['cls']
            boxes = image['boxes']
            assert cls.shape[0] == boxes.shape[0]
            file.write('& {} {}\n'.format(image_name, cls.shape[0]))
            for k in range(cls.shape[0]):
                file.write('{0} {1:.1f} {2:.1f} {3:.1f} {4:.1f} {5:.5f}\n'.format( int(cls[k]), 
                   boxes[k,0], boxes[k,1], boxes[k,2], boxes[k,3], float(boxes[k,4])/4))


    file.close()

