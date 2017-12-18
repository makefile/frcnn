import os
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import subprocess
import uuid


def Get_Class_Ind(Class_INT):

    concepts = []
    concepts.append(('Animal', [
        'n01443537', 'n01503061', 'n01639765', 'n01662784', 'n01674464', 'n01726692', 'n01770393',
        'n01784675', 'n01882714', 'n01910747', 'n01944390', 'n01990800', 'n02062744', 'n02076196',
        'n02084071', 'n02118333', 'n02129165', 'n02129604', 'n02131653', 'n02165456', 'n02206856',
        'n02219486', 'n02268443', 'n02274259', 'n02317335', 'n02324045', 'n02342885', 'n02346627',
        'n02355227', 'n02374451', 'n02391049', 'n02395003', 'n02398521', 'n02402425', 'n02411705',
        'n02419796', 'n02437136', 'n02444819', 'n02445715', 'n02454379', 'n02484322', 'n02503517',
        'n02509815', 'n02510455', 'Animal']))

    concepts.append(('Music', [
        'n02672831', 'n02787622', 'n02803934', 'n02804123', 'n03249569', 'n03372029', 'n03467517', 'n03800933',
        'n03838899', 'n03928116', 'n04141076', 'n04536866', 'Music']))

    #######
    concepts.append(('Bike', ['Bike', 'n02834778', 'n02835271', 'n03792782', 'n04126066']))
    concepts.append(('Baby', ['Baby', 'n10353016']))

    concepts.append(('Boy', ['Boy', 'n09871229', 'n09871867', 'n10078719']))
    concepts.append(('Fire', ['n03343560', 'Fire', 'FIre', 'n03346135',
        'n10091450', 'n10091564', 'n10091861', 'n14891255']))
    concepts.append(('Skier', ['Skier', 'n04228054']))
    concepts.append(('Running', ['Running']))
    concepts.append(('Dancing', ['Dancing']))
    concepts.append(('Sit', ['Sit']))

    if (Class_INT == 2):
        concepts = concepts[:2]
    elif (Class_INT == 5):
        concepts = concepts[2:7]
    elif (Class_INT == 8):
        concepts = concepts[2:10]
    elif (Class_INT == 10):
        concepts = concepts[:]
    else:
        assert False

    _wind_to_ind = {}
    _class_to_ind = {}
    _classes = ('__background__',)
    for synset in concepts:
        _classes = _classes + (synset[0],)
        class_ind = len(_classes) - 1
        _class_to_ind[ synset[0] ] = class_ind
        print ('%9s [Label]: %d' % (synset[0], class_ind))
        for wnid in synset[1]:
            _wind_to_ind[ wnid ] = class_ind
            #print ('--------> : {}'.format(wnid))

    return _wind_to_ind, _class_to_ind

def load_annotation(filename, _class_to_ind):

    #filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')

    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.int32)
    gt_classes = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) 
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        cls = _class_to_ind[obj.find('name').text.strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls


    return {'boxes' : boxes,
            'gt_classes': gt_classes}


if __name__ == '__main__':
    #Save_Name = './dataset/8.train_val'
    ImageSets = ['../LOC/LOC_Split/trecvid_val_8.txt', '../LOC/LOC_Split/trecvid_train_8.txt']
    ImageSets = ['../LOC/LOC_Split/trecvid_train_Animal_Music.txt', '../LOC/LOC_Split/trecvid_val_Animal_Music.txt']
    ImageSets = ['../LOC/LOC_Split/trecvid_5_manual_train.txt']
    ImageSets = ['../LOC/LOC_Split/trecvid_train_8.txt', '../LOC/LOC_Split/trecvid_val_8.txt', '../LOC/LOC_Split/trecvid_train_Animal_Music.txt', '../LOC/LOC_Split/trecvid_val_Animal_Music.txt']
    num_cls = 10
    Save_Name = '../dataset/{}.train_val'.format(num_cls)
    _wind_to_ind, _class_to_ind = Get_Class_Ind(num_cls)
    for ImageSet in ImageSets:
        if not os.path.isfile(ImageSet):
            print 'File({}) does not exist'.format(ImageSet)
            sys.exit(1)
        else:
            print 'Open File : {} '.format(ImageSet)

    print 'Save into : {} '.format(Save_Name)
    out_file = open(Save_Name, 'w')
    ids = 0
    count_cls = np.zeros((num_cls+1), dtype=np.int32)
    assert count_cls.shape[0]-1 == len(_class_to_ind)
    for ImageSet in ImageSets:
        file = open(ImageSet, 'r')
        while True:
            line = file.readline()
            if line == '':
                break
            line = line.strip('\n')
            xml_path = '../LOC/BBOX/{}.xml'.format(line)
            rec = load_annotation(xml_path, _wind_to_ind)

            out_file.write('# {}\n'.format(ids))
            ids = ids + 1
            out_file.write('{}.JPEG\n'.format(line))
            boxes = rec['boxes']
            gt_classes = rec['gt_classes']
            assert boxes.shape[0] == gt_classes.shape[0]
            out_file.write('{}\n'.format(boxes.shape[0]))

            for j in range(boxes.shape[0]):
                out_file.write('{} {} {} {} {} 0\n'.format(int(gt_classes[j]),int(boxes[j,0]),int(boxes[j,1]),int(boxes[j,2]),int(boxes[j,3])))
                count_cls[ int(gt_classes[j]) ] = count_cls[ int(gt_classes[j]) ] + 1
                    

            if ids % 2000 == 0:
                print 'print {} image with recs into {}'.format(ids, Save_Name)

        file.close()

    for i in range(count_cls.shape[0]):
        print ('%2d th : %4d' % (i, count_cls[i]))
        i = i + 1
    out_file.close()


