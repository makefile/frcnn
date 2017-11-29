%{
ILSVRC_Root = '~/data/ILSVRC2015';
Save_Dir = '../../examples/FRCNN/dataset';
%}
function ILSVRC2015_VID_FRCNN(ILSVRC_Root, Save_Dir)
% Format : 
% # image_index
% img_path (relative path)
% num_roi
% label x1 y1 x2 y2 difficult
%ILSVRC_Root = '~/data/ILSVRC2015';
%Save_Dir = '../../examples/FRCNN/dataset';
clc; clearvars -except ILSVRC_Root Save_Dir;
Cpath = pwd;
cd(ILSVRC_Root);ILSVRC_Root=pwd;cd(Cpath);
cd(Save_Dir);Save_Dir=pwd;cd(Cpath);
devkit = fullfile(ILSVRC_Root, 'devkit', 'evaluation');
assert( ~isempty(devkit) && exist(devkit,'dir') );
assert( ~isempty(Save_Dir) && exist(Save_Dir,'dir') );
addpath( devkit );

ImagePath = fullfile(ILSVRC_Root, 'Data', 'VID', 'train');

train_txt = fopen(fullfile(Save_Dir, 'DET_train.txt'), 'w');
class = 30;

meta = load(fullfile(ILSVRC_Root, 'devkit', 'data', 'meta_det.mat'));
classes = {'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic cat', 'elephant', 'fox', 'giant panda', 'hamster',...
'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra'};
assert(length(classes) == 30);
assert(length(meta.synsets) == 200);
metname = {meta.synsets.WNID};
metcls  = cell(length(meta.synsets), 1);
temp_check = 0;
for index = 1:length(meta.synsets)
    metcls{index} = -1;
    for cls = 1:length(classes)
        if(strcmp(classes{cls}, meta.synsets(index).name) == 1)
            metcls{index} = cls;
            fprintf('%s : %s\n', classes{cls}, metname{index});
            temp_check = temp_check + 1;
            break;
        end
    end
end
assert(temp_check == 30, ['Got ', int2str(temp_check) , ' class']);

for cls = 1:class
    tic;
    text = fullfile(ILSVRC_Root, 'ImageSets', 'VID' , ['train_', int2str(cls), '.txt']);
    [folders, ~] = textread(text, '%s%d');
    for i = 1:length(folders)
        %assert( classes(i) == cls );
        Dir = fullfile(ILSVRC_Root, 'Annotations', 'VID', 'train', folders{i});
        xmls = dir(fullfile(Dir, '*xml'));
        assert(length(xmls) > 0);
        temp=fullfile(Dir, xmls(1).name); temp=VOCreadxml(temp); temp=temp.annotation;
        width = temp.size.width;  height = temp.size.height;
        folder = temp.folder;

        fprintf(train_txt, '# %s\n', fullfile(ImagePath, folder));
        fprintf(train_txt, '%d %s %s\n', length(xmls), height, width);

        bool_only = true;
        for index = 1:length(xmls)
            xml = fullfile(Dir, xmls(index).name);
            annotation = VOCreadxml( xml );
            annotation = annotation.annotation;

            assert( strcmp(width, annotation.size.width) == 1);
            assert( strcmp(height, annotation.size.height) == 1);

            %image = fullfile(ImagePath, annotation.folder, [annotation.filename, '.JPEG']);
            image = fullfile([annotation.filename, '.JPEG']);
            if isfield(annotation, 'object') == 0
                fprintf(train_txt, '%s 0\n', image);
                continue;
            end
            objects = annotation.object;
            %fprintf('%s %d\n', xml, length(objects));
            %assert( length(annotation.object.bndbox) == 1);

            fprintf(train_txt, '%s %d\n', image, length(objects));
            for obj = 1:length(objects)
                name = objects(obj).name;
                vidcls = GetLabel(metname, metcls, name);
                if (vidcls ~= cls)
                    bool_only = false;
                end
                assert(vidcls>=1 && vidcls<=30);
                fprintf(train_txt, '%3s %2d %4s %4s %4s %4s\n', objects(obj).trackid, vidcls, objects(obj).bndbox.xmin...
                    , objects(obj).bndbox.ymin, objects(obj).bndbox.xmax, objects(obj).bndbox.ymax);
            end
        end
        if bool_only == false
            fprintf('folder : %s : %s\n', folder, classes{vidcls});
        end

    end
    fprintf('cls %02d , cost %4.2f s\n', cls, toc);
end

fclose(train_txt);

end

function cls = GetLabel(metname, metcls, name) 
    assert(length(metname) == 200);
    assert(length(metcls) == 200);
    cls = -1;
    for index = 1:length(metname)
        if(strcmp(metname(index), name) == 1)
            cls = metcls{index};break;
        end
    end
    assert(cls > 0, ['Unknow type : ', name]);
end
