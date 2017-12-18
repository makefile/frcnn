import os
import os.path as osp

if  __name__ == '__main__':
    ImageSet = '../LOC/LOC_Split/tv16.loc.testing.data.txt'
    Save_Name = '../dataset/test.list'
    if not os.path.isfile(ImageSet):
        print 'Ground File({}) does not exist'.format(ImageSet)
        sys.exit(1)
    file = open(ImageSet, 'r')
    
    otfile = open(Save_Name, 'w')
    total_img = 0
    total_shot = 0
    Pre_Compute_Img = 298065
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip('\n').split(' ')
        assert(len(line) == 2)
        line = line[1]
        assert(line[0:4] == 'shot')
        line = line[4:]
        line = line.split('_')
        assert(len(line) == 2)
        shot_dir = osp.join('..','LOC','filtered',line[0],line[1])
        files = os.listdir(shot_dir)
        jpeglist = []
        for f in files:
            if f[-4:] == 'jpeg':
                jpeglist.append(f)
            else:
                print '{}/{} Unsupport Suffix : {}'.format(line[0], line[1], f)

        otfile.write('# {}\t{}/{}   {}\n'.format(total_shot, line[0], line[1], len(jpeglist)))

        for f in jpeglist:
            otfile.write('{}\t'.format(f))
            total_img = total_img + 1

        if len(jpeglist) > 0:
            otfile.write('\n')

        total_shot = total_shot + 1
        if total_img > Pre_Compute_Img/2:
            print '{}====={}'.format(shot_dir, total_shot)
            Pre_Compute_Img = 1e10

    print 'Total Img : {}'.format(total_img)
    file.close()
    otfile.close()
