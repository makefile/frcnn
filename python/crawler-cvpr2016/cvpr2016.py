#!/usr/bin/env python
# coding=utf-8
import urllib
import urllib2
import re
import os
from time import time
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from os.path import isfile, join
from os import listdir
from bs4 import BeautifulSoup
import unicodedata
import shutil

def getHtml(url):
    ts = time()
    page = urllib.urlopen(url)
    html = page.read()
    print('Load html : {} , took {}s'.format(url, time()-ts))
    return html

def download_file(download_url, file_name):

    ts = time()
    response = urllib2.urlopen(download_url)
    file = open(file_name, 'w')
    file.write(response.read())
    file.close()

    print("Completed : {}".format(download_url))
    return '{} cost {}s'.format(file_name, time()-ts)

def find_in_array(file_name_list, search_str):
    flag = 0
    index = 0
    search_str = unicodedata.normalize('NFKD', search_str).encode('ascii','ignore')
    for i in range(0,len(file_name_list)):
        if(not file_name_list[i].lower().find(search_str.lower())==-1):
            flag+=1
            index =i
    if flag == 1:
        return index
    else:
        return False

def letters(input):
    valids = []
    for character in input:
        if character.isalpha() or character == '-' or character.isdigit():
            valids.append(character)
    return ''.join(valids)

def classify_all_papers(paper_dir, cls_dir):
    url = 'http://cvpr2016.thecvf.com/program/main_conference#O1-1B'
    files = [ f for f in listdir(paper_dir) if isfile(join(paper_dir,f)) ]
    html = getHtml(url)
    soup = BeautifulSoup(html, "lxml")
    program_list = soup.find_all("ul",class_="program-list")
    count = 0
    no_copy = 0
    check = []
    for i in xrange(0,len(program_list)):
        name = program_list[i].strong.text
        name_array = name.split(' ')
        for idx in xrange(len(name_array)):
            name_array[idx] = letters(name_array[idx])

        search_str = name_array[0]+'_'+name_array[1]+'_'+name_array[2]
        pre_h4 = program_list[i].find_previous("h4").text
        pre_h3 = program_list[i].find_previous("h3").text
        dir_path = join(cls_dir, pre_h3, pre_h4)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        index = find_in_array(files, search_str)
        # if index in check:
        #     print 'error'
        #     print(index)
        check.append(index)
        if index:
            shutil.copy(join(paper_dir, files[index]), join(dir_path,files[index]))
            count += 1
        else:
            flag = 0
            while (not index) and flag < 4:
                index = find_in_array(files,name_array[flag])
                flag += 1
            if index:
                shutil.copy(join(paper_dir, files[index]), join(dir_path,files[index]))
                count += 1
            else:
                #shutil.copy(join(paper_dir, files[index]), join(cls_dir, files[index]))
                no_copy += 1
                count += 1
                print name

def download_all_papers(save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    cv_root = 'http://www.cv-foundation.org/openaccess/'

    url = join(cv_root, 'CVPR2016.py')
    html = getHtml(url)

    ts = time()
    parttern = re.compile(r'\bcontent_cvpr_2016.*paper\.pdf\b')
    url_list = parttern.findall(html)
    
    print '\n>>>>>>>>>>>>>>>Total {} Papers>>>>>>>>>>>>>'.format(len(url_list))
    print '>>>>>>>>>>>>>>>Multiprocess with {} Pool>>>>>>>>>>>>>\n'.format(cpu_count())
    pool = Pool(processes=cpu_count())
    results = []
    for url in url_list:
        name = url.split('/')[-1]
        file_name = join(save_path, name)
        results.append( pool.apply_async(
            download_file, (join(cv_root, url), file_name)) )

    pool.close()
    pool.join()

    print '\n>>>>>>>>>>>>>>>Download {} Papers Done in {}s>>>>>>>>>>>>>>>'.format(len(url_list), time()-ts)
    for res in results:
        print res.get()


if __name__ == "__main__":
    save_name = 'cvpr2016'
    root_path = os.path.expanduser('~')
    paper_dir = join(root_path, save_name)
    cls_dir = join(paper_dir, 'class')

    download_all_papers(paper_dir)
    classify_all_papers(paper_dir, cls_dir)
