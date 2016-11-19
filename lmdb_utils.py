import glob,os
os.environ['GLOG_minloglevel'] = '2'

import lmdb
import caffe

from random import randint
from subprocess import call
import caffe
import re
import fileinput
import shutil

def clean_folder():
    '''Clean for convert_imageset'''
    try:
        shutil.rmtree('./train_lmdb_iterations/faces')
    except:
        pass
    try:
        shutil.rmtree('./train_lmdb_iterations/non_faces')
    except:
        pass
    try:
        shutil.rmtree('./train_lmdb_iterations/non_faces2')
    except:
        pass
    try:
        shutil.rmtree('./textures_lmdb')
    except:
        pass

    try:
        os.remove('train_lmdb_iterations/lock.mdb')
        os.remove('train_lmdb_iterations/data.mdb')
    except:
        pass

def merge_lmdb(path_lmdb1, path_lmdb2, out_path):
    '''Merge two lmdb in out_path'''
    txn1 = lmdb.open(path_lmdb1).begin(write=True)
    txn2 = lmdb.open(path_lmdb2).begin(write=True)

    database1 = txn1.cursor()
    database2 = txn2.cursor()

    with lmdb.open(out_path, map_size=2**39).begin(write=True) as txn3:
        for (key, value) in database1:
            txn3.put(key, value)

        for (key, value) in database2:
            txn3.put(key, value)

def generate_lmdb_from_random_pics(number_to_generate, path_images, output, among_faces):
    '''Select number_to_generate faces or non_faces, depending on among_faces, and \
    convert it to lmdb in output'''
    local_posneg = './posneg_iteration.txt'

    #64770: number of positive images
    if(among_faces):
        line_to_extract = [randint(0, 64770) for x in range(number_to_generate)]
    else:
        line_to_extract = [randint(64771, 91719) for x in range(number_to_generate)]

    in_memory_posneg = []

    with open('posneg.txt', 'r') as f:
        for line in f:
            in_memory_posneg.append(line)


    extracted_lines = [in_memory_posneg[i] for i in line_to_extract]

    with open(local_posneg, 'w+') as f:
        f.write(''.join(extracted_lines))

    call('convert_imageset --shuffle --gray {}/ {} {}'.format(path_images, \
        local_posneg, output), shell=True)

    os.remove(local_posneg)

    return output

def generate_lmdb_from_images(path_images):
    '''Generate lmdb from the images in path_images. Create the coresponding \
    posneg file'''
    local_posneg = "textures_posneg.txt"
    output = "textures_lmdb"

    with open("./{}".format(local_posneg), 'w+') as f:
        for image in os.listdir(path_images):
            if image.endswith(".pgm"):
                f.write('{} 0\n'.format(image))

    call('convert_imageset --shuffle --gray {}/ {} {}'.format(path_images, \
        local_posneg, output), shell=True)

    #we can delete this file now
    os.remove(local_posneg)

    return output

def generate_train_lmdb(number, path_lmdb_new_non_faces, count_mistakes = 0):
    '''Generate a lmdb of faces and non-faces, and merge it with the lmdb\
    from path_lmdb_new_non_faces'''

    path_lmdb_faces = generate_lmdb_from_random_pics(number, \
        './train_images', './train_lmdb_iterations/faces' , True)
    path_lmdb_non_faces = generate_lmdb_from_random_pics(number-count_mistakes, \
        './train_images', './train_lmdb_iterations/non_faces' , False)


    merge_lmdb(path_lmdb_non_faces, path_lmdb_new_non_faces, './train_lmdb_iterations/non_faces2')
    merge_lmdb('./train_lmdb_iterations/non_faces2', path_lmdb_faces, './train_lmdb_iterations')
    #error here

def generate_train_lmdb_without_new_non_faces(number_faces):
    '''Generate a lmdb of faces and non-faces'''
    path_lmdb_faces = generate_lmdb_from_random_pics(number_faces, \
        './train_images', './train_lmdb_iterations/faces' ,True)
    path_lmdb_non_faces = generate_lmdb_from_random_pics(number_faces, \
        './train_images', './train_lmdb_iterations/non_faces', False)

    merge_lmdb(path_lmdb_non_faces, path_lmdb_faces, './train_lmdb_iterations')

def change_number_iterations_training(number_iterations):
    '''Change the number of iterations in facenet_solver.prototxt'''
    with open('./facenet_solver.prototxt', 'r+w') as f:
        memory_file = f.readlines()
        text_to_write = []
        for i,line in enumerate(memory_file):
            text_to_write.append(re.sub('max_iter: [0-9]+', 'max_iter: {}'\
            .format(number_iterations), line))

        f.seek(0)
        f.write(''.join(text_to_write))
        f.truncate()
