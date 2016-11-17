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

import pdb

from is_face import is_face

def clean_folder():
    try:
        shutil.rmtree('./train_lmdb_iterations/faces')
    except:
        pass
    try:
        shutil.rmtree('./train_lmdb_iterations/non_faces')
    except:
        pass
    try:
        shutil.rmtree('./textures_lmdb')
    except:
        pass
    os.remove('train_lmdb_iterations/lock.mdb')
    os.remove('train_lmdb_iterations/data.mdb')

def merge_lmdb(path_lmdb1, path_lmdb2, out_path):
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
    '''we read from posneg.txt, then we randomly select among the faces'''
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

    call('convert_imageset --shuffle --gray {}/ {} {}'.format(path_images, local_posneg, output), shell=True)

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

    print('convert_imageset --shuffle --gray {}/ {} {}'.format(path_images, local_posneg, output))
    call('convert_imageset --shuffle --gray {}/ {} {}'.format(path_images, local_posneg, output), shell=True)

    print('convet done')
    #we can delete this file now
    os.remove(local_posneg)

    return output

def generate_train_lmdb(number_faces, path_lmdb_new_non_faces, count_mistakes = 0):
    #we remove those folder if they are present because of convert imageset

    path_lmdb_faces = generate_lmdb_from_random_pics(number_faces, \
        './train_images', './train_lmdb_iterations/faces' , True)
    path_lmdb_non_faces = generate_lmdb_from_random_pics(number_faces-count_mistakes, \
        './train_images', './train_lmdb_iterations/non_faces' , False)

    merge_lmdb(path_lmdb_non_faces, path_lmdb_new_non_faces, './train_lmdb_iterations')
    merge_lmdb('./train_lmdb_iterations', path_lmdb_faces, './train_lmdb_iterations')

def generate_train_lmdb_without_new_non_faces(number_faces):
    #we remove those folder if they are present because of convert imageset
    path_lmdb_faces = generate_lmdb_from_random_pics(number_faces, \
        './train_images', './train_lmdb_iterations/faces' ,True)
    path_lmdb_non_faces = generate_lmdb_from_random_pics(number_faces, \
        './train_images', './train_lmdb_iterations/non_faces', False)

    merge_lmdb(path_lmdb_non_faces, path_lmdb_faces, './train_lmdb_iterations')


number_faces = 20000
THR = 0.9
clean_folder()
generate_train_lmdb_without_new_non_faces(number_faces)

for count_iteration in range(6):
    print('Iteration number :{}'.format(count_iteration))

    #1 we train the network with 20000 Faces
    caffe.set_mode_cpu()
    solver = caffe.get_solver('/datas/facenet_solver.prototxt')
    solver.solve()

    #2 we make it work on textures images (so non-faces), and we isolate the ones that are > THR (detected as faces)
    #os.chdir("./")

    #we remove files in the folder containing images for the next lmdb generation
    files = glob.glob('/datas/train_lmdb_iterations_images/*')
    for f in files:
        os.remove(f)

    #we add images
    count_mistakes = 0
    for f in os.listdir('./textures'):
        if f.endswith(".pgm"):
            image = caffe.io.load_image('./textures/{}'.format(f), color=False)
            if is_face(image) > THR:
                shutil.copy('./textures/{}'.format(f), '/datas/train_lmdb_iterations_images')
                count_mistakes += 1

    #3 we generate lmdb file from those isolated images
    #those are textures
    path_lmdb_new_non_faces = generate_lmdb_from_images('/datas/train_lmdb_iterations_images')

    print("step 4")

    #4 we generate new F and NF for next iterations: merge NF and generate_lmdb_faces for faces
    clean_folder()
    generate_train_lmdb(number_faces, path_lmdb_new_non_faces, count_mistakes)

    print("iteration finished")

    THR -= 2
    number_faces += count_mistakes
    clean_folder()
