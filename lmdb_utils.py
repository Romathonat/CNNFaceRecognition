import lmdb
import glob,os
from random import randint
from subprocess import call

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

def generate_posneg_faces(number_faces_to_generate):
    #we read from posneg.txt, then we randomly select among the faces
    #64770: number of positive images
    line_to_extract = [randint(0, 64770) for x in range(number_faces_to_generate)]

    in_memory_posneg = []

    with open('posneg.txt', 'r') as f:
        for line in f:
            in_memory_posneg.append(line)

    extracted_lines = [in_memory_posneg[i] for i in line_to_extract]

    with open('posneg_iteration.txt', 'w+') as f:
        f.write(''.join(extracted_lines))

def generate_lmdb_from_non_faces(path_non_faces):
    local_posneg = "textures_posneg.txt"
    output = "textures_lmdb"

    with open("./{}".format(local_posneg), 'w+') as f:
        os.chdir("/.")
        for file in glob.glob("*.pgm"):
            f.write('{} 0\n'.format(file.name))

    call(["convert_imageset", "--suffle", "--gray", path_non_faces, local_posneg, output])

    #we can delete this file now
    os.remove('./{}'.format(output))

#1 we train the network with 20000 F
#2 we make him work on textures images (so non-faces), and we isolate the ones that are > THR (detected as faces)
#3 we generate lmdb file from those isolated images
#4 we generate new F and NF for next iterations: merge NF and generate_posneg_faces for faces
#5 goto 1
