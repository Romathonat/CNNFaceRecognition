import lmdb
from random import randint

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

generate_posneg_faces(10000)
