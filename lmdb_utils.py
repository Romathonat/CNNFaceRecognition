import lmdb

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

merge_lmdb('./train_lmdb', './test_lmdb', './test')
