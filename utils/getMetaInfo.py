"""
generate a meta file which records mapping information from a basecalled folder

"""
import h5py
import os,sys
from .myFast5 import *
import subprocess
import time
from io import StringIO

from multiprocessing import Pool
from multiprocessing import Manager

# data_dir = '/media/quanc/E/Data/MinION/R9/ecoli_er2925.pcr.r9.timp.061716.fast5/pass'
# data_dir = '/media/quanc/E/Data/RNA/Hopkins_Run2_20170928_DirectRNA_albacore/workspace/pass'
data_dir = '/media/quanc/E/Data/RNA/Hopkins_Run1_20170928_DirectRNA_albacore/workspace/pass'
# data_dir = '/media/quanc/E/Data/RNA/Hopkins_Run5_20171003_DirectRNA_albacore/workspace/pass'
# data_dir = '/media/quanc/E/Data/chiron_train/test/data_10000_12000/pcr_MSssI'
out_dir = '/media/quanc/E/Data/chiron_train/train_raw/data_50000_100000/pcr'
# meta_file = '/media/quanc/E/Data/chiron_train/train_raw/meta_pcr_50000_100000.txt'
meta_file = '/media/quanc/E/Data/RNA/res/meta_run1.txt'

start_pos = 50000
end_pos = 100000
# start_pos = 4619600
# end_pos = 4620000


def copy_file(fast5_tmp, path, q):
    res_str = ''
    try:
        mf5 = h5py.File(os.path.join(path, fast5_tmp), 'r')
    except:
        mf5 = None

    if not mf5 == None:

        try:
            mapped_chrom, mapped_start, mapped_strand = ReadMapInfoInRef(mf5)
            nanoraw_events = ReadNanoraw_events(mf5)

            res_str = os.path.join(path, fast5_tmp) + '\t' \
                      + mapped_chrom + '\t' + mapped_strand + '\t' + \
                      str(mapped_start) + '\t' + str(mapped_start + len(nanoraw_events)) + \
                      '\n'

        except:
            print("cannot open " + fast5_tmp)

    q.put(res_str)


def run():

    pool = Pool(8)
    q = Manager().Queue()

    f5list = os.listdir(data_dir)

    num_sum = 1

    for f5_ind in range(len(f5list)):

        f5 = f5list[f5_ind]
        fast5list = os.listdir(os.path.join(data_dir,f5))
        num_sum += len(fast5list)

    num = 0
    for f5_ind in range(len(f5list)):

        f5 = f5list[f5_ind]
        fast5list = os.listdir(os.path.join(data_dir,f5))
        path = os.path.join(data_dir,f5)

        for fast5_ind in range(len(fast5list)):
            fast5_tmp = fast5list[fast5_ind]
            pool.apply_async(copy_file, args=(fast5_tmp, path, q))

    start = time.time()
    with open(meta_file, 'w') as file:
        while num<num_sum:
            strtmp = q.get()
            if strtmp != '':
                file.write(strtmp)

            num += 1
            copyrate = num * 1.0 / num_sum
            end = time.time()
            sys.stdout.write("\r process: %.2f%%, num: %d/%d, Elapsed Time: %0.2fs\n"
                             % (copyrate * 100, num, num_sum, float(end-start)))
            sys.stdout.flush()


if __name__ == '__main__':
    run()