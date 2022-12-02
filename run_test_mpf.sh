#!/bin/bash

# 12 on /data/ceph 
# 11 and 13 on /cfs/cfs-a769xugn
dir1="/data/ceph/seqrec/UMMD/www/baselines"
dir2="/cfs/cfs-a769xugn/xxxxx/UMMD/baselines"

for domain in 0 1 2 3 4
do
echo "$domain"
# 12
/data/miniconda3/bin/python main_mpf.py --batch_size 32000 --num_run 12 --domain "${domain}" --epoch 400 --n_worker 5 --result_dir "${dir1}"
# 11 and 13
/data/miniconda3/bin/python main_mpf.py --batch_size 32000 --num_run 11 --domain "${domain}" --epoch 400 --n_worker 5 --result_dir "${dir2}"
/data/miniconda3/bin/python main_mpf.py --batch_size 32000 --num_run 13 --domain "${domain}" --epoch 400 --n_worker 5 --result_dir "${dir2}"
done

/data/miniconda3/bin/python main_mpf.py --batch_size 32000 --num_run 13 --domain 4 --epoch 400 --n_worker 5 --result_dir "${dir1}"
