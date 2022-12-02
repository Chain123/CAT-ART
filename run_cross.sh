#!/bin/bash

/data/miniconda3/bin/pip install sklearn

input=$*
IFS=" " read -ra ADDR <<< "$input"
domain=${ADDR[0]}
run=${ADDR[1]}
#echo "${domain}"
#echo "${run}"
method='base3'
/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True
/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True --fix_user True

#for i in 1 2 3
#do
#method="base${i}"
##echo "$method"
#/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True
#/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True --fix_user True
##/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True --fix_user True --fix_item True
##/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True --fix_item True
#done
