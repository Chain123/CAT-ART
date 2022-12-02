#!/bin/bash

/data/miniconda3/bin/pip install sklearn

input=$*
IFS=" " read -ra ADDR <<< "$input"
domain=${ADDR[0]}
run=${ADDR[1]}
rec_loss=${ADDR[2]}
#echo "${domain}"
#echo "${run}"
method="base4"
#echo "$domain"
#echo "$run"
#echo "$rec_loss"
/data/miniconda3/bin/python main_cross_pre.py --num_run "${run}" --domain "${domain}" --epoch 50 --n_worker 8 --cross "${method}" --batch_size 8129 --train True --fix_user True --knowledge "d_in"
/data/miniconda3/bin/python main_cross_pre.py --num_run "${run}" --domain "${domain}" --epoch 50 --n_worker 8 --cross "${method}" --batch_size 8129 --train True --fix_user True --knowledge "d_sp" --encoder_loss "${rec_loss}"
/data/miniconda3/bin/python main_cross_pre.py --num_run "${run}" --domain "${domain}" --epoch 50 --n_worker 8 --cross "${method}" --batch_size 8129 --train True --fix_user True --knowledge "both" --encoder_loss "${rec_loss}"
/data/miniconda3/bin/python main_cross_pre.py --num_run "${run}" --domain "${domain}" --epoch 50 --n_worker 8 --cross "${method}" --batch_size 8129 --train True  --knowledge "d_in"
/data/miniconda3/bin/python main_cross_pre.py --num_run "${run}" --domain "${domain}" --epoch 50 --n_worker 8 --cross "${method}" --batch_size 8129 --train True  --knowledge "d_sp" --encoder_loss "${rec_loss}"
/data/miniconda3/bin/python main_cross_pre.py --num_run "${run}" --domain "${domain}" --epoch 50 --n_worker 8 --cross "${method}" --batch_size 8129 --train True  --knowledge "both" --encoder_loss "${rec_loss}"

#for i in 1 2 3
#do
#method="base4"
##echo "$method"
##/data/miniconda3/bin/python main_cross_pre.py --num_run "${run}" --domain "${domain}" --epoch 50 --n_worker 6 --cross "${method}" --batch_size 44096 --train True --fix_user True
##/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True --fix_user True
##/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True --fix_user True --fix_item True
##/data/miniconda3/bin/python main_cross.py --num_run "${run}" --domain "${domain}" --epoch 12 --n_worker 6 --cross "${method}" --batch_size 8192 --bar_dis True --train True --fix_item True
#done
