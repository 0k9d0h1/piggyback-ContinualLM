#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o posttrain_procy_qa-%j.out
#SBATCH --gres gpu:2

export HF_DATASETS_CACHE='/home/0k9d0h1/piggyback/dataset_cache'
export TRANSFORMERS_CACHE='/home/0k9d0h1/piggyback/model_cache'

max_samples=640000

seed=(2021 111 222 333 444 555 666 777 888 999)

for round in 0;
do
  for idrandom in 0;
  do
  for pt_task in 0
    do
      for ft_task in $(seq 0 ${pt_task});
        do
          CUDA_VISIBLE_DEVICES=0 python finetune.py \
          --max_seq_length 164 \
          --pt_task ${pt_task} \
          --ft_task ${ft_task} \
          --idrandom ${idrandom} \
          --ntasks 6 \
          --max_samples ${max_samples} \
          --seed ${seed[$round]} \
        --baseline 'piggyback'
      done
    done
  done
done
