#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o posttrain_procy_qa-%j.out
#SBATCH --gres gpu:2

export HF_DATASETS_CACHE='/home/eecomp_test/donghoon/dataset_cache'
export TRANSFORMERS_CACHE='/home/eecomp_test/donghoon/model_cache'

max_samples=640000

seed=(2021 111 222 333 444 555 666 777 888 999)

for round in 0 1 2 3 4;
do
  for idrandom in 0;
  do
  for ft_task in 0 1 2 3 4 5;
    do
      CUDA_VISIBLE_DEVICES=0 python finetune.py \
      --base_model_name_or_path "t5-base" \
      --max_seq_length 164 \
      --pt_task ${ft_task} \
      --ft_task ${ft_task} \
      --idrandom ${idrandom} \
      --ntasks 6 \
      --max_samples ${max_samples} \
      --seed ${seed[$round]} \
      --baseline 'lora' \
      --finetune_type 'lora_piggyback'
    done
  done
done
