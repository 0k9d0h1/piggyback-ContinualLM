#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o posttrain_procy_qa-%j.out
#SBATCH --gres gpu:2

export HF_DATASETS_CACHE='/home/eecomp_test/donghoon/dataset_cache'
export TRANSFORMERS_CACHE='/home/eecomp_test/donghoon/model_cache'
max_samples=640000

for idrandom in 0
do
  for pt_task in 0 1 2 3 4 5
  do
    python -m torch.distributed.launch --nproc_per_node 1 --use_env posttrain.py \
    --base_model_name_or_path "t5-base" \
    --per_device_train_batch_size 128 \
    --max_seq_length 164 \
    --max_samples ${max_samples} \
    --idrandom ${idrandom} \
    --ntasks 6 \
    --pt_task ${pt_task} \
    --baseline 'lora'
  done
done

