#!/bin/bash

set -e
set -x

date=$1
ckpt=$2
task=$3
dataset=$4
model_type=$5
gpu=$6
gpunum=$7
start_point=$8
num_train_epochs=$9
masking_prob=${10}

max_steps=-1
num_task_inst=-1
num_train_task_epochs=2

checkpoint=$ckpt
output_dir="./test_logs/logs_$date-1-$dataset"

if [ $model_type = "bert" ]; then
    bsz=6
elif [ $model_type = "distilbert" ]; then
    bsz=8
fi

let valid_bsz=4*$bsz

masking="neural"
mask_dir=0
seed=42

export CUDA_VISIBLE_DEVICES=$gpu

for ((mask_dir=$start_point; mask_dir <= 1000; mask_dir+=50))
do
    python -m torch.distributed.launch --nproc_per_node=$gpunum \
    --master_port=9977 \
     main.py --test --checkpoint=$checkpoint \
        --output_dir=$output_dir \
        --max_steps=$max_steps \
        --per_gpu_train_batch_size=$bsz \
        --per_gpu_eval_batch_size=$valid_bsz \
        --model_type=$model_type \
        --masking=$masking \
        --mute \
        --num_train_epochs=$num_train_epochs \
        --num_train_task_epochs=$num_train_task_epochs \
        --masking_prob=$masking_prob \
        --truncate_task_dataset=$num_task_inst \
        --log_file="$dataset-test-$date.txt" \
        --mask_dir=mask/$mask_dir \
        --seed=$seed
done

