#!/bin/bash

set -e
set -x

date=$1

max_steps=-1
task_max_steps=-1
num_pretrain_inst=200
outer_epoch=1000
num_task_inst=1000

task=$2
dataset=$3
model_type=$4
gpu=$5
masking="neural"

if [ $task = "qa" ]; then
    block_size=510
    num_task_epochs=1
elif [ $task = "glue" ]; then
    block_size=254
    num_task_epochs=5
fi

if [ $model_type = "bert" ]; then
    if [ $task = "qa" ]; then
        bsz=4
    elif [ $task = "glue" ]; then
        bsz=8
    fi
elif [ $model_type = "distilbert" ]; then
    if [ $task = "qa" ]; then
        bsz=8
    elif [ $task = "glue" ]; then
        bsz=16
    fi
fi

if [ $task = "qa" ]; then
    replay_start=2000
    masking_prob=0.05
elif [ $task = "glue" ]; then
    replay_start=500
    masking_prob=0.05
fi

let valid_bsz=4*$bsz


output_dir="./results/$dataset/$model_type/$date-$masking"

export CUDA_VISIBLE_DEVICES=$gpu

python main.py \
        --output_dir=$output_dir \
        --model_type=$model_type \
        --do_lower_case \
        --max_steps=$max_steps \
        --task_max_steps=$task_max_steps \
        --masking=$masking \
        --outer_epoch=$outer_epoch \
        --num_train_task_epochs=$num_task_epochs \
        --task=$task \
        --truncate_task_dataset=$num_task_inst \
        --truncate_pretraining_dataset=$num_pretrain_inst \
        --learning_rate=2e-5 \
        --dataset=$dataset \
        --block_size=$block_size \
        --mask_learning_rate=1e-4 \
        --matching_sample \
        --fix_task 10000 \
        --num_train_epochs 3 \
        --masking_prob $masking_prob \
        --mute \
        --actor_critic \
        --replay_batch_size 64 \
        --per_gpu_train_batch_size=$bsz \
        --per_gpu_eval_batch_size=$valid_bsz \
        --memory_capacity=50000 --replay_start=$replay_start \
        --self_play learning \
        --continual \
        --importance_sampling --replay_step=10 \
        --ppo_policy