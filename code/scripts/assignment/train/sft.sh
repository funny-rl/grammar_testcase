#!/bin/bash

set -x

nproc_per_node=4
project_name="Assignment_SFT"
experiment_name="deepseek_distilled_qwen_14B"
save_path="./checkpoints/sft/$project_name/$experiment_name"
shift 2

cd ../../../

export task_name="assignment"
export train_file_name="train"
export val_file_name="valid"
export incorr_code_dir="./reward_model/data/solutions/incorrect_solutions"
data_path="./data/${task_name}_sft"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node ./sft.py\
    data.task_name=$task_name\
    data.train_batch_size=8\
    data.micro_batch_size_per_gpu=1\
    data.path="${data_path}"\
    data.train_files="${data_path}/parquet/${train_file_name}.parquet"\
    data.val_files="${data_path}/parquet/${val_file_name}.parquet"\
    data.max_length=6500\
    trainer.default_local_dir=$save_path\
    trainer.project_name=$project_name\
    trainer.experiment_name=$experiment_name\
    trainer.logger='["console","wandb"]' $@