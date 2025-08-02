#!/bin/bash

set -x
cd ../../../

temperature=1.0
top_p=1.0
batch_size_per_gpu=1

project_name="Assignment_RL"
experiment_name="deepseek_distilled_qwen_14B"
save_path="./checkpoints/rl/$project_name/$experiment_name"

export task_name="assignment"
export train_file_name="train"
export val_file_name="valid"
export incorr_code_dir="./reward_model/data/solutions/incorrect_solutions"

data_path="./data/${task_name}_rl"
dataset=$task_name

python rl.py \
    custom_reward_function.path="./reward_model/${dataset}_RM.py"\
    trainer.project_name=$project_name\
    trainer.experiment_name="$experiment_name"\
    data.path="${data_path}"\
    data.train_files="${data_path}/parquet/${train_file_name}.parquet"\
    data.val_files="${data_path}/parquet/${val_file_name}.parquet"\
    data.max_prompt_length=5700\
    data.max_response_length=1500\
    data.task_name=$task_name\
    data.train_batch_size=4\
    data.val_batch_size=32\
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_size_per_gpu}\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${batch_size_per_gpu}\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${batch_size_per_gpu}\
    actor_rollout_ref.rollout.temperature=${temperature}\
    actor_rollout_ref.rollout.top_p=${top_p}\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5\
    actor_rollout_ref.rollout.n=5\
    actor_rollout_ref.rollout.max_num_batched_tokens=8192\
    actor_rollout_ref.rollout.max_num_seqs=1024\
    trainer.save_freq=1\
    trainer.test_freq=10\
    trainer.total_epochs=2\
    actor_rollout_ref.model.load_param=True\
    actor_rollout_ref.model.load_param_path="./models/assignment_sft.pt"\
    actor_rollout_ref.rollout.tensor_model_parallel_size=4\
    trainer.n_gpus_per_node=4\
    trainer.logger=['console','wandb']\
    trainer.default_local_dir=$save_path\