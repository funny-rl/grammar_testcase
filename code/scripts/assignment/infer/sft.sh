#!/bin/bash

set -x

cd ../../../

export task_name="assignment"
export test_file_name="test"
export incorr_code_dir="./reward_model/data/solutions/incorrect_solutions"

python ./assignment_infer.py\
    trainer.nnodes=1\
    trainer.n_gpus_per_node=4\
    model.load_param=true\
    model.load_param_path="./models/assignment_sft.pt"\
    data.task_name="$task_name"\
    data.n_samples=1\
    data.batch_size=32\
    data.prompt_key="prompt"\
    data.output_path="./model_output/assignment_sft@1.jsonl"\
    data.path="./data/test/parquet/${test_file_name}.parquet"\
    rollout.temperature=0.0\
    rollout.top_p=1.0 \
    rollout.n=1\
    rollout.prompt_length=5700\
    rollout.response_length=1500\