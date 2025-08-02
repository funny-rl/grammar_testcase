#!/bin/bash

set -x

cd ../../../

export task_name="grammar"
export test_file_name="test"

python ./grammar_infer.py\
    data.batch_size=32\
    trainer.nnodes=1\
    trainer.n_gpus_per_node=4\
    model.load_param=true\
    model.load_param_path="./models/grammar_rl.pt"\
    data.task_name="$task_name"\
    data.n_samples=1\
    data.output_path="./model_output/grammar_rl@1.jsonl"\
    data.path="./data/test/parquet/${test_file_name}.parquet"\
    rollout.temperature=0.0\
    rollout.top_p=1.0 \
    rollout.n=1\
    rollout.prompt_length=2200\
    rollout.response_length=500\
