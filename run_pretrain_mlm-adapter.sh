# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Pre-train lin-adapter

task=mlm
GPU='0,1,2,3'
CUDA_VISIBLE_DEVICES=$GPU python mlm-adapter.py  \
--model_type roberta \
--model_name=roberta-large  \
--data_dir=./data/ner_data/mlm_data  \
--output_dir mlm_ouptut \
--restore 'yes' \
--do_train  \
--do_eval   \
--evaluate_during_training True \
--task_name=$task     \
--comment 'mlm-adapter' \
--per_gpu_train_batch_size=32   \
--per_gpu_eval_batch_size=8   \
--num_train_epochs 2 \
--max_seq_length 64 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--warmup_steps=500 \
--save_steps 1000 \
--adapter_size 768 \
--adapter_list "0,11,22" \
--adapter_skip_layers 0 \
--meta_adapter_model=""