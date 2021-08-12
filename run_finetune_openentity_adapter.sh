# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

task=entity_type
python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=data/ner_data/et_data \
    --output_dir=./et_output_no_pretraining  \
    --comment 'combine-adapter-trf' \
    --max_seq_length=64  \
    --per_gpu_eval_batch_size=4   \
    --per_gpu_train_batch_size=4   \
    --learning_rate=1e-4 \
    --gradient_accumulation_steps=1 \
    --max_steps=18000  \
    --model_name=roberta-large  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=120 \
    --restore 'yes' \
    --save_steps=10000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel="./trex_output/trex_maxlen-64_batch-64_lr-5e-05_warmup-1200_epoch-5_fac-adapter/checkpoint-7000/pytorch_model.bin" \
    --meta_lin_adaptermodel="./mlm_output/mlm_maxlen-64_epoch-2_batch-32_lr-2e-05_warmup-500_mlm-adapter/pytorch_model.bin"