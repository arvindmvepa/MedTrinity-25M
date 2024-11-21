#!/bin/bash
checkpoint=./LLaVA-Meta-Llama-3-8B-Instruct-FT-S2
projector=./llava-llama-med-8b-stage2-finetune-vqa_rad_orift


python llava/eval/run_med_datasets_eval_batch.py --num-chunks  1 --device 2 --image-folder "NA" --model-name $checkpoint \
    --projector-name $projector \
    --question-file /local2/amvepa91/MedTrinity-25M/PMC_VQA_num_samples100_prompt_typeq1_test.jsonl \
    --answers-file PMC_VQA_preds/PMC_VQA_modeltest_answer_file_$current_datetime.jsonl

