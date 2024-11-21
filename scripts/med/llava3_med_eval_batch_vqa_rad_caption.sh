#!/bin/bash
checkpoint=./LLaVA-Meta-Llama-3-8B-Instruct-FT-S2
projector=./llava-llama-med-8b-stage2-finetune-vqa_rad_orift


python llava/eval/run_med_datasets_eval_batch.py --num-chunks  1 --device 2 --image-folder "/local2/shared_data/VQA-RAD/images" --model-name $checkpoint \
    --projector-name $projector \
    --question-file /local2/amvepa91/MedTrinity-25M/VQA_RAD_v1_prompt_typeq1.1_test.json \
    --answers-file VQA_RAD_preds/VQA_RAD_caption_prompt_typeq1.1_modeltest_answer_file_$current_datetime.jsonl

