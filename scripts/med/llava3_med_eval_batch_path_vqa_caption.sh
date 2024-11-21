#!/bin/bash
checkpoint=./LLaVA-Meta-Llama-3-8B-Instruct-FT-S2
projector=./llava-llama-med-8b-stage2-finetune-pathvqa_orift


python llava/eval/run_med_datasets_eval_batch.py --num-chunks  1 --device 3 --image-folder "/local2/amvepa91/MedTrinity-25M/pvqa/images/test" --model-name $checkpoint \
    --projector-name $projector \
    --question-file /local2/amvepa91/MedTrinity-25M/Path_VQA_prompt_typeq1.6_test_v1.json \
    --answers-file Path_VQA_preds/Path_VQA_caption_prompt_typeq1.6_test_v1_modeltest_answer_file_$current_datetime.jsonl

