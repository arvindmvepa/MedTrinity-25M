#!/bin/bash
checkpoint=./checkpoints/llava-llama-med-8b-stage2-finetune


#python llava/eval/run_med_datasets_eval_batch.py --num-chunks  1 --device 7 --model-name $checkpoint \
#    --question-file /local2/amvepa91/MedTrinity-25M/data/vqa_rad/data.json \
#    --image-folder /local2/amvepa91/MedTrinity-25M/data/vqa_rad/images \
#    --answers-file vqa_preds/vqa_rad_modeltest_answer_file_$current_datetime.jsonl && \

python llava/eval/run_eval_nocandi.py \
    --gt /local2/amvepa91/MedTrinity-25M/data/vqa_rad/data.json \
    --pred vqa_preds/vqa_rad_modeltest_answer_file_$current_datetime.jsonl

