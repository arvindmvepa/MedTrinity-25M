#!/bin/bash
checkpoint=/local2/amvepa91/MedTrinity-25M/MBZUAI/LLaVA-Meta-Llama-3-8B-Instruct-FT-S2
projector=/local2/amvepa91/MedTrinity-25M/MBZUAI/LLaVA-Meta-Llama-3-8B-Instruct-FT-S2


python llava/eval/run_med_datasets_eval_batch.py --num-chunks  1 --device 5 --model-name $checkpoint \
    --projector-name $projector \
    --question-file /local2/amvepa91/Slake1.0/test.json \
    --image-folder /local2/amvepa91/Slake1.0/imgs/ \
    --answers-file slake/slake_modeltest_answer_file_noimage_$current_datetime.jsonl

