python /data5/langgao/GroundingModels/cellposse/cellpose_infer_batch.py \
    --image_path /data2/langgao/Data/cytoimagenet/y6545/ \
    --model_type cyto3 \
    --use_gpu True \
    --num_gpus 2 \
    --batch_size 128 \
    --diameter 30.0 \
    --cellprob_threshold 0 \
    --mask_path /data5/langgao/GroundingModels/cellposse/mask \
    --wmask_path /data5/langgao/GroundingModels/cellposse/wmask \
    --mechine_name 2u2 \
    --mask_json /data5/langgao/GroundingModels/cellposse/mask_message.json