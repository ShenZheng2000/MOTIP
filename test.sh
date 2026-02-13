ACC_CONFIG=/home/shenzhen/.cache/huggingface/accelerate/default_config8gpus.yaml
ACC_CONFIG_DEBUG=/home/shenzhen/.cache/huggingface/accelerate/default_config1gpu.yaml
DATA_ROOT=/ssd0/shenzhen/Datasets/tracking/
INF_DATA=DanceTrack

ulimit -n 65535

# # # # evaluate + val (8 gpu)
accelerate launch \
    --config_file $ACC_CONFIG \
    submit_and_evaluate.py \
    --data-root $DATA_ROOT \
    --inference-mode evaluate \
    --config-path ./configs/r50_deformable_detr_motip_dancetrack_warp_test_alpha05.yaml \
    --inference-model ./checkpoints/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth \
    --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack_warp_test_alpha05/debug_test_pretrained \
    --inference-dataset $INF_DATA \
    --inference-split val