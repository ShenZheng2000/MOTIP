# NOTE: use this to install torch
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# and then do this to install newer version of gcc
# conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 ninja -y

ACC_CONFIG=/home/shenzhen/.cache/huggingface/accelerate/default_config8gpus.yaml
ACC_CONFIG_DEBUG=/home/shenzhen/.cache/huggingface/accelerate/default_config1gpu.yaml
DATA_ROOT=/ssd0/shenzhen/Datasets/tracking/
INF_DATA=DanceTrack

ulimit -n 65535

# evaluate + val (8 gpus)
# accelerate launch \
#     --config_file $ACC_CONFIG \
#     submit_and_evaluate.py \
#     --data-root $DATA_ROOT \
#     --inference-mode evaluate \
#     --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml \
#     --inference-model ./checkpoints/r50_deformable_detr_motip_dancetrack/r50_deformable_detr_motip_dancetrack.pth \
#     --outputs-dir ./checkpoints/r50_deformable_detr_motip_dancetrack/ \
#     --inference-dataset $INF_DATA \
#     --inference-split val

# # # evaluate + val (1 gpu for debug): $ACC_CONFIG_DEBUG

# # training (8 gpus)
# accelerate launch \
#     --config_file $ACC_CONFIG \
#     train.py \
#     --data-root $DATA_ROOT \
#     --exp-name r50_deformable_detr_motip_dancetrack \
#     --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml


# # training (8 gpus): train & test using gt bbox to warp!
accelerate launch \
    --config_file $ACC_CONFIG \
    train.py \
    --data-root $DATA_ROOT \
    --exp-name r50_deformable_detr_motip_dancetrack_warp_test \
    --config-path ./configs/r50_deformable_detr_motip_dancetrack_warp_test.yaml

# TODO: write better train and test scripts (otherwise messy path for outputs/checkpoints)