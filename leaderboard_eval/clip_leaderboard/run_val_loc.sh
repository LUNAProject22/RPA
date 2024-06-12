#!/bin/bash

mkdir val_ori_set_predictions

python predict_clip_leaderboard.py \
       ./val_localization/val_localization_instances.json \
       clip_multitask/ORI=RN50x64~batch=64~warmup=1000~lr=1e-05~valloss=0.0000~randomclueinfhighlightbbox~widescreen_STEP=25200.pt \
       val_ori_set_predictions/localization.npy \
       --vcr_dir images/ \
       --vg_dir images/ \
       --clip_model RN50x64 \
       --batch_size 300 \
       --hide_true_bbox 8 \
       --workers_dataloader 8 \
       --task localization


mkdir val_set_predictions


python predict_clip_leaderboard.py \
       ./val_localization/val_localization_instances.json \
       clip_multitask/model=RN50x64~batch=16~warmup=1000~lr=1e-05~valloss=0.4803~randomclueinfhighlightbbox~cluewithprefix~widescreen.pt \
       val_set_predictions/localization.npy \
       --vcr_dir images/ \
       --vg_dir images/ \
       --clip_model RN50x64 \
       --batch_size 300 \
       --hide_true_bbox 8 \
       --workers_dataloader 8 \
       --task localization

