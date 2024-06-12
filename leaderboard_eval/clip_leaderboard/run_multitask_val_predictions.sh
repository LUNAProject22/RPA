#!/bin/bash

mkdir val_ori_set_predictions

python predict_clip_leaderboard.py \
       ./val_comparison/val_comparison_instances.json \
       clip_multitask/ORI=RN50x64~batch=64~warmup=1000~lr=1e-05~valloss=0.0000~randomclueinfhighlightbbox~widescreen_STEP=25200.pt \
       val_ori_set_predictions/comparison.npy \
       --vcr_dir images/ \
       --vg_dir images/ \
       --clip_model RN50x64 \
       --batch_size 25 \
       --hide_true_bbox 8 \
       --workers_dataloader 8 \
       --task comparison


for split in {0..22};
do
    python predict_clip_leaderboard.py \
	   ./val_retrieval/val_retrieval_$split\_instances.json \
           clip_multitask/ORI=RN50x64~batch=64~warmup=1000~lr=1e-05~valloss=0.0000~randomclueinfhighlightbbox~widescreen_STEP=25200.pt \
	   val_ori_set_predictions/retrieval_$split\.npy \
	   --vcr_dir images/ \
	   --vg_dir images/ \
	   --clip_model RN50x64 \
       	   --batch_size 25 \
	   --hide_true_bbox 8 \
	   --workers_dataloader 8 \
	   --task retrieval
done

