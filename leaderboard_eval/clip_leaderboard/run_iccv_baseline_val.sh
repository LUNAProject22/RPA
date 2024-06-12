#!/bin/bash

mkdir val_iccv_set_predictions

python predict_clip_leaderboard.py \
       ./val_comparison/val_comparison_instances.json \
	sherlock_pretrained_model/model=ViT-B16~batch=512~warmup=500~lr=1e-05~valloss=0.0000~highlightbbox~widescreen_STEP=1800.pt \
       val_iccv_set_predictions/comparison.npy \
       --vcr_dir images/ \
       --vg_dir images/ \
	--clip_model ViT-B/16 \
       --batch_size 1 \
       --hide_true_bbox 8 \
       --workers_dataloader 8 \
       --task comparison


for split in {0..22};
do
    python predict_clip_leaderboard.py \
	   ./val_retrieval/val_retrieval_$split\_instances.json \
	   sherlock_pretrained_model/model=ViT-B16~batch=512~warmup=500~lr=1e-05~valloss=0.0000~highlightbbox~widescreen_STEP=1800.pt \
	   val_iccv_set_predictions/retrieval_$split\.npy \
	   --vcr_dir images/ \
	   --vg_dir images/ \
	   --clip_model ViT-B/16 \
       	   --batch_size 1 \
	   --hide_true_bbox 8 \
	   --workers_dataloader 8 \
	   --task retrieval
done

python predict_clip_leaderboard.py \
       ./val_localization/val_localization_instances.json \
	sherlock_pretrained_model/model=ViT-B16~batch=512~warmup=500~lr=1e-05~valloss=0.0000~highlightbbox~widescreen_STEP=1800.pt \
       val_iccv_set_predictions/localization.npy \
       --vcr_dir images/ \
       --vg_dir images/ \
	--clip_model ViT-B/16 \
       --batch_size 1 \
       --hide_true_bbox 8 \
       --workers_dataloader 8 \
       --task localization
