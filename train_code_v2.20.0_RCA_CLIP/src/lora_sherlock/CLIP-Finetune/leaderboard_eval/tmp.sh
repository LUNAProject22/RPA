#!/bin/bash

rm -rf val_mm_predictions
mkdir val_mm_predictions



python predict_leaderboard.py \
	   --instances ./sherlock/val_comparison/val_comparison_instances.json \
	   --output_npy val_mm_predictions/comparison.npy \
	    --load_model_from None \
	   --clip_model ViT-B-16 \
	   --Adapt_TxtEncoder False \
	   --Adapt_VisEncoder False \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained  openai \
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 1 \
	   --workers_dataloader 8 \
	   --task comparison


for split in {0..22};
do
	python predict_leaderboard.py \
	   --instances ./sherlock/val_retrieval/val_retrieval_$split\_instances.json \
	   --output_npy val_mm_predictions/retrieval_$split\.npy \
	    --load_model_from None \
	   --clip_model ViT-B-16 \
	   --Adapt_TxtEncoder False \
	   --Adapt_VisEncoder False \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained  openai\
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 1 \
	   --workers_dataloader 8 \
	   --task retrieval
done


python predict_leaderboard.py \
		--instances   ./sherlock/val_localization/val_localization_instances.json \
	    --output val_mm_predictions/localization.npy \
	    --load_model_from None \
	   --clip_model ViT-B-16 \
	   --Adapt_TxtEncoder False \
	   --Adapt_VisEncoder False \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained openai \
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 1 \
	   --workers_dataloader 8 \
	   --task localization
