#!/bin/bash

rm -rf val_mm_predictions
mkdir val_mm_predictions



python predict_leaderboard.py \
	   --instances ./sherlock/val_comparison/val_comparison_instances.json \
	   --output_npy val_mm_predictions/comparison.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-B-16-448x224-LR0.0002-B1600-Pamp_bf16-openai-E10-DiV2-Lam-0.1-2024_06_13-14_49_44/checkpoints/epoch_10.pt \
	   --clip_model ViT-B-16-448x224 \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained  openai \
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 100 \
	   --workers_dataloader 8 \
	   --task comparison


for split in {0..22};
do
	python predict_leaderboard.py \
	   --instances ./sherlock/val_retrieval/val_retrieval_$split\_instances.json \
	   --output_npy val_mm_predictions/retrieval_$split\.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-B-16-448x224-LR0.0002-B1600-Pamp_bf16-openai-E10-DiV2-Lam-0.1-2024_06_13-14_49_44/checkpoints/epoch_10.pt \
	   --clip_model ViT-B-16-448x224 \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained  openai\
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 20 \
	   --workers_dataloader 8 \
	   --task retrieval
done


python predict_leaderboard.py \
		--instances   ./sherlock/val_localization/val_localization_instances.json \
	    --output val_mm_predictions/localization.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-B-16-448x224-LR0.0002-B1600-Pamp_bf16-openai-E10-DiV2-Lam-0.1-2024_06_13-14_49_44/checkpoints/epoch_10.pt \
	   --clip_model ViT-B-16-448x224 \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained openai \
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 1 \
	   --workers_dataloader 8 \
	   --task localization
