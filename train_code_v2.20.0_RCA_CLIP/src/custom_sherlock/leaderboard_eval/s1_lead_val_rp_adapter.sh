#!/bin/bash

rm -rf val_nips_predictions
mkdir val_nips_predictions



python predict_leaderboard.py \
	   --instances ./sherlock/val_comparison/val_comparison_instances.json \
	   --output_npy val_nips_predictions/comparison.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E10-SC-2023_09_11-14_42_36/checkpoints/epoch_10.pt \
	   --clip_model ViT-L-14-672x336 \
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
	   --output_npy val_nips_predictions/retrieval_$split\.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E10-SC-2023_09_11-14_42_36/checkpoints/epoch_10.pt \
	   --clip_model ViT-L-14-672x336 \
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
	    --output val_nips_predictions/localization.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E10-SC-2023_09_11-14_42_36/checkpoints/epoch_10.pt \
	   --clip_model ViT-L-14-672x336 \
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
