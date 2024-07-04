#!/bin/bash

rm -rf val_nips_predictions
mkdir val_nips_predictions



python predict_leaderboard.py \
	   --instances ./sherlock/val_comparison/val_comparison_instances.json \
	   --output_npy val_nips_predictions/comparison.npy \
	   --load_model_from ../../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-B-16-448x224-LR0.0002-B1600-Pamp_bf16-openai-E10-DCEmbTIP-2023_08_22-03_16_47/checkpoints/epoch_9.pt \
	   --load_model_from_2 ../../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR3e-05-B200-Pamp_bf16-openai-E10-DLossAdaTIP-2023_08_18-06_25_14/checkpoints/epoch_10.pt \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --clip_model ViT-B-16-448x224 \
	   --clip_model_2 ViT-L-14-672x336 \
	   --pretrained  openai \
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 20 \
	   --workers_dataloader 8 \
	   --task comparison


for split in {0..22};
do
	python predict_leaderboard.py \
	   --instances ./sherlock/val_retrieval/val_retrieval_$split\_instances.json \
	   --output_npy val_nips_predictions/retrieval_$split\.npy \
	   --load_model_from ../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-B-16-448x224-LR0.0002-B1600-Pamp_bf16-openai-E10-DCEmbTIP-2023_08_22-03_16_47/checkpoints/epoch_9.pt \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --clip_model ViT-B-16-448x224 \
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
	   --load_model_from ../logs/Gx2-RPA-V220-Mix-TxtVisAdapter-MAA-0.25-ViT-B-16-448x224-LR0.0002-B1600-Pamp_bf16-openai-E10-DCEmbTIP-2023_08_22-03_16_47/checkpoints/epoch_9.pt \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 1001 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained openai \
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --clip_model ViT-B-16-448x224 \
	   --batch_size 1 \
	   --workers_dataloader 8 \
	   --task localization
