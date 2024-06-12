#!/bin/bash

rm -rf val_nips_predictions_2
mkdir val_nips_predictions_2



python predict_leaderboard.py \
	   --instances ./sherlock/val_comparison/val_comparison_instances.json \
	   --output_npy val_nips_predictions_2/comparison.npy \
	   --load_model_from ../logs/Gx2-RPA-V220-R-CTX-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E10-SC-2023_09_13-04_23_47/checkpoints/epoch_10.pt \
	   --clip_model ViT-L-14-672x336 \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 101 \
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
	   --output_npy val_nips_predictions_2/retrieval_$split\.npy \
	   --load_model_from ../logs/Gx2-RPA-V220-R-CTX-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E10-SC-2023_09_13-04_23_47/checkpoints/epoch_10.pt \
	   --clip_model ViT-L-14-672x336 \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 101 \
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
	    --output val_nips_predictions_2/localization.npy \
	   --load_model_from ../logs/Gx2-RPA-V220-R-CTX-TxtVisAdapter-MAA-0.25-ViT-L-14-672x336-LR2e-05-B200-Pamp_bf16-openai-E10-SC-2023_09_13-04_23_47/checkpoints/epoch_10.pt \
	   --clip_model ViT-L-14-672x336 \
	   --Adapt_TxtEncoder True \
	   --Adapt_VisEncoder True \
	   --region_prompt 101 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained openai \
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 1 \
	   --workers_dataloader 8 \
	   --task localization
