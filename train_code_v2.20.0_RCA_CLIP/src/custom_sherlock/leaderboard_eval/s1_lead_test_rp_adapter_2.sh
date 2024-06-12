#!/bin/bash

rm -rf test_set_predictions_2
mkdir test_set_predictions_2


python predict_leaderboard.py \
		--instances   ./sherlock/test_localization_public/test_localization_instances.json \
	    --output test_set_predictions_2/localization.npy \
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


python predict_leaderboard.py \
	   --instances ./sherlock/test_comparison_public/test_comparison_instances.json \
	   --output_npy test_set_predictions_2/comparison.npy \
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
	   --instances ./sherlock/test_retrieval_public/test_retrieval_$split\_instances.json \
	   --output_npy test_set_predictions_2/retrieval_$split\.npy \
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


