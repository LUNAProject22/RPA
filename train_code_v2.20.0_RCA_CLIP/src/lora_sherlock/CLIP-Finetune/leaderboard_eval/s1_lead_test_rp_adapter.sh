#!/bin/bash

rm -rf test_set_predictions
mkdir test_set_predictions

python predict_leaderboard.py \
		--instances   ./sherlock/test_localization_public/test_localization_instances.json \
	    --output test_set_predictions/localization.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-CPT--ViT-B-16-672x336-LR1e-05-B800-Pamp_bf16-openai-E10-SC2Full-2024_06_14-11_29_51/checkpoints/epoch_10.pt \
           --clip_model ViT-B-16-672x336 \
           --Adapt_TxtEncoder False \
           --Adapt_VisEncoder False \
           --region_prompt 2 \
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
	   --output_npy test_set_predictions/comparison.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-CPT--ViT-B-16-672x336-LR1e-05-B800-Pamp_bf16-openai-E10-SC2Full-2024_06_14-11_29_51/checkpoints/epoch_10.pt \
           --clip_model ViT-B-16-672x336 \
           --Adapt_TxtEncoder False \
           --Adapt_VisEncoder False \
           --region_prompt 2 \
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
	   --output_npy test_set_predictions/retrieval_$split\.npy \
	    --load_model_from ../logs/Gx2-RPA-V220-CPT--ViT-B-16-672x336-LR1e-05-B800-Pamp_bf16-openai-E10-SC2Full-2024_06_14-11_29_51/checkpoints/epoch_10.pt \
           --clip_model ViT-B-16-672x336 \
           --Adapt_TxtEncoder False \
           --Adapt_VisEncoder False \
           --region_prompt 2 \
	   --adapter_rate 0.25 \
	   --AdaType 0 \
	   --pretrained  openai\
	   --vcr_dir sherlock/images/ \
	   --vg_dir sherlock/images/ \
	   --batch_size 20 \
	   --workers_dataloader 8 \
	   --task retrieval
done


