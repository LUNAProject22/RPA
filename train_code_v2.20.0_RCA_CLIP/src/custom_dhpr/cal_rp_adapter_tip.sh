#!/usr/bin/env python
import os


#	--model ViT-B-16-448x224 \
#	--batch-size 1600  \
#	--val-batch-size 1600 \
#	--lr 2e-4 \

#	--model ViT-L-14-672x336 \
cmd = "torchrun --nproc_per_node 2 --master_port 9527 -m main_rp_adapter_sherlock \
	--train-data sherlock/sherlock_train_v1_1.json \
	--val-data sherlock/sherlock_val_with_split_idxs_v1_1.json \
	--Adapt_TxtEncoder True \
	--Adapt_VisEncoder True \
	--adapter_rate 0.03125 \
	--region_prompt 1001 \
	--AdaType 0 \
	--model ViT-B-16-448x224 \
	--batch-size 200  \
	--val-batch-size 1 \
	--lr 2.5e-5 \
	--accum-freq 1 \
	--pretrained openai \
	--precision amp_bf16 \
	--warmup 5  \
	--epochs 10 \
	--workers 8 \
	--report-to wandb \
	--wandb-notes yes \
	--gather-with-grad \
	--local-loss \
	--name DLossAdaTIP \
	--seed 42 \
	--evaluate"

os.system(cmd)
