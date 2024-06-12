#!/usr/bin/env python
import os

## lr = B / 800 * 2.0e-5

#	--model ViT-L-14-672x336 \
#	--batch-size 400  \
#	--val-batch-size 600 \
#	--lr 4.0e-6 \

cmd = "torchrun --nproc_per_node 2 --master_port 9527 -m main_rp_adapter_sherlock \
	--train-data sherlock/sherlock_train_v1_1.json \
	--val-data sherlock/sherlock_val_with_split_idxs_v1_1.json \
	--model ViT-L-14-672x336 \
	--batch-size 200  \
	--val-batch-size 200 \
	--lr 2.0e-6 \
	--Adapt_TxtEncoder False \
	--Adapt_VisEncoder False \
	--adapter_rate 0.25 \
	--AdaType 0 \
	--full_tune \
	--region_prompt 2 \
	--accum-freq 1 \
	--pretrained openai \
	--precision amp_bf16 \
	--warmup 5  \
	--epochs 10 \
	--workers 8 \
	--report-to wandb \
	--wandb-notes yes \
	--gather-with-grad \
	--grad-checkpointing \
	--local-loss \
	--torchcompile \
	--name SC2Full \
	--seed 42"

os.system(cmd)
