#!/usr/bin/env python
import os



#	--model ViT-L-14-672x336 \
#	--batch-size 200  \
#	--val-batch-size 200 \
#	--lr 2.0e-5 \

#	--batch-size 1600  \
#	--val-batch-size 1600 \
#	--lr 2e-4 \

cmd = "torchrun --nproc_per_node 2 --master_port 9527 -m main_rp_adapter_dhpr \
	--train-data DHPR_data/annotation/anno_train_NEAT.json \
	--val-data   DHPR_data/annotation/anno_val_GEN_NEAT.json \
	--vcr_dir DHPR_data/dhpr_image/ \
	--model ViT-L-14-672x336 \
	--batch-size 200  \
	--val-batch-size 200 \
	--lr 2.0e-5 \
	--Adapt_TxtEncoder True \
	--Adapt_VisEncoder True \
	--adapter_rate 0.25 \
	--region_prompt 1001 \
	--AdaType 0 \
	--accum-freq 1 \
	--pretrained openai \
	--precision amp_bf16 \
	--warmup 5  \
	--epochs 25 \
	--workers 8 \
	--report-to wandb \
	--wandb-notes yes \
	--gather-with-grad \
	--grad-checkpointing \
	--local-loss \
	--torchcompile \
	--name CM \
	--seed 42"

os.system(cmd)


