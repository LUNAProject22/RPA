#!/usr/bin/env python
import os

	#--val-data refexp/finetune_rpa/refcoco/refcoco_val_mattn.json  \

cmd = "torchrun --nproc_per_node 2 --master_port 9527 -m main_rp_adapter_refcoco \
	--train-data refexp/finetune_rpa/refcoco/refcoco_train_mattn.json \
	--val-data refexp/finetune/refcoco/refcoco_testA_yolov8.json  \
	--val-data-2 refexp/finetune/refcoco/refcoco_testB_yolov8.json  \
	--val-data-3 refexp/finetune/refcoco/refcoco_val_yolov8.json  \
	--use_negative_box \
	--model ViT-B-16-448x224 \
	--batch-size 800  \
	--val-batch-size 1 \
	--lr 1.0e-4 \
	--Adapt_TxtEncoder True \
	--Adapt_VisEncoder True \
	--adapter_rate 0.25 \
	--region_prompt 101 \
	--AdaType 0 \
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
	--name yv8 \
	--seed 42"

os.system(cmd)


