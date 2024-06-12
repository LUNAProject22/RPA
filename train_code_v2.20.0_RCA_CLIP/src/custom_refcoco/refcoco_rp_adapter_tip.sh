#!/usr/bin/env python
import os

#	--model ViT-B-16-448x224 \
#	--batch-size 800  \
#	--val-batch-size 1 \
#	--lr 1.0e-4 \



#	--train-data refexp/finetune/refcocog/refcocog_train_yolov8.json \
#	--val-data refexp/finetune/refcocog/refcocog_val_yolov8.json \
#	--val-data-2 refexp/finetune/refcocog/refcocog_test_yolov8.json \
#	--val-data-3 refexp/finetune/refcocog/refcocog_test_yolov8.json \

#	--train-data refexp/finetune/refcoco+/refcoco+_train_yolov8.json \
#	--val-data refexp/finetune/refcoco+/refcoco+_testA_yolov8.json  \
#	--val-data-2 refexp/finetune/refcoco+/refcoco+_testB_yolov8.json  \
#	--val-data-3 refexp/finetune/refcoco+/refcoco+_val_yolov8.json  \

cmd = "torchrun --nproc_per_node 2 --master_port 9527 -m main_rp_adapter_refcoco \
	--use_negative_box \
	--model ViT-L-14-672x336 \
	--train-data refexp/finetune/refcoco/refcoco_train_yolov8.json \
	--val-data refexp/finetune/refcoco/refcoco_testA_yolov8.json  \
	--val-data-2 refexp/finetune/refcoco/refcoco_testB_yolov8.json  \
	--val-data-3 refexp/finetune/refcoco/refcoco_val_yolov8.json  \
	--batch-size 100  \
	--val-batch-size 1 \
	--lr 2.0e-5 \
	--Adapt_TxtEncoder True \
	--Adapt_VisEncoder True \
	--adapter_rate 0.25 \
	--region_prompt 102 \
	--AdaType 0 \
	--accum-freq 1 \
	--pretrained openai \
	--precision amp_bf16 \
	--warmup 5  \
	--epochs 5 \
	--workers 8 \
	--report-to wandb \
	--wandb-notes yes \
	--gather-with-grad \
	--grad-checkpointing \
	--local-loss \
	--torchcompile \
	--name RFCC \
	--seed 42"

os.system(cmd)


