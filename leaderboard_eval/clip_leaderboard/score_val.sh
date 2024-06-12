#!/bin/bash

# Original
python ../score_comparison.py val_iccv_set_predictions/comparison.npy val_comparison/val_comparison_answer_key.json --instance_ids val_comparison/val_comparison_instance_ids.json



for i in {0..22}; 
do echo $i; 
	python ../score_retrieval.py val_iccv_set_predictions/retrieval_$i\.npy val_retrieval/val_retrieval_$i\_answer_key.json --instance_ids val_retrieval/val_retrieval_$i\_instance_ids.json; 

done;

#python ../score_localization.py val_ori_set_predictions/localization.npy val_localization/val_localization_answer_key.json --instance_ids val_localization/val_localization_instance_ids.json

# Repet
python ../score_comparison.py val_iccv_set_predictions/comparison.npy val_comparison/val_comparison_answer_key.json --instance_ids val_comparison/val_comparison_instance_ids.json


for i in {0..22}; 
do echo $i; 
	python ../score_retrieval.py val_set_predictions/retrieval_$i\.npy val_retrieval/val_retrieval_$i\_answer_key.json --instance_ids val_retrieval/val_retrieval_$i\_instance_ids.json; 

done;

#python ../score_localization.py val_set_predictions/localization.npy val_localization/val_localization_answer_key.json --instance_ids val_localization/val_localization_instance_ids.json
