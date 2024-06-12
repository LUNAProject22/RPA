#!/bin/bash

# Dist Rept
python ./score_comparison.py  val_nips_predictions_2/comparison.npy sherlock/val_comparison/val_comparison_answer_key.json --instance_ids sherlock/val_comparison/val_comparison_instance_ids.json


for i in {0..22}; 
do echo $i; 
	python ./score_retrieval.py val_nips_predictions_2/retrieval_$i\.npy sherlock/val_retrieval/val_retrieval_$i\_answer_key.json --instance_ids sherlock/val_retrieval/val_retrieval_$i\_instance_ids.json; 

done;

python ./score_localization.py val_nips_predictions_2/localization.npy sherlock/val_localization/val_localization_answer_key.json --instance_ids sherlock/val_localization/val_localization_instance_ids.json
