#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=(256 128 64 32 16)
lr_lst=(0.01 0.05 0.002)


for hidden_channels in "${hidden_channels_lst[@]}"; do
    for lr in "${lr_lst[@]}"; do
		if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
			python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method gprgnn --lr $lr --hidden_channels $hidden_channels  --display_step 1 --runs 5 --directed
		else
			python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method gprgnn --lr $lr --hidden_channels $hidden_channels  --display_step 1 --runs 5
		fi
    done
done
