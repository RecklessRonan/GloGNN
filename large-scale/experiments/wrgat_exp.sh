#!/bin/bash
startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=(32)
lr_lst=(0.01)
dropout_lst=(0.5)
epochs=500
runs=5



for hidden_channels in "${hidden_channels_lst[@]}"; do
    for lr in "${lr_lst[@]}"; do
		for dropout in "${dropout_lst[@]}"; do
			if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
				python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method wrgat --epochs $epochs --lr $lr --hidden_channels $hidden_channels --dropout $dropout --original_edges --display_step 1 --runs $runs --directed
			else
				python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method wrgat --epochs $epochs --lr $lr --hidden_channels $hidden_channels --dropout $dropout --original_edges --display_step 1 --runs $runs 
			fi
		done
    done
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 