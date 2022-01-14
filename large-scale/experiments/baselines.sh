#!/bin/bash

# datasets=(pokec snap-patents arxiv-year genius fb100 twitch-gamer)
# h2gcn can only experiment arxiv-year and fb100
datasets=(pokec)
baselines=(gprgnn linkx)


hidden_channels_lst=(8)
num_layers_lst=(2)
dropout_lst=(0)

display_step=1

for baseline in "${baselines[@]}"; do
	for dataset in "${datasets[@]}"; do
		for hidden_channels in "${hidden_channels_lst[@]}"; do
			for num_layers in "${num_layers_lst[@]}"; do
				for dropout in "${dropout_lst[@]}"; do
					if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
						python main.py --dataset $dataset --sub_dataset None\
						--method $baseline --num_layers $num_layers --hidden_channels $hidden_channels \
						--display_step $display_step --runs 5 --dropout $dropout  --directed
					else
						if [ "$dataset" = "fb100" ]; then
							python main.py --dataset $dataset --sub_dataset Penn94 \
							--method $baseline --num_layers $num_layers --hidden_channels $hidden_channels \
							--display_step $display_step --runs 5 --dropout $dropout 
						else
							python main.py --dataset $dataset --sub_dataset None\
							--method $baseline --num_layers $num_layers --hidden_channels $hidden_channels \
							--display_step $display_step --runs 5 --dropout $dropout 
						fi
					fi
				done
			done
		done
	done            
done
