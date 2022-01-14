#!/bin/bash

# dataset_lst=(pokec snap-patents arxiv-year genius fb100 twitch-gamer)
# hidden_channel_lst=(80 64 32 16 8)
# weight_decay_lst=(1e-7 1e-2)
# dropout_lst=(0.0 0.7)
# decay_rate_lst=(0.0 1.5)
# num_layers_lst=(3 2 1)

dataset_lst=(arxiv-year)
hidden_channel_lst=(1)
weight_decay_lst=(1e-7)
dropout_lst=(0.0)
decay_rate_lst=(0.0)
num_layers_lst=(1)
display_step=1

for dataset in "${dataset_lst[@]}"; do
    for num_layers in "${num_layers_lst[@]}"; do
        for weight_decay in "${weight_decay_lst[@]}"; do
            for dropout in "${dropout_lst[@]}"; do
                for decay_rate in "${decay_rate_lst[@]}"; do
                    for hidden_channel in "${hidden_channel_lst[@]}"; do
                        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                python main.py --dataset $dataset --sub_dataset None --method ggcn --lr 0.01 \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay --decay_rate $decay_rate \
                                --display_step $display_step --runs 5 --directed
                        else
                            if [ "$dataset" = "fb100" ]; then
                                python main.py --dataset $dataset --sub_dataset Penn94 --method ggcn --lr 0.01 \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay --decay_rate $decay_rate \
                                --display_step $display_step --runs 5
                            else
                                python main.py --dataset $dataset --sub_dataset None --method ggcn --lr 0.01 \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay --decay_rate $decay_rate \
                                --display_step $display_step --runs 5
                            fi
                        fi
                    done
                done
            done
        done
    done
done