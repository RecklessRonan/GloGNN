#!/bin/bash

# dataset_lst=(pokec snap-patents arxiv-year genius fb100 twitch-gamer)
# hidden_channel_lst=(80 64 32 16 8)
# weight_decay_lst=(1e-7 1e-2)
# dropout_lst=(0.0 0.7)
# decay_rate_lst=(0.0 1.5)
# num_layers_lst=(3 2 1)

dataset_lst=(arxiv-year)
hidden_channel_lst=(256)
weight_decay_lst=(0.005)
dropout_lst=(0.3)
lr_lst=(0.01)
num_layers_lst=(3)
display_step=1

for dataset in "${dataset_lst[@]}"; do
    for lr in "${lr_lst[@]}"; do
        for num_layers in "${num_layers_lst[@]}"; do
            for weight_decay in "${weight_decay_lst[@]}"; do
                for dropout in "${dropout_lst[@]}"; do
                    for hidden_channel in "${hidden_channel_lst[@]}"; do
                        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                python -u main.py --dataset $dataset --sub_dataset None --method acmgcn --lr $lr \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay  \
                                --display_step $display_step --runs 5 --directed
                        else
                            if [ "$dataset" = "fb100" ]; then
                                python -u main.py --dataset $dataset --sub_dataset Penn94 --method acmgcn --lr $lr \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay  \
                                --display_step $display_step --runs 5
                            else
                                python -u main.py --dataset $dataset --sub_dataset None --method acmgcn --lr $lr \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay  \
                                --display_step $display_step --runs 5
                            fi
                        fi
                    done
                done
            done
        done
    done
done