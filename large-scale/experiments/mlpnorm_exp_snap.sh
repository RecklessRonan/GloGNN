#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

dataset=snap-patents
sub_dataset=${2:-''}

lr_lst=(0.005)
hidden_channels_lst=(64)
dropout_lst=(0.7 0.8 0.9)
weight_decay_lst=(1e-3 1e-2 1e-1)
alpha_lst=(0.0)
beta_lst=(1)
gamma_lst=(0.6 0.7 0.8)
delta_lst=(0.3 0.5 0.7)
norm_layers_lst=(1)
orders_lst=(3)

epochs=100
runs=5
norm_func_id=1
order_func_id=2


for lr in "${lr_lst[@]}"; do
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for beta in "${beta_lst[@]}"; do
            for gamma in "${gamma_lst[@]}"; do
                for norm_layers in "${norm_layers_lst[@]}"; do
                    for dropout in "${dropout_lst[@]}"; do
                        for weight_decay in "${weight_decay_lst[@]}"; do
                            for orders in "${orders_lst[@]}"; do
                                for alpha in "${alpha_lst[@]}"; do
                                    for delta in "${delta_lst[@]}"; do
                                        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                            python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method mlpnorm --epochs $epochs --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --delta $delta --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 1 --runs $runs --directed
                                        else
                                            python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method mlpnorm --epochs $epochs --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --delta $delta --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 1 --runs $runs
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done 
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 