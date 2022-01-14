 python -u main.py --dataset snap-patents --sub_dataset None --method acmgcn --lr 0.01 \
                                --num_layers 2 --hidden_channels 64 --dropout 0.1 \
                                --weight_decay 0.00005  \
                                --display_step 1 --runs 5 --directed
 python -u main.py --dataset arxiv-year --sub_dataset None --method acmgcn --lr 0.01 \
                                --num_layers 2 --hidden_channels 64 --dropout 0.1 \
                                --weight_decay 0.0001  \
                                --display_step 1 --runs 5 --directed
 python -u main.py --dataset fb100 --sub_dataset Penn94 --method acmgcn --lr 0.001 \
                                --num_layers 2 --hidden_channels 64 --dropout 0.6 \
                                --weight_decay 0.0005  \
                                --display_step 1 --runs 5
python -u main.py --dataset genius --sub_dataset None --method acmgcn --lr 0.01 \
                                --num_layers 2 --hidden_channels 64 --dropout 0.5 \
                                --weight_decay 0.0005  \
                                --display_step 1 --runs 5
python -u main.py --dataset pokec --sub_dataset None --method acmgcn --lr 0.01 \
                                --num_layers 2 --hidden_channels 64 --dropout 0.0 \
                                --weight_decay 0.00001  \
                                --display_step 1 --runs 5
python -u main.py --dataset twitch-gamer --sub_dataset None --method acmgcn --lr 0.01 \
                                --num_layers 2 --hidden_channels 64 --dropout 0.0 \
                                --weight_decay 0.001  \
                                --display_step 1 --runs 5
