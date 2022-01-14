# from left to right
# lr, hidden_dim, dropout, weight_decay, beta1, beta2, gamma, alpha, norm_layers, orders

# glognn
best_config12 = {
    'arxiv-year': [0.005, 256, 0.7, 0.001, 0.0, 1.0, 0.7, 0.1, 1, 1],
    'genius': [0.001, 256, 0.0, 0.0, 0.0, 1.0, 0.9, 0.5, 2, 2],
    'pokec': [0.005, 128, 0.5, 0.01, 0.0, 1.0, 0.5, 0.5, 1, 1],
    'fb100': [0.001, 256, 0.5, 0.1, 0.0, 1.0, 0.5, 0.5, 2, 2],
    'twitch-gamer': [0.01, 256, 0.9, 0.01, 0.0, 1.0, 0.1, 0.5, 1, 1],
    'snap-patents': [0.005, 64, 0.8, 0.01, 0.0, 1.0, 0.6, 0.5, 1, 3]
}

# glognn++
best_config22 = {
    'arxiv-year': [0.005, 256, 0.7, 0.1, 0.1, 1.0, 0.7, 0.1, 1, 1],
    'genius': [0.001, 256, 0.0, 0.0, 1.0, 0.1, 0.9, 0.5, 2, 3],
    'pokec': [0.005, 128, 0.5, 0.01, 0.0, 1.0, 0.5, 0.5, 1, 1],
    'fb100': [0.001, 256, 0.5, 0.1, 0.0, 1.0, 0.5, 0.5, 2, 2],
    'twitch-gamer': [0.01, 256, 0.8, 0.1, 1.0, 0.1, 0.2, 0.2, 1, 1],
    'snap-patents': [0.005, 64, 0.8, 0.01, 0.0, 1.0, 0.7, 0.3, 1, 3]
}

run_shs = ""

datasets = ['arxiv-year', 'genius', 'pokec',
            'fb100', 'twitch-gamer', 'snap-patents']


def get_sh(paras, d, sub_dataset, norm_func):
    sh = 'python -u main.py --dataset ' + d + ' --sub_dataset ' + sub_dataset + \
        ' --method mlpnorm --epochs 500 --hidden_channels ' + str(paras[1]) + \
        ' --lr ' + str(paras[0]) + ' --dropout ' + str(paras[2]) + ' --weight_decay ' + str(paras[3]) +\
        ' --alpha ' + str(paras[4]) + ' --beta ' + str(paras[5]) + ' --gamma ' + str(paras[6]) +\
        ' --delta ' + str(paras[7]) + ' --norm_func_id ' + str(norm_func) + ' --norm_layers ' + str(paras[8]) +\
        ' --orders_func_id ' + str(2) + ' --orders ' + \
        str(paras[9]) + ' --display_step 1 --runs 5 '
    if d == 'snap-patents' or d == 'arxiv-year':
        sh += '--directed'
    sh += '\n'
    return sh


for d in datasets:
    if d == 'fb100':
        sub_dataset = 'Penn94'
    else:
        sub_dataset = 'None'

    paras = best_config12[d]
    run_shs += get_sh(paras, d, sub_dataset, 1)

    # paras = best_config12[d] * 1
    # paras[5] = 0.0001
    # run_shs += get_sh(paras, d, sub_dataset, 1)

    # paras = best_config12[d] * 1
    # paras[7] = 0.0
    # run_shs += get_sh(paras, d, sub_dataset, 1)

    # paras = best_config12[d] * 1
    # paras[7] = 1.0
    # run_shs += get_sh(paras, d, sub_dataset, 1)

    paras = best_config22[d]
    run_shs += get_sh(paras, d, sub_dataset, 2)

    # paras = best_config22[d] * 1
    # paras[5] = 0.0001
    # run_shs += get_sh(paras, d, sub_dataset, 2)

    # paras = best_config22[d] * 1
    # paras[7] = 0.0
    # run_shs += get_sh(paras, d, sub_dataset, 2)

    # paras = best_config22[d] * 1
    # paras[7] = 1.0
    # run_shs += get_sh(paras, d, sub_dataset, 2)


with open('ablation_study.sh', 'w') as f:
    f.writelines(run_shs)
