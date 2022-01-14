# %%
import torch
import numpy as np
import networkx as nx
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator

# %%
datasets = ['wisconsin', 'cornell', 'texas', 'film', 'chameleon', 'squirrel']

for dataset in datasets:

    # %%
    z_dir = '../pygcn/z_vis/' + dataset + '/0/'

    # %%
    graph_edges = '../pygcn/new_data/' + dataset + '/out1_graph_edges.txt'
    with open(graph_edges, 'r') as f:
        lines = f.readlines()

    # %%
    node_nums = {
        'texas': 183,
        'wisconsin': 251,
        'cornell': 183,
        'film': 7600,
        'squirrel': 5201,
        'chameleon': 2277,
        'cora': 2708,
        'citeseer': 3327,
        'pubmed': 19717
    }

    node_num = node_nums[dataset]
    adj = np.zeros((node_num, node_num))

    for i in range(1, len(lines)):
        line = lines[i]
        line = line.rstrip().split('\t')
        adj[int(line[0]), int(line[1])] = 1
        adj[int(line[1]), int(line[0])] = 1

    # %%
    # adj

    # %%
    G = nx.from_numpy_matrix(adj)

    # %%
    # path_dir = 'hop_pkls/' + dataset + '_path.pkl'

    # if os.path.exists(path_dir):
    #     with open(path_dir, 'rb') as f:
    #         path = pickle.load(f)
    # else:
    #     path = dict(nx.all_pairs_shortest_path(G))
    #     with open(path_dir, 'wb') as f:
    #         pickle.dump(path, f)

    # %%
    path_length_dir = 'hop_pkls/' + dataset + '_path_length.pkl'

    if os.path.exists(path_length_dir):
        with open(path_length_dir, 'rb') as f:
            path_length = pickle.load(f)
    else:
        path_length = dict(nx.all_pairs_shortest_path_length(G))
        with open(path_length_dir, 'wb') as f:
            pickle.dump(path_length, f)

    # %%
    distance = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            distance[i][j] = path_length[i][j]

    # %%
    unique, counts = np.unique(distance, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    # %%
    # unique

    # %%
    # counts

    # %%
    epoch_list = []
    for file in os.listdir(z_dir):
        if file != 'label.pt' and file != 'adj.pt':
            # print(file)
            name = file.split('.')[0]
            name = name.split('_')[1]
            epoch_list.append(int(name))

    epoch = max(epoch_list)
    # print(epoch)

    # %%
    z = torch.load(z_dir+'z_' + str(epoch) + '.pt')

    # %%
    # z.shape

    # %%
    z = z.detach().numpy()

    # %%
    labels = torch.load(z_dir + 'label.pt')

    # %%
    # labels.shape

    # %%
    labels = labels.detach().numpy()

    # %%
    z_pos_sum = [0] * unique.shape[0]
    friends_sum = [0] * unique.shape[0]

    for i in range(node_num):
        for j in range(node_num):
            if labels[i] == labels[j]:
                friends_sum[int(distance[i][j])] += 1
                if z[i][j] > 0:
                    z_pos_sum[int(distance[i][j])] += 1

    if friends_sum[-1] == 0:
        z_pos_sum = z_pos_sum[:-1]
        friends_sum = friends_sum[:-1]

    # print('z_pos_sum', z_pos_sum)
    # print('friends_sum', friends_sum)

    # %%
    z_pos_sum_log = [np.log2(p) for p in z_pos_sum]
    friends_sum_log = [np.log2(f) for f in friends_sum]

    z_pos_sum_log = np.around(z_pos_sum_log, 2)
    friends_sum_log = np.around(friends_sum_log, 2)

    # print('z_pos_sum_log', z_pos_sum_log)
    # print('friends_sum_log', friends_sum_log)

    # %%
    # counts

    # %%
    ratio = [z_pos_sum[i]/counts[i] for i in range(len(z_pos_sum))]
    # ratio

    # %%
    # len(z_pos_sum)

    # %%
    matplotlib.rcParams.update({'font.size': 15})

    # %%
    plt.figure()
    width = 0.35
    plt.xlim(0, len(z_pos_sum))
    plt.ylabel('Number (log2)')
    plt.xlabel('Hops')
    ax = plt.gca()
    y_major_locator = MultipleLocator(4)
    ax.yaxis.set_major_locator(y_major_locator)
    hops = np.arange(1, len(z_pos_sum), 1)
    # new_ratio = [r * 100 for r in ratio[1:]]
    # plt.plot(hops, new_ratio, marker='*')
    plt.bar(hops - width/2, z_pos_sum_log[1:], label='z_pos_num',
            width=width, hatch='//', edgecolor='black', color='coral')
    plt.bar(hops + width/2, friends_sum_log[1:], label='friends_num',
            width=width, hatch="", edgecolor='black', color='royalblue')
    plt.legend(loc='upper right')
    plt.savefig('case_study/case_study_{}.pdf'.format(dataset),
                bbox_inches='tight')
    plt.show()

# %%
