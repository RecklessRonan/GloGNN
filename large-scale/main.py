import argparse
import sys
import os
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from correct_smooth import double_correlation_autoscale, double_correlation_fixed
from data_utils import evaluate_acmgcn, normalize, gen_normalized_adjs, evaluate, evaluate_mlpnorm, evaluate_acmgcn, eval_acc, eval_rocauc, to_sparse_tensor, load_fixed_splits
from parse import parse_method, parser_add_main_args
import faulthandler
faulthandler.enable()


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
seed = 0
np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

if args.method == 'mlpnorm':
    torch.set_default_dtype(torch.float64)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
if args.cpu:
    device = torch.device('cpu')

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

if args.rand_split or args.dataset in ['ogbn-proteins', 'wiki']:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)

if args.dataset == 'ogbn-proteins':
    if args.method == 'mlp' or args.method == 'cs':
        dataset.graph['node_feat'] = scatter(dataset.graph['edge_feat'], dataset.graph['edge_index'][0],
                                             dim=0, dim_size=dataset.graph['num_nodes'], reduce='mean')
    else:
        dataset.graph['edge_index'] = to_sparse_tensor(dataset.graph['edge_index'],
                                                       dataset.graph['edge_feat'], dataset.graph['num_nodes'])
        dataset.graph['node_feat'] = dataset.graph['edge_index'].mean(dim=1)
        dataset.graph['edge_index'].set_value_(None)
    dataset.graph['edge_feat'] = None

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize matters a lot!! pay attention to this
# e.g. directed edges are temporally useful in arxiv-year,
# so we usually do not symmetrize, but for label prop symmetrizing helps
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])


# used in GGCN
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    # row_sum = np.array(adj.sum(1))
    # row_sum = (row_sum == 0)*1+row_sum
    # adj_normalized = adj/row_sum
    return sp.coo_matrix(adj_normalized)


def precompute_degree_s(adj):
    adj_i = adj._indices()
    adj_v = adj._values()
    # print('adj_i', adj_i.shape)
    # print(adj_i)
    # print('adj_v', adj_v.shape)
    # print(adj_v)
    adj_diag_ind = (adj_i[0, :] == adj_i[1, :])
    adj_diag = adj_v[adj_diag_ind]
    # print(adj_diag)
    # print(adj_diag[0])
    v_new = torch.zeros_like(adj_v)
    for i in tqdm(range(adj_i.shape[1])):
        # print('adj_i[0,', i, ']', adj_i[0, i])
        v_new[i] = adj_diag[adj_i[0, i]]/adj_v[i]-1
    degree_precompute = torch.sparse.FloatTensor(
        adj_i, v_new, adj.size())
    return degree_precompute


def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high


if args.method == 'mlpnorm':
    x = dataset.graph['node_feat']
    edge_index = dataset.graph['edge_index']
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(
        dataset.graph['num_nodes'], dataset.graph['num_nodes'])).to_torch_sparse_coo_tensor()
    # adj = adj.to_dense()
    x = x.to(device)
    adj = adj.to(device)
    x = x.to(torch.float64)
    adj = adj.to(torch.float64)
elif args.method == 'ggcn':
    # adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(
    #     dataset.graph['num_nodes'], dataset.graph['num_nodes'])).to_torch_sparse_coo_tensor()
    # adj = adj.to_dense()
    # edge_index_nx = []
    # for i in tqdm(range(len(edge_index[0]))):
    #     edge_index_nx.append((edge_index[0][i], edge_index[1][i]))
    # G = nx.Graph()
    # G.add_edges_from(edge_index_nx)
    # adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    # adj = to_dense_adj(edge_index)
    # adj = dense_to_sparse(adj)
    x = dataset.graph['node_feat']
    edge_index = dataset.graph['edge_index']

    adj_pt = 'ggcn_features/' + args.dataset + '_adj.pt'
    degree_adj_pt = 'ggcn_features/' + args.dataset + '_degree_adj.pt'
    if os.path.exists(adj_pt):
        adj = torch.load(adj_pt)
    else:
        adj = to_scipy_sparse_matrix(edge_index)
        adj = row_normalized_adjacency(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        torch.save(adj, adj_pt)

    # if os.path.exists(degree_adj_pt):
    #     degree_adj = torch.load(degree_adj_pt)
    # else:
    #     degree_adj = precompute_degree_s(adj)

    x = x.to(device)
    adj = adj.to(device)
elif args.method == 'acmgcn':
    x = dataset.graph['node_feat']
    edge_index = dataset.graph['edge_index']
    adj_low_pt = 'acmgcn_features/' + args.dataset + '_adj_low.pt'
    adj_high_pt = 'acmgcn_features/' + args.dataset + '_adj_high.pt'
    if os.path.exists(adj_low_pt) and os.path.exists(adj_high_pt):
        adj_low = torch.load(adj_low_pt)
        adj_high = torch.load(adj_high_pt)
    else:
        adj_low = to_scipy_sparse_matrix(edge_index)
        adj_low = row_normalized_adjacency(adj_low)
        # print(adj_low)
        adj_high = get_adj_high(adj_low)
        # print(adj_high)
        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low)
        adj_high = sparse_mx_to_torch_sparse_tensor(adj_high)
        # adj_high = (torch.eye(n) - adj_low).to_sparse()
        torch.save(adj_low, adj_low_pt)
        torch.save(adj_high, adj_high_pt)
    x = x.to(device)
    adj_low = adj_low.to(device)
    adj_high = adj_high.to(device)
else:
    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(
            device), dataset.graph['node_feat'].to(device)
train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")


# sys.exit()
### Load method ###

model = parse_method(args, dataset, n, c, d, device)
# print('model', next(model.parameters()).device)


# using rocauc as the eval function
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc

logger = Logger(args.runs, args)


if args.method == 'cs':
    cs_logger = SimpleLogger('evaluate params', [], 2)
    model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}'
    model_dir = f'models/{model_path}'
    print(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    DAD, AD, DA = gen_normalized_adjs(dataset)

if args.method == 'lp':
    # handles label propagation separately
    for alpha in (.01, .1, .25, .5, .75, .9, .99):
        logger = Logger(args.runs, args)
        for run in range(args.runs):
            split_idx = split_idx_lst[run]
            train_idx = split_idx['train']
            model.alpha = alpha
            out = model(dataset, train_idx)
            result = evaluate(model, dataset, split_idx, eval_func, result=out)
            logger.add_result(run, result[:-1])
            print(f'alpha: {alpha} | Train: {100*result[0]:.2f} ' +
                  f'| Val: {100*result[1]:.2f} | Test: {100*result[2]:.2f}')

        best_val, best_test = logger.print_statistics()
        filename = f'results/{args.dataset}.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
            write_obj.write(f"{args.method}," + f"{sub_dataset}" +
                            f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
                            f"{best_test.mean():.3f} ± {best_test.std():.3f}\n")
    sys.exit()


model.train()
print('MODEL:', model)

### Training loop ###
last_time = time.time()

for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    if args.sampling:
        if args.num_layers == 2:
            sizes = [15, 10]
        elif args.num_layers == 3:
            sizes = [15, 10, 5]
        train_loader = NeighborSampler(dataset.graph['edge_index'], node_idx=train_idx,
                                       sizes=sizes, batch_size=1024,
                                       shuffle=True, num_workers=12)
        subgraph_loader = NeighborSampler(dataset.graph['edge_index'], node_idx=None, sizes=[-1],
                                          batch_size=4096, shuffle=False,
                                          num_workers=12)

    model.reset_parameters()
    if args.adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.SGD:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, nesterov=args.nesterov, momentum=args.momentum)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('-inf')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', factor=0.9, patience=10, verbose=True, min_lr=0.001)
    for epoch in range(args.epochs):
        model.train()

        if not args.sampling:
            optimizer.zero_grad()
            if args.method == 'mlpnorm' or args.method == 'ggcn':
                # print('x', x.device)
                # print(x.shape)
                # print('adj', adj.device)
                # print(adj.shape)
                # print(adj)
                # print('model', next(model.parameters()).device)
                out = model(x, adj)
            elif args.method == 'acmgcn':
                out = model(x, adj_low, adj_high)
            else:
                out = model(dataset)
            #loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx].type_as(out))

            # print('out', out.shape)
            if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
                if dataset.label.shape[1] == 1:
                    # change -1 instances to 0 for one-hot transform
                    # dataset.label[dataset.label==-1] = 0
                    true_label = F.one_hot(
                        dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label

                loss = criterion(out[train_idx], true_label.squeeze(1)[
                    train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(
                    out[train_idx], dataset.label.squeeze(1)[train_idx])
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)
        else:
            pbar = tqdm(total=train_idx.size(0))
            pbar.set_description(f'Epoch {epoch:02d}')

            for batch_size, n_id, adjs in train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                print('batch_size', batch_size.shape)
                print('n_id', n_id.shape)
                print('adjs', adjs.shape)
                adjs = [adj.to(device) for adj in adjs]

                optimizer.zero_grad()
                out = model(dataset, adjs, dataset.graph['node_feat'][n_id])
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, dataset.label.squeeze(1)
                                 [n_id[:batch_size]])
                loss.backward()
                optimizer.step()
                pbar.update(batch_size)
            pbar.close()
        if args.method == 'mlpnorm' or args.method == 'ggcn':
            result = evaluate_mlpnorm(model, x, adj, dataset, split_idx, eval_func,
                                      sampling=args.sampling, subgraph_loader=subgraph_loader)
        elif args.method == 'acmgcn':
            result = evaluate_acmgcn(model, x, adj_low, adj_high, dataset, split_idx, eval_func,
                                     sampling=args.sampling, subgraph_loader=subgraph_loader)
        else:
            result = evaluate(model, dataset, split_idx, eval_func,
                              sampling=args.sampling, subgraph_loader=subgraph_loader)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            if args.dataset != 'ogbn-proteins':
                best_out = F.softmax(result[-1], dim=1)
            else:
                best_out = result[-1]

        now_time = time.time()
        time_elapsed = now_time - last_time
        last_time = now_time

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Time: {time_elapsed:.4f}')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(
                    return_counts=True)[1].float()/pred.shape[0])
    logger.print_statistics(run)
    if args.method == 'cs':
        torch.save(best_out, f'{model_dir}/{run}.pt')
        _, out_cs = double_correlation_autoscale(dataset.label, best_out.cpu(),
                                                 split_idx, DAD, 0.5, 50, DAD, 0.5, 50, num_hops=args.hops)
        result = evaluate(model, dataset, split_idx, eval_func, out_cs)
        cs_logger.add_result(run, (), (result[1], result[2]))


### Save results ###
if args.method == 'cs':
    print('Valid acc -> Test acc')
    res = cs_logger.display()
    best_val, best_test = res[:, 0], res[:, 1]
else:
    best_val, best_test = logger.print_statistics()
filename = f'results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.method}," + f"{sub_dataset}" +
                    f"{best_val.mean():.3f}, {best_val.std():.3f}," +
                    f"{best_test.mean():.3f}, {best_test.std():.3f}\n")
