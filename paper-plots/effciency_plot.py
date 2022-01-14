import sys
import argparse
import ujson as json
import matplotlib.pyplot as plt
import numpy as np
from time import time

n_seed = 3


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log-path', type=str, default='efficiency_logs/')
    # parser.add_argument('--log-file', type=str, default='')
    parser.add_argument('--dataset', type=str, default='freebase')
    parser.add_argument('--title', type=str, default='DBLP')
    parser.add_argument('--epochs', type=int, default='500')
    args = parser.parse_args()

    return args


def plot_loss(args):

    models = ['H2GCN', 'GPR-GNN', 'ACM-GCN', 'LINKX', 'GloGNN']  # 'Hetgnn',
    colors = ['goldenrod', 'lawngreen', 'blue', 'orchid', 'gray']
    # fill_colors=['pink','yellow','blue','orchid']
    n_models = len(models)

    plt.figure()
    plt.title(args.title)

    fig, ax = plt.subplots()
    # plt.xlim(0, args.epochs)
    plt.xlim(0, 200)
    # x_axis = np.arange(1,args.epochs)
    # x_axis = np.arange(1,args.epochs+1,20)
    # plt.ylim(0, 1)
    for i, model in enumerate(models):
        epoches = [[]for _ in range(n_seed)]
        time = [[]for _ in range(n_seed)]
        val_loss = [[]for _ in range(n_seed)]
        train_loss = [[]for _ in range(n_seed)]
        val_acc = [[]for _ in range(n_seed)]
        train_acc = [[]for _ in range(n_seed)]
        test_acc = [[]for _ in range(n_seed)]
        for seed in range(n_seed):
            with open('{}/{}/{}_{}.txt'.format(args.log_path, args.dataset, model, seed), mode='r') as f:
                print('file: {}/{}/{}_{}.txt'.format(args.log_path,
                      args.dataset, model, seed))
                for line in f:
                    if '{' not in line:
                        continue
                    if 'Graph' in line or 'num' in line:
                        continue

                    # print(line)

                    line = json.loads(line)
                    # if 'train_loss' in line:
                    #     train_loss[seed].append(line['train_loss'])
                    # if 'val_loss' in line:
                    #     val_loss[seed].append(line['val_loss'])

                    if 'epoch' in line:
                        epoches[seed].append(line['epoch'])
                    if 'time' in line:
                        time[seed].append(line['time'])
                    # if 'train_metric' in line:
                    #     if line['epoch_progress'] == 0:
                    #         train_acc.append(line['train_metric']['accuracy'])
                    if 'val_metric' in line:
                        val_acc[seed].append(line['val_metric']['accuracy'])
                    # if 'test_metric' in line:
                    #     test_acc[seed].append(line['test_metric']['accuracy'])

            l = min(len(val_acc[0]),  len(epoches[0]),
                    args.epochs)  # len(test_acc[0])

            print(l)

            train_loss[seed] = train_loss[seed][:l]
            val_loss[seed] = val_loss[seed][:l]
            test_acc[seed] = test_acc[seed][:l]
            val_acc[seed] = val_acc[seed][:l]
            epoches[seed] = epoches[seed][:l]
            time[seed] = time[seed][:l]
            # print(len(time[seed]))
        # epoches = np.array(epoches)[:,:l]
        val_acc = np.array(val_acc)[:, :l]
        time = np.array(time)[:, :l]

        # print (time)
        #

        val_acc = val_acc[:, :-1:1]
        time = time[:, :-1:1]

    # plt.subplot(211)
        # ax.plot(x_axis, np.mean(val_acc,axis=0), colors[i], label=model)
        # ax.fill_between(x_axis, np.max(val_acc,axis=0),np.min(val_acc,axis=0),color=colors[i], alpha=0.2)
        ax.plot(np.mean(time, axis=0), np.mean(
            val_acc, axis=0), colors[i], label=model)
        # ax.fill_between(np.mean(time,axis=0), np.max(val_acc,axis=0),np.min(val_acc,axis=0),color=colors[i], alpha=0.2)
    # plt.plot(epoches, val_acc, 'g', label='val')
    plt.ylabel('accuracy')
    # plt.xlabel('epoches')
    plt.xlabel('time')
    plt.legend(loc='lower right')

    # plt.subplot(212)
    # plt.plot(epoches, test_acc, 'r', label='test')
    # plt.plot(epoches, val_acc, 'g', label='val')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoches')
    # plt.legend(loc='lower right')
    # plt.suptitle('file: {}{}.txt'.format(args.log_path, args.log_file))
    plt.show()
    plt.savefig('{}/{}_val.pdf'.format(args.log_path, args.dataset))


if __name__ == "__main__":
    args = parse_args()
    plot_loss(args)
