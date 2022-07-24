# -*- coding: utf-8 -*-
import inspect
import logging
import os.path
import pickle
from itertools import islice
from pathlib import Path
from time import time

dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

folder_pickles = dir_f+"/pickles/"
Path(folder_pickles).mkdir(parents=True, exist_ok=True)

def returnPathStruc2vec():
    return dir_f

def isPickle(fname):
    return os.path.isfile(dir_f+'/pickles/'+fname+'.pickle')

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

def partition(lst, n):
    lst = list(lst)
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def restoreVariableFromDisk(name):
    logging.info('Recovering variable...')
    t0 = time()
    val = None
    with open(folder_pickles + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
    t1 = time()
    logging.info('Variable recovered. Time: {}m'.format((t1-t0)/60))

    return val



def saveVariableOnDisk(f,name):
    logging.info('Saving variable on disk...')
    t0 = time()
    with open(folder_pickles + name + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()
    logging.info('Variable saved. Time: {}m'.format((t1-t0)/60))

    return
