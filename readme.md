# GloGNN

This is the official repository of paper *Finding Global Homophily in Graph Neural Networks When Meeting Heterophily*.

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.10.0

- torch-geometric==2.0.2

- networkx==2.3

- scipy==1.5.4

- numpy==1.19.2

- sklearn==0.0

- matplotlib==3.1.1

- pandas==1.1.5

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- large-scale # experiments for 6 large-scale datasets
    |-- data/ # some large-scale datasets
    |-- dataset/  # the remaining large-scale datasets
    |-- experiments/  # all run shs
    |-- main.py # the main code
    |-- main_z.py # obtains coefficient matrix z 
    |-- models.py # includes all model implementations
|-- paper-plots # all experimental plots in our paper
|-- small-scale # experiments for 9 small-scale datasets
    |-- data/ # 3 old datasets, including cora, citeseer, and pubmed
    |-- new-data/ # 6 new datasets, including texas, wisconsin, cornell, actor, squirrel, and chameleon
    |-- splits/ # splits for 6 new datasets
    |-- sh/ # all run shs
    |-- main.py  # the main code
    |-- main_z.py  # obtains coefficient matrix z
    |-- main_h.py # obtains final layer embedding h
```

## Run pipeline for big-scale datasets
1. Entering the large-scale directory and download 6 big-scale datasets from the repository of LINKX[https://github.com/CUAI/Non-Homophily-Large-Scale]. Notice, you should rename the datasets and place them in the right directory.
```python
cd large-scale
```

2. You can run any models implemented in `models.py'. For examples, you can run our model on `genius' dataset by the script:
```python
python -u main.py --dataset genius --sub_dataset None --method mlpnorm
```
