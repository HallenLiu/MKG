import dgl
import torch
import pandas as pd


data = pd.read_csv('../dataset/FB15k-237/train.tsv',sep='\t',names=['src','rel','dst'])
# nodes = pd.read_csv('../dataset/FB15k-237/entities.txt',names=['data'])


data_dict = {(data.loc[index]['src'],data.loc[index]['rel'],data.loc[index]['dst']): (torch.LongTensor([0]),torch.LongTensor([0])) for index in range(len(data))}
# print(data_dict)
num_nodes_dict={node: 1 for node in nodes['data']}
# print(num_nodes_dict)
graph = dgl.heterograph(data_dict=data_dict)
# print(graph)
dgl.save_graphs('../dataset/FB15k-237/FB15k-237.bin',graph)
