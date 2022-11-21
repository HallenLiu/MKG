import dgl
import torch.nn as nn
import dgl.nn.pytorch as dgl_nn
import torch.nn.functional as F

graph = dgl.load_graphs('../dataset/FB15k-237/FB15k-237.bin')

