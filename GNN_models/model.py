import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dgl_nn


class NodeEmbeddingGraphSage(nn.Module):
    def __init__(self,
                 in_feats,
                 hid_feats,
                 out_feats,
                 rel_names):
        super(NodeEmbeddingGraphSage, self).__init__()
        self.conv1 = dgl_nn.HeteroGraphConv(
            {
                rel: dgl_nn.GraphConv(in_feats,hid_feats,norm='right')
                for rel in rel_names
            }
        )
        self.conv2 = dgl_nn.HeteroGraphConv(
            {
                rel: dgl_nn.GraphConv(hid_feats, hid_feats, norm='right')
                for rel in rel_names
            }
        )
        self.conv3 = dgl_nn.HeteroGraphConv(
            {
                rel: dgl_nn.GraphConv(hid_feats, out_feats, norm='right')
                for rel in rel_names
            }
        )

    def forward(self,blocks,inputs):
        h = F.relu(self.conv1(blocks[0],inputs))
        h = F.relu(self.conv2(blocks[1]),h)
        h = F.relu(self.conv3(blocks[2]),h)
        return h


class EdgeScorePredictor(nn.Module):
    def __init__(self,num_classes,in_feats):
        super(EdgeScorePredictor, self).__init__()
        self.W = nn.Linear(2*in_feats,num_classes)

    def apply_edge(self,edges):
        data = torch.cat(edges.src['h'],edges.dst['h'])
        return {'score':self.W(data)}

    def forward(self,edge_subgraph,h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(self.apply_edge,etype=etype)
            return edge_subgraph.edata['score']


class GNNModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hid_feats,
                 out_feats,
                 num_classes,
                 etypes):
        super(GNNModel, self).__init__()
        self.rgcn = NodeEmbeddingGraphSage(in_feats,hid_feats,out_feats,etypes)
        self.pred = EdgeScorePredictor(num_classes,out_feats)

    def forward(self,edge_subgraph,blocks,input):
        h = self.rgcn(blocks,input)
        return  {'node_embedding':h,'edge_embedding':self.pred(edge_subgraph,h)}


                
