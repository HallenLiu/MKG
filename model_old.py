import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dgl_nn
from dgl.nn.pytorch import RelGraphConv


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

class ScorePredictor(nn.Module):
    def forward(self,edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('h','h','score'),etype=etype)
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


class PredModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hid_feats,
                 out_feats,
                 num_classes,
                 etypes):
        super(PredModel, self).__init__()
        self.rgcn = NodeEmbeddingGraphSage(in_feats,hid_feats,out_feats,etypes)
        self.pred = ScorePredictor()

    def forward(self,positive_graph,negative_graph,blocks,x):
        h = self.rgcn(blocks,x)
        pos_score = self.pred(positive_graph,h)
        neg_score = self.pred(negative_graph,h)
        return pos_score,neg_score




class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels):
        super().__init__()
        # two-layer RGCN
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.conv1 = RelGraphConv(h_dim, h_dim, num_rels, regularizer='bdd',
                                  num_bases=100, self_loop=True)
        self.conv2 = RelGraphConv(h_dim, h_dim, num_rels, regularizer='bdd',
                                  num_bases=100, self_loop=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, nids):
        x = self.emb(nids)
        h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], g.edata['norm']))
        h = self.dropout(h)
        h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata['norm'])
        return self.dropout(h)

class LinkPredict(nn.Module):
    def __init__(self, num_nodes, num_rels, h_dim = 500, reg_param=0.01):
        super().__init__()
        self.rgcn = RGCN(num_nodes, h_dim, num_rels * 2)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, nids):
        return self.rgcn(g, nids)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss
