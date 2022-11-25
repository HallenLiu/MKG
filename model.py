"""RGCN layer implementation"""
from collections import defaultdict

import torch
import torch as th
import torch.nn as nn
import tqdm

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F



class RelGraphConvTwoLayer(nn.Module): #两层之后需要加上邻居采样
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_feat,

        out_feat,
        rel_names,
        num_bases,
        *,
        hid_feat=100,
        weight=True,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0
    ):
        super(RelGraphConvTwoLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feat, hid_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    hid_feat, out_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names)
                )
            else:
                self.weight = nn.Parameter(
                    th.Tensor(len(self.rel_names), in_feat, out_feat)
                )
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain("relu")
                )

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                self.rel_names[i]: {"weight": w.squeeze(0)}
                for i, w in enumerate(th.split(weight, 1, dim=0))
            }
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {
                k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
            }
        else:
            inputs_src = inputs_dst = inputs

        h = F.relu(self.conv1(g, inputs, mod_kwargs=wdict))
        h = self.dropout(h)
        hs = self.conv2(g,h,mod_kwargs=wdict)



        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        hs = {ntype: _apply(ntype, h) for ntype, h in hs.items()} #返回节点的embedding以及h为当前节点的
        return hs


class RelGAT(nn.Module):
    r"""
    R-GAT
    """

    def __init__(self,
                 in_feat,
                 hid_feat,
                 out_feat,
                 rel_names,
                 heads
                 ):
        """

        :param in_feat: int
        :param hid_feat: int
        :param out_feat: int
        :param num_classes: int
        :param rel_names: [etype]
        """
        super(RelGAT, self).__init__()
        self.get_layers = nn.ModuleList()
        #three-layer gat
        self.get_layers.append(
            dglnn.HeteroGraphConv(
                {rel:dglnn.GATConv(in_feats=in_feat,
                                   out_feats=hid_feat,
                                   num_heads=heads[0],
                                   activation=F.elu())
                 for rel in rel_names
                 }
            )
        )
        self.get_layers.append(
            dglnn.HeteroGraphConv(
                {rel:dglnn.GATConv(in_feats=hid_feat*heads[0],
                                   out_feats=hid_feat,
                                   num_heads=heads[0],
                                   activation=F.elu())
                 for rel in rel_names
                 }
            )
        )
        self.get_layers.append(
            dglnn.HeteroGraphConv(
                {rel:dglnn.GATConv(in_feats=hid_feat*heads[1],
                                   out_feats=out_feat,
                                   num_heads=heads[0],
                                   activation=F.elu())
                 for rel in rel_names
                 }
            )
        )
    def forward(self,g,inputs):
        """

        :param g: 输入图
        :param inputs: 输入节点特征
        :return: 最后得节点embedding
        """
        h = inputs
        for i,layer in enumerate(self.get_layers):
            h = layer(g,h)
            if i == 2:
                h = h.mean()
            else:
                h = h.flatten(1)
        g.ndata['h'] = h
        return h


class RelLinkPredictor(nn.Module):
    def __init__(self, in_feat,num_classes=237):
        super(RelLinkPredictor, self).__init__()
        self.W = nn.Linear(2 * in_feat, num_classes)

    def apply_edges(self, edges):
        data = torch.cat(edges.src['h'], edges.dst['h'])
        return {'score': F.relu(self.W(data))}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            for etype in g.canonical_etypes:
                g.apply_edges(self.apply_edges, etype=etype)
            return g.edata['score']


class LinkScoreModel(nn.Module):
    r"""
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 hid_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 weight,
                 bias,
                 activation,
                 self_loop,
                 dropout):
        super(LinkScoreModel, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.hid_feat = hid_feat
        self.rgcn = RelGraphConvTwoLayer(in_feat=in_feat,
                                         out_feat=out_feat,
                                         rel_names=rel_names,
                                         hid_feat=hid_feat,
                                         bias=bias,
                                         activation=activation,
                                         self_loop=self_loop,
                                         dropout=dropout,
                                         weight=weight)
        self.pred = RelLinkPredictor(in_feat=in_feat,num_classes=len(rel_names))

    def forward(self,g,inputs):
        h = self.rgcn(g,inputs)
        score = self.pred(g,h)

        return {'NodeEmbeddings':h,'LinkEmbedding':score}


