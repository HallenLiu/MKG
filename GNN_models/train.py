import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dgl_nn
import torch.nn.functional as F

from GNN_models.model_old import PredModel


def compute_loss(pos_score,neg_score):
    #计算间隔损失
    n_edges = pos_score.shape[0]
    return (1-pos_score.unsqueeze(1) + neg_score.view(n_edges,-1)).clamp(min=0).mean()


graph_list, _ = dgl.load_graphs('../dataset/FB15k-237/FB15k-237_labeled.bin')
graph = graph_list[0]

train_eid_dict = {
    graph.edges(etype=etype,form='eid')
    for etype in graph.canonical_etypes
}

sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
train_dataloader = dgl.dataloading.EdgeDataLoader(
    graph,
    train_eid_dict,
    sampler,
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
    batch_size=1024,
    shuffle=True,
    drop_last=False,
)

model = PredModel(in_features,hid_features,out_features,len(graph.ntypes),len(graph.canonical_etypes))
model = model.cuda()

opt = torch.optim.Adam(model.parameters())

def train(epoches):
    model.train()
    for epoch in epoches:
        for input_nodes,positive_graph,negative_graph,blocks in train_dataloader:
            positive_graph = positive_graph.to(torch.device('cuda'))
            negative_graph = negative_graph.to(torch.device('cuda'))
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            input_features = blocks[0].srcdata['features']
            pos_score,neg_score = model(positive_graph,negative_graph,blocks,input_features)
            loss = compute_loss(pos_score,neg_score)
            opt.zero_grad()
            loss.backwar()
            opt.step()
            print(loss)


def valid():
    model.eval()

