import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dgl_nn
import torch.nn.functional as F
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import f1_score

from GNN_models.model_old import PredModel


def compute_loss(pos_score,neg_score):
    #计算间隔损失
    n_edges = pos_score.shape[0]
    return (1-pos_score.unsqueeze(1) + neg_score.view(n_edges,-1)).clamp(min=0).mean()


def evaluate(g, features, labels, model):
    model.eval()
    with torch.no_grad():
        output = model(g, features)
        pred = np.where(output.data.cpu().numpy() >= 0, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), pred, average='micro')
        return score

def evaluate_in_batches(dataloader, device, model):
    total_score = 0
    for batch_id, batched_graph in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata['feat']
        labels = batched_graph.ndata['label']
        score = evaluate(batched_graph, features, labels, model)
        total_score += score
    return total_score / (batch_id + 1) # return average score


def train(train_dataloader,val_dataloader,device,model):

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer =  torch.optim.Adam(model.parameters(),lr=5e-3,weight_decay=0)

    for epoch in range(500):
        model.train()
        logits = []
        total_loss = 0
        for batch_id,batched_graph in train_dataloader:
            batched_graph =  batched_graph.to(device)
            features = batched_graph.ndata['feature'].float()
            labels = batched_graph.ndata['label'].float()
            logits = model(batched_graph,features)
            loss = loss_fn(logits,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch {:05d} | Loss {:.4f} |".format(epoch, total_loss / (batch_id + 1)))

        if (epoch + 1) % 5 == 0:
            avg_score = evaluate_in_batches(val_dataloader, device, model)  # evaluate F1-score instead of loss
            print("                            Acc. (F1-score) {:.4f} ".format(avg_score))


if __name__=='__main__':
    print(f'Training  with DGL built-in GATConv module.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load and preprocess datasets
    train_dataset = PPIDataset(mode='train')
    val_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    features = train_dataset[0].ndata['feat']

    # create GAT model
    in_size = features.shape[1]
    out_size = train_dataset.num_labels
    model = GAT(in_size, 256, out_size, heads=[4, 4, 6]).to(device)

    # model training
    print('Training...')
    train_dataloader = GraphDataLoader(train_dataset, batch_size=2)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=2)
    train(train_dataloader, val_dataloader, device, model)

    # test the model
    print('Testing...')
    test_dataloader = GraphDataLoader(test_dataset, batch_size=2)
    avg_score = evaluate_in_batches(test_dataloader, device, model)
    print("Test Accuracy (F1-score) {:.4f}".format(avg_score))
