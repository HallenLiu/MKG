import json
import os
import dgl
import torch

#
from torch import unsqueeze

graph_list, _ = dgl.load_graphs('../dataset/FB15k-237/FB15k-237.bin')
graph = graph_list[0]

# print(torch.zeros(4,5).scatter_(1,torch.tensor([1,3,2,4]).unsqueeze(1),1))
def get_relation_label_dict(data_dir):
    """
    get relation_labels_dict
    Args:
        data_dir:dataset_dir

    Returns:dict{realtion:label}

    """
    relation_label_dict = {}
    with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            relation_label_dict[tmp[0]] = tmp[1]
        return relation_label_dict


def get_entity_label_dict(data_dir):
    """
    get entity_labels_dict
    Args:
        data_dir:dataset_dir

    Returns:dict{entity:label}

    """
    entity_label_dict = {}
    with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            entity_label_dict[tmp[0]] = tmp[1]
        return entity_label_dict


def get_entity(data_dir):
    """
       get entity
       Args:
           data_dir:dataset_dir

       Returns:list[entity]

       """
    entities = []
    with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            entities.append(tmp[0])
        return entities


def get_relation(data_dir):
    """
       get relation
       Args:
           data_dir:dataset_dir

       Returns:list[relations]

       """
    relations = []
    with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split("\t")
            relations.append(tmp[0])
        return relations


entities = get_entity('../dataset/FB15k-237/')
entity_to_id = {entity: id for id, entity in enumerate(entities)}
id_to_entity = {id: entity for id,entity in enumerate(entities)}
# print(entity_to_id)
relations =get_relation('../dataset/FB15k-237/')
relations_to_id = {relation: id for id, relation in enumerate(relations)}
id_to_relations = {id: relation for id, relation in enumerate(relations)}
train_node_id = {} #id 保存在label中
train_edge_id = {}
ntypes = graph.ntypes
for ntype in ntypes:
    train_node_id[ntype] = 0
    #graph.nodes[ntype].data['label'] =  torch.tensor([entity_to_id[ntype]])#torch.zeros(graph.num_nodes(ntype), len(entity_to_id)).scatter_(1, torch.tensor([entity_to_id[ntype]]).unsqueeze(1), 1)
for canonical_etype in graph.canonical_etypes:
    _,etype,_ = canonical_etype
    train_edge_id[canonical_etype] = 0
    #graph.edges[canonical_etype].data['label'] =torch.tensor([relations_to_id[etype]]) #torch.zeros(graph.num_edges(canonical_etype), len(relations_to_id)).scatter_(1, torch.tensor([relations_to_id[etype]]).unsqueeze(1),1)

# print(graph)

# dgl.save_graphs('../dataset/FB15k-237/FB15k-237_labeled_ONE_HOT.bin',graph)
with open('../dataset/FB15k-237/train_node_id_dict.json','w+') as f:
    json.dump(str(train_node_id),f)

with open('../dataset/FB15k-237/train_edge_id_dict.json','w+') as f:
    json.dump(str(train_edge_id),f)