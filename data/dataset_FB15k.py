import os

import numpy.random
import torch
from torch.utils.data import Dataset


class FB15KDataset(Dataset):
    def __init__(self,
                 text_dir: str,
                 mode: str,
                 image_dir: str,
                 k: int):
        """

        :param text_dir: 文本数据所在地址
        :param mode: train、valid、test
        :param image_dir: 图像数据所在地址
        :param k: 负采样，采样数量
        """
        self.entity_to_text = {}  # k:实体名 v:实体文本描述
        self.text_dir = text_dir  # 实体文本数据dir
        self.image_dir = image_dir  # 实体图像数据dir
        self.relation_to_id = {}  # k:关系名 v:关系id
        self.id_to_relation = {}  # k: id v:关系名

        self.id_to_entity = {}  # k:实体id v:实体名
        self.entity_to_id = {}  # k:实体名 v:实体id
        self.h_r_to_t = {}  # k: 头实体 关系 v:尾实体
        self.t_r_to_h = {}  # k:尾实体 关系 v:头实体
        self.examples = []  # dataset 数据集 {'h':{'entity_name':'/m/04w1j9','/m/04w1j9':实体文本描述,'image_dir':实体图像集地址},'r':{'relation_name':关系名,关系名:关系文本描述},'t':{'entity_name':'/m/04w1j9','/m/04w1j9':实体文本描述,'image_dir':实体图像集地址},'label':0 or 1}

        self.mode = mode  # train valid test
        self.relation_to_text = {}  # k:关系名 v:关系文本 描述
        self.get_entities()
        self.get_realtions()
        self.get_id_to_entity()
        self.get_id_to_relation()
        self.get_entity_to_id()
        self.get_relation_to_id()
        self.get_examples()
        self.creat_neg_examples(k)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):  # 切片
        """
        :param item: 切片索引
        :return: 返回三元组
        """
        return self.examples[item]

    def get_entities(self):
        """
        获取 entity_to_text:dict
        :return: 返回 entity_to_text
        """

        with open(os.path.join(self.text_dir, "entity2textlong.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.replace("\n", "").split("\t")
                self.entity_to_text[tmp[0]] = tmp[1].replace("\\n", "").replace("//", "").replace("\\", "")
        return self.entity_to_text

    def get_examples(self):
        """
        构建输入样例正例数据集
        :return: dataset 数据集 []
        """
        if self.mode in ['train', 'valid', 'test']:
            if self.mode in ['train', 'test']:
                path = os.path.join(self.text_dir, self.mode + ".tsv")
            else:
                path = os.path.join(self.text_dir, "dev.tsv")
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    h = {}
                    r = {}
                    t = {}
                    example = {}
                    tmp = line.replace("\n", "").split("\t")

                    h['entity_name'] = tmp[0]
                    h[tmp[0]] = self.entity_to_text[tmp[0]]
                    h['image_dir'] = self.get_image_dir(tmp[0])
                    r['relation_name'] = tmp[1]
                    r[tmp[1]] = self.relation_to_text[tmp[1]]

                    t['entity_name'] = tmp[2]
                    t[tmp[2]] = self.entity_to_text[tmp[2]]
                    t['image_dir'] = self.get_image_dir(tmp[2])
                    h_r = tmp[0] + " " + tmp[1]
                    if h_r in self.h_r_to_t.keys():
                        self.h_r_to_t[h_r].append(self.entity_to_id[tmp[2]])
                    else:
                        self.h_r_to_t[h_r] = [self.entity_to_id[tmp[2]]]

                    t_r = tmp[2] + " " + tmp[1]
                    if t_r in self.t_r_to_h.keys():
                        self.t_r_to_h[t_r].append(self.entity_to_id[tmp[0]])
                    else:
                        self.t_r_to_h[t_r] = [self.entity_to_id[tmp[0]]]

                    example['h'] = h
                    example['r'] = r
                    example['t'] = t
                    example['label'] = 1
                    self.examples.append(example)
        else:
            raise ValueError("Mode must be train or valid or test")
        return self.examples

    def get_realtions(self):
        """
        获取relation_to_text : dict
        :return: 返回 relation_to_text
        """
        with open(os.path.join(self.text_dir, "relation2text.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.replace("\n", "").split("\t")
                self.relation_to_text[tmp[0]] = tmp[1]
        return self.relation_to_text

    def get_image_dir(self, entity):
        """

        :param entity: 实体名 '/m/04tng0'
        :return: 实体图像数据集地址 ../dataset/FB15K-images/m.04tng0
        """
        tmp = entity[1:].replace('/', '.')
        image_dir = os.path.join(self.image_dir, tmp)
        # print(image_dir)
        return image_dir

    def get_relation_to_id(self):
        """
        获取 relation_to_id:dict
        :return: 返回relation_to_id
        """
        path = os.path.join(self.text_dir, "relations.txt")
        with open(path, "r") as f:
            lines = f.readlines()
            tmp = []
            for line in lines:
                tmp.append(line.replace("\n", ""))
            self.relation_to_id = {relation: index for index, relation in enumerate(tmp)}

        return self.relation_to_id

    def get_entity_to_id(self):
        """
        获取entity_to_id:dict
        :return: 返回entity_to_id
        """
        path = os.path.join(self.text_dir, "entities.txt")
        with open(path, "r") as f:
            lines = f.readlines()
            ent = []
            for line in lines:
                ent.append(line.replace("\n", ""))
            self.entity_to_id = {entity: index for index, entity in enumerate(ent)}
        return self.entity_to_id

    def get_id_to_entity(self):
        """
        获取id_to_entity:dict{}
        :return: 返回id_to_entity
        """
        path = os.path.join(self.text_dir, "entities.txt")
        with open(path, "r") as f:
            lines = f.readlines()
            ent = []
            for line in lines:
                ent.append(line.replace("\n", ""))
            self.id_to_entity = {index: entity for index, entity in enumerate(ent)}

        return self.id_to_entity

    def get_id_to_relation(self):
        """
        获取id_to_relation:dict {}
        :return: 返回id_to_relation
        """
        path = os.path.join(self.text_dir, "relations.txt")
        with open(path, "r") as f:
            lines = f.readlines()
            tmp = []
            for line in lines:
                tmp.append(line.replace("\n", ""))
            self.id_to_relation = {index: relation for index, relation in enumerate(tmp)}
        return self.id_to_relation

    def creat_neg_examples(self, k):
        """

        :param k: 固定h r 创建k个尾实体不在t[]中,但在entity中的实体作为尾实体的负例  以及 固定t r 创建k个尾实体不在h[]中,但在entity中的实体作为尾实体的负例 t为 h-r 对应尾实体列表 h为 r-t对应头实体列表
        :return:
        """
        for h_r, t in self.h_r_to_t.items():
            for i in range(k):

                id_t = numpy.random.randint(low=0, high=len(self.entity_to_id))
                while id_t in t:
                    id_t = numpy.random.randint(low=0, high=len(self.entity_to_id))
                example = {}
                h = {}
                r = {}
                new_t = {}
                tmp = h_r.split(" ")

                h['entity_name'] = tmp[0]
                h[tmp[0]] = self.entity_to_text[tmp[0]]
                h['image_dir'] = self.get_image_dir(tmp[0])

                r['relation_name'] = tmp[1]
                r[tmp[1]] = self.relation_to_text[tmp[1]]

                new_t['entity_name'] = self.id_to_entity[id_t]
                new_t[self.id_to_entity[id_t]] = self.entity_to_text[self.id_to_entity[id_t]]
                new_t['image_dir'] = self.get_image_dir(self.id_to_entity[id_t])

                example['h'] = h
                example['r'] = r
                example['t'] = new_t
                example['label'] = 0
                self.examples.append(example)
        for t_r, h in self.t_r_to_h.items():
            for i in range(k):

                id_h = numpy.random.randint(low=0, high=len(self.entity_to_id))
                while id_h in h:
                    id_h = numpy.random.randint(low=0, high=len(self.entity_to_id))
                example = {}
                t = {}
                r = {}
                new_h = {}
                tmp = t_r.split(" ")

                t['entity_name'] = tmp[0]
                t[tmp[0]] = self.entity_to_text[tmp[0]]
                t['image_dir'] = self.get_image_dir(tmp[0])

                r['relation_name'] = tmp[1]
                r[tmp[1]] = self.relation_to_text[tmp[1]]

                new_h['entity_name'] = self.id_to_entity[id_h]
                new_h[self.id_to_entity[id_h]] = self.entity_to_text[self.id_to_entity[id_h]]
                new_h['image_dir'] = self.get_image_dir(self.id_to_entity[id_h])

                example['h'] = new_h
                example['r'] = r
                example['t'] = t
                example['label'] = 0
                self.examples.append(example)

#
# dataset = FB15KDataset("../dataset/FB15k-237/", "test", "../dataset/FB15k-images/", k=5)
#
# print(dataset.examples)
