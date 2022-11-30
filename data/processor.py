import argparse
import json
import os
import random

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from transformers.models.clip import CLIPProcessor

# Constructs a CLIP processor which wraps a CLIP feature extractor and a CLIP tokenizer into a single processor.
# 构建一个包含feature extractor 和 tokenizer 的CLIP processor
# 返回的text:[B, 512]
# 返回的image:[B, 512]
aux_size, rcnn_size = 128, 64

clip_processor = CLIPProcessor.from_pretrained(
    "../pretrained_model/clip-vit-base-patch32"
)
aux_processor = CLIPProcessor.from_pretrained(
    "../pretrained_model/clip-vit-base-patch32"
)
aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = (
    aux_size,
    aux_size,
)
rcnn_processor = CLIPProcessor.from_pretrained(
    "../pretrained_model/clip-vit-base-patch32"
)
rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = (
    rcnn_size,
    rcnn_size,
)


class Data_Processor:
    """
    对知识图谱中节点和关系的图像、文本模态进行数据预处理，将其添加到feature变量中。
    """

    def __init__(self, args, tokenizer):
        super().__init__()
        self.task_name = args.task_name
        self.data_dir = args.data_dir
        self.features = self.features_define()
        self.tokenizer = tokenizer

    def features_define(self):
        entities = self.get_entities()
        relations = self.get_relations()
        features = {}
        features.update({ent: {"image": [], "text": [], "type": "entity"} for ent in entities})
        features.update({rel: {"image": [], "text": [], "type": "relation"} for rel in relations})
        return features

    def __call__(self):
        # 将实体相关特征添加进来
        self.features = self.load_ent_image()
        self.features = self.load_ent_text()
        # 添加关系特征（只有文本特征）
        self.features = self.load_rel_text()
        return self.features

    def get_entities(self):
        """
        获取知识图谱中的节点
        """
        entity_path = "".join(self.data_dir + "entities.txt")
        with open(entity_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])
        return entities

    def get_relations(self):
        """
        获取知识图谱中的关系
        """
        with open(
                os.path.join(self.data_dir, "relations.txt"), "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_ent2img(self):
        """
        构建以ent:img_path为存储形式的字典
        """
        # 获取存储实体的列表
        entities = self.get_entities()
        # 获取存放图像模态的前置路径
        img_file_path = {"FB15k-237": "../dataset/FB15k-images/"}[self.task_name]
        # ents 为包含所有实体的并且将在处理路径相关操作时的非法符号进行合法替换后的name
        ents = os.listdir(img_file_path)
        # 构造存储实体路径的字典
        ents2imgs_path = {}
        cnt = 0
        for entity in entities:
            # if self.task_name == "FB15k-237":
            ent_path = entity[1:].replace("/", ".")
            ents2imgs_path[entity] = []
            if ent_path in ents:
                imgs_path_li = os.listdir(img_file_path + ent_path + "/")
                for _ in imgs_path_li:
                    _ = "".join(img_file_path + ent_path + "/" + _)
                    ents2imgs_path[entity].append(_)
            else:
                cnt += 1
        print(
            "There are total {} entities which don't have image modal in KG.".format(
                cnt
            )
        )
        dict_store_path = "".join(self.data_dir + "ent2imgs_path.json")
        # 根据是否存在json文件存储与否
        with open(dict_store_path, "w", encoding="utf-8") as fp:
            json.dump(ents2imgs_path, fp, ensure_ascii=True)
            fp.close()
        return ents2imgs_path

    def load_ent_image(self, img_cnt=6):
        """
        加载并提取实体对应的图像模态的特征
        """
        # 加载或生成以entity：img_path_list形式存储的字典
        dict_store_path = "".join(self.data_dir + "ent2imgs_path.json")
        if os.path.exists(dict_store_path):
            with open(dict_store_path, "r", encoding="utf-8") as fp:
                ents2imgs = json.load(fp)
        else:
            ents2imgs = self.get_ent2img()
        # 定义不处理、mask处理、rcnn处理图像的数组
        pixel_images, aux_images, rcnn_images = [], [], []
        for ent, ent_img_li in tqdm(ents2imgs.items(), desc="Start to get ents' image features"):
            img_li_len = len(ent_img_li)
            img_cnt = min(img_cnt, img_li_len)
            # 随机选取6个图片进行数据增强 有一个不增强
            if img_li_len > 7:
                ent_img_li = random.sample(ent_img_li, k=7)
            ent_full_imgs_path, ent_aux_imgs_path, ent_rcnn_imgs_path = [], [], []
            ent_full_imgs_path = ent_img_li[:1]
            ent_aux_imgs_path = ent_img_li[1:4]
            ent_rcnn_imgs_path = ent_img_li[4:]
            if len(ent_full_imgs_path) > 0:
                full_img = Image.open(ent_full_imgs_path[0]).convert("RGB")
                # 通过CLIP Processor 提取图像特征
                full_img = clip_processor(images=full_img, return_tensors="pt")[
                    "pixel_values"
                ].squeeze()
                pixel_images.append(full_img)
            else:
                pixel_images.append(torch.zeros((3, 224, 224)))
            aux_imgs, rcnn_imgs = [], []
            for i in range(min(3, len(ent_aux_imgs_path))):
                aux_img = Image.open(ent_aux_imgs_path[i]).convert("RGB")
                # aux_img = turn2tensor(aux_img)
                aux_img = aux_processor(images=aux_img, return_tensors="pt")[
                    "pixel_values"
                ].squeeze()
                aux_imgs.append(aux_img)
            for i in range(min(3, len(ent_rcnn_imgs_path))):
                rcnn_img = Image.open(ent_rcnn_imgs_path[i]).convert("RGB")
                # rcnn_img = turn2tensor(rcnn_img)
                rcnn_img = rcnn_processor(images=rcnn_img, return_tensors="pt")[
                    "pixel_values"
                ].squeeze()
                rcnn_imgs.append(rcnn_img)
            # 确保每个实体的图像都是7个，如若前边三种图片不满7 则padding补全个数
            for i in range(3 - len(ent_aux_imgs_path)):
                aux_imgs.append(torch.zeros((3, aux_size, aux_size)))
            for i in range(3 - len(ent_rcnn_imgs_path)):
                rcnn_imgs.append(torch.zeros((3, rcnn_size, rcnn_size)))
            aux_images.append(torch.stack(aux_imgs))
            rcnn_images.append(torch.stack(rcnn_imgs))
            # 更新feature，entity的image
            self.features[ent]["image"] = [pixel_images, aux_images, rcnn_images]
        return self.features

    def get_entity2text(self):
        # 防止出现有的实体没有非结构文本模态而造成key error
        entities = self.get_entities()
        ent2longtext = {entity: [] for entity in entities}
        long_text_path = os.path.join(self.data_dir, "entity2textlong.txt")
        with open(long_text_path, "r", encoding="utf-8") as fp:
            ent2longtext_lines = fp.readlines()
            for line in ent2longtext_lines:
                # 创建 entity：非结构文本模态内容的字典
                temp = line.strip().split("\t")
                if temp[0] in entities:
                    ent2longtext[temp[0]] = temp[1].replace("\\n", " ").replace("\\", "")
        # 如果部分实体不存在非结构文本模态，便将短文本，即该实体在客观世界中的名称作为文本模态训练
        # TODO（可能不如直接加入等量的0向量？后期实验需要尝试）
        short_text_path = os.path.join(self.data_dir, "entity2text.txt")
        with open(short_text_path, "r", encoding="utf-8") as fp:
            ent2text_lines = fp.readlines()
            for line in ent2text_lines:
                temp = line.strip().split("\t")
                if temp[0] in entities and len(ent2longtext[temp[0]]) == 0:
                    ent2longtext[temp[0]] = temp[1].replace("\\n", " ").replace("\\", "")
        # 将entity: text以字典形式进行存储
        with open(
                os.path.join(self.data_dir, "ent2text.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(ent2longtext, fp, ensure_ascii=True)
        return ent2longtext

    def load_ent_text(self):
        """
        加载并提取实体对应的非结构文本模态的特征
        """
        # 加载或生成以entity：img_path_list形式存储的字典
        dict_store_path = "".join(self.data_dir + "ent2text.json")
        if os.path.exists(dict_store_path):
            with open(dict_store_path, "r", encoding="utf-8") as fp:
                ent2longtext = json.load(fp)
        else:
            ent2longtext = self.get_entity2text()
        entities = self.get_entities()
        # 将text -> text_token -> text_token_id
        for ent in tqdm(ent2longtext.keys(), desc="Start to get ents' text features"):
            text = ent2longtext[ent]
            # 分词
            text_token = self.tokenizer.tokenize(text)
            # 加入 [CLS] [SEP]
            text_token = ["[CLS]"] + text_token + ["[SEP]"]
            # 将token转换成vocab.txt中对应的id
            text_id = [self.tokenizer._convert_token_to_id(token) for token in text_token]
            self.features[ent]["text"] = text_id
        return self.features

    def get_relation2text(self):
        """
        通过relation2text.txt文件获取relation的标注与名称的索引
        """
        relations = self.get_relations()
        rel2text_path = os.path.join(self.data_dir, "relation2text.txt")
        # 定义rel2text_dict的存储形式
        rel2text_dict = {rel: [] for rel in relations}
        with open(rel2text_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines:
                temp = line.strip().split("\t")
                if temp[0] in relations:
                    rel2text_dict[temp[0]] = temp[1]
        with open(
                os.path.join(args.data_dir, "rel2text.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(rel2text_dict, fp)
        return rel2text_dict

    def load_rel_text(self):
        """
        对关系的文本模态进行tokenize,因步骤与实体相关过程极度相似，便不在做详细注释。
        """
        rel2text_dict_path = os.path.join(self.data_dir, "rel2text.json")
        if os.path.exists(rel2text_dict_path):
            with open(rel2text_dict_path, "r", encoding="utf-8") as fp:
                rel2text_dict = json.load(fp)
        else:
            rel2text_dict = self.get_relation2text()
        # text -> text_token -> text_id
        for rel in tqdm(rel2text_dict.keys(), desc="Start to get rels' text features"):
            text = rel2text_dict[rel]
            text_token = self.tokenizer.tokenize(text)
            text_token = ["[CLS]"] + text_token + ["[SEP]"]
            text_id = [self.tokenizer._convert_token_to_id(token) for token in text_token]
            self.features[rel]["text"] = text_id
        return self.features


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, default="../dataset/FB15k-237/")
#     parser.add_argument("--task_name", type=str, default="FB15k-237")
#     args = parser.parse_args()
#     tokenizer = BertTokenizer("../pretrained_model/bert-base-uncased/vocab.txt")
#     d = Data_Processor(args, tokenizer)
#     dic = d.get_features()
#     print(dic)
