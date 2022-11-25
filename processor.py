import torch
from PIL import Image
import os 
from torchvision import transforms
import argparse
import json
import random
from transformers.models.clip import CLIPProcessor

# Constructs a CLIP processor which wraps a CLIP feature extractor and a CLIP tokenizer into a single processor.
# 构建一个包含feature extractor 和 tokenizer 的CLIP processor
aux_size, rcnn_size = 128, 64
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
aux_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = aux_size, aux_size
rcnn_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = rcnn_size, rcnn_size


def get_entities(args):
    """
    获取知识图谱中的节点
    """
    entity_path = "".join(args.data_dir + "/entities.txt") 
    with open(entity_path, "r") as f:
        lines = f.readlines()
        entities = []
        for line in lines:
            entities.append(line.strip().split("\t")[0])
    return entities
def make_ent2img_dict(args):
    """
    构建以ent：img_path为存储形式的字典
    """
    # 获取存储实体的列表
    entities = get_entities(args)
    # 获取存放图像模态的前置路径
    img_file_path = {"FB15k-237":"./dataset/FB15k-images/"}[args.task_name]
    #ents 为包含所有实体的并且将在处理路径相关操作时的非法符号进行合法替换后的name
    ents = os.listdir(img_file_path)
    # 构造存储实体路径的字典
    ents2imgs = {}
    cnt = 0
    for entity in entities:
        # if args.task_name == "FB15k-237":
        ent_path = entity[1:].replace("/", ".")
        ents2imgs[entity] = []
        if ent_path in ents:
            imgs_path_li = os.listdir(img_file_path + ent_path + "/")
            for _ in imgs_path_li:
                _ = "".join(img_file_path + ent_path + '/' + _)
                ents2imgs[entity].append(_)
        else:
            cnt += 1
    print("There is total {} entities which don't have image modal in KG.".format(cnt))
    dict_store_path = "".join(args.data_dir + "ents2imgs.json")
    # 根据是否存在json文件存储与否
    if not os.path.exists(dict_store_path):
        with open(dict_store_path, 'w', encoding='utf-8') as fp:
            json.dump(ents2imgs, fp)  
            fp.close()
    else:
        print("{}have existed".format(dict_store_path))
    return ents2imgs
def load_image(args, img_cnt=6):
    """
    加载并提取选取的图像特征
    """
    turn2tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 加载或生成以entity：img_path_list形式存储的字典
    dict_store_path = "".join(args.data_dir + "ents2imgs.json")
    if os.path.exists(dict_store_path):
        with open(dict_store_path, "r", encoding="utf-8") as fp:
            ents2imgs = json.load(fp)
    else:
        ents2imgs = make_ent2img_dict(args)
    # 定义不处理、mask处理、rcnn处理图像的数组
    pixel_images, aux_images, rcnn_images = [], [], []
    for ent, ent_img_li in ents2imgs.items():
        img_li_len = len(ent_img_li)
        img_cnt = min(img_cnt, img_li_len)
        # 随机选取6个图片进行数据增强 有一个不增强
        if img_li_len > 7:
            ent_img_li = random.sample(ent_img_li, k=7)
        ent_full_imgs_path, ent_aux_imgs_path, ent_rcnn_imgs_path = [], [], []
        ent_full_imgs_path = ent_img_li[:1]
        ent_aux_imgs_path = ent_img_li[1: 4]
        ent_rcnn_imgs_path = ent_img_li[4:]
        
        if len(ent_full_imgs_path) > 0:
            full_img = Image.open(ent_full_imgs_path[0]).convert("RGB")
            # full_img = turn2tensor(full_img)
            # 通过CLIP Processor 提取图像特征
            full_img = clip_processor(images=full_img, return_tensors='pt')['pixel_values'].squeeze()
            pixel_images.append(full_img)
        else:
            pixel_images.append(torch.zeros((3, 224, 224)))
        aux_imgs, rcnn_imgs = [], []
        for i in range(min(3, len(ent_aux_imgs_path))):
            aux_img = Image.open(ent_aux_imgs_path[i]).convert('RGB')
            # aux_img = turn2tensor(aux_img)
            aux_img = aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
            aux_imgs.append(aux_img)
        for i in range(min(3, len(ent_rcnn_imgs_path))):
            rcnn_img = Image.open(ent_rcnn_imgs_path[i]).convert('RGB')
            # rcnn_img = turn2tensor(rcnn_img)
            rcnn_img = rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
            rcnn_imgs.append(rcnn_img)
        # 确保每个实体的图像都是7个，如若前边三种图片不满7 则padding
        for i in range(3-len(ent_aux_imgs_path)):
            aux_imgs.append(torch.zeros((3, aux_size, aux_size))) 
        for i in range(3-len(ent_rcnn_imgs_path)):
            rcnn_imgs.append(torch.zeros((3, rcnn_size, rcnn_size)))
        aux_images.append(torch.stack(aux_imgs))
        rcnn_images.append(torch.stack(rcnn_imgs))
    return pixel_images, aux_images, rcnn_images
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset/FB15k-237/")
    parser.add_argument("--task_name", type=str, default="FB15k-237")
    args = parser.parse_args()
    print(load_image(args))
