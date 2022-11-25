import argparse
import json
import os
import shutil
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm



class FilterGate:
    def __init__(self, args):
        self.base_path = args.base_path
        self.best_imgs = {}

    def extract_features(self, img):
        img = Image.open(img)
        inputs = processor(images=img, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features

    def cosine_sim(self, img1, img2):
        img1_feature = self.extract_features(img1)
        img2_feature = self.extract_features(img2)
        sim = torch.cosine_similarity(img1_feature, img2_feature, dim=1)
        return sim.float()

    def filter_topn_imgs(self, img_num):
        self.best_imgs = {}
        ents = os.listdir(self.base_path)
        pbar = tqdm(total=len(ents))
        while len(ents) > 0:
            ent = ents.pop()
            if "DS_" in ent:
                continue
            imgs = os.listdir(self.base_path + ent + "/")
            n_img = len(imgs)
            if n_img == 0:
                pbar.update(1)
                continue
            sim_matrix = [[0] * n_img for i in range(n_img)]
            for i in range(n_img):
                for j in range(i + 1, n_img):
                    img1 = self.base_path + ent + "/" + imgs[i]
                    img2 = self.base_path + ent + "/" + imgs[j]
                    sim = self.cosine_sim(img1, img2)
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim
            max_index = 0
            max_sim = sum(sim_matrix[0])
            for i in range(1, n_img):
                if sum(sim_matrix[i]) > max_sim:
                    max_index = i
                    max_sim = sum(sim_matrix[i])
            img_li = sim_matrix[max_index]
            best_img_li = sorted(img_li, reverse=True)
            index_li = [max_index]
            for i in range(min(img_num, len(img_li))):
                index = img_li.index(best_img_li[i])
                if index not in index_li:
                    index_li.append(index)
                    img_li[index] = -1
            self.best_imgs[ent] = [
                "./dataset/FB15k-images/" + ent + "/" + imgs[index]
                for index in index_li
            ]
            pbar.update(1)
        pbar.close()
        return self.best_imgs

    def save_best_imgs(self, output_file, n=1):
        with open(output_file, "wb") as out:
            pickle.dump(self.best_imgs, out)

    def create_topn_imgs_dataset(self, output_file):
        entity_img_path = {
            "wn18": "../dataset/FB15k-237/wn18_best_imgs/",
            "fb15k-237": "../dataset/FB15k-237/FB15k_best_imgs/",
        }[self.task_name]
        file = open(output_file, "r")
        fb15k_best_img = json.load(file)
        file.close()
        img_paths = list(fb15k_best_img.values())
        path = Path(entity_img_path)
        if not path.exists():
            os.mkdir(path)
        for imgs_paths in tqdm(img_paths):
            ent = imgs_paths[0].split("/")[3]
            ent_img_path = entity_img_path + ent
            if not Path(ent_img_path).exists():
                os.mkdir(Path(ent_img_path))
            for img_path in imgs_paths:
                shutil.copy(img_path, ent_img_path)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("../pretrained_model/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("../pretrained_model/clip-vit-base-patch32")
    model.to(device)
    parser = argparse.ArgumentParser(add_help=False)
    # 定义基本超参数
    parser.add_argument("--task_name", type=str, default="fb15k-237")
    # parser.add_argument("--hash_size", type=int, default=16)
    parser.add_argument("--base_path", type=str, default="../dataset/FB15k-images/")
    args = parser.parse_args()
    # 输出存储相似度topn的json文件
    output_file = "../dataset/FB15k-237/fb15k237_clip_best_img.json"
    f = FilterGate(args)
    if not Path(output_file).exists():
        f.filter_topn_imgs(img_num=5)
        f.save_best_imgs(output_file)
    else:
        print("{} exists".format(output_file))
    f.create_topn_imgs_dataset(output_file)