# -*- coding: UTF-8 -*-
'''
Image encoder.  Get visual embeddings of images.
'''
import os
import imagehash
import numpy
from PIL import Image
import pickle

from tqdm import tqdm
class FilterGate():
    def __init__(self,base_path,hash_size):
        self.base_path=base_path
        self.hash_size=hash_size
        self.best_imgs={}

    def phash_sim(self,img1,img2,hash_size=None):
        if not hash_size:
            hash_size=self.hash_size
        img1_hash = imagehash.phash(Image.open(img1), hash_size=hash_size)
        img2_hash = imagehash.phash(Image.open(img2), hash_size=hash_size)

        return 1 - (img1_hash - img2_hash) / len(img1_hash) ** 2


    def filter(self):
        self.best_imgs={}
        ents = os.listdir(self.base_path)
        pbar = tqdm(total=len(ents))
        # print(ents)
        while len(ents)>0:
            ent=ents.pop()
            if 'DS_' in ent:
                continue
            # print(self.base_path+ent+"/")
            imgs=os.listdir(self.base_path + ent + '/')
            # print(imgs)
            n_img=len(imgs)
            # print(n_img)
            if n_img == 0:
                pbar.update(1)
                continue
            sim_matrix=[[0]*n_img for i in range(n_img)]
            for i in range(n_img):
                for j in range(i+1,n_img):
                    sim=self.phash_sim(self.base_path + ent + '/'+imgs[i], self.base_path + ent + '/'+imgs[j])
                    # print(sim)
                    sim_matrix[i][j]=sim
                    sim_matrix[j][i] =sim
            # sim_matrix=[sum(i) for i in sim_matrix]
            # max_index=0
            # print(sim_matrix)
            max_matrix = sorted(enumerate(sim_matrix), key=lambda x: x[1], reverse=True)
            # print(max_matrix)
            self.best_imgs[ent] = []
            if len(max_matrix) >= 6:
                for index in range(6):
                    index_of_img = max_matrix[index][0]
                    self.best_imgs[ent].append(self.base_path + ent + '/'+imgs[index_of_img]) #保存6张图片
            else:
                for matrix in max_matrix:
                    index_of_img = matrix[0]
                    # print(imgs[index_of_img])
                    self.best_imgs[ent].append(self.base_path + ent + '/'+imgs[index_of_img]) #保存6张图片

            pbar.update(1)
        pbar.close()
        # print(self.best_imgs)
        return self.best_imgs

    def save_best_imgs(self,output_file,n=1):
        with open(output_file, 'wb') as out:
            pickle.dump(self.best_imgs, out)




if __name__ == '__main__':
    f=FilterGate('/home/liuchang/MKG/dataset/FB15k-images/',hash_size=16)
    f.filter()
    if not os.path.exists('/home/liuchang/MKG/data'):
        os.mkdir('/home/liuchang/MKG/data/')
    if not os.path.exists('/home/liuchang/MKG/data/FB15k-pickle'):
        os.mkdir('/home/liuchang/MKG/data/FB15k-pickle/')
    f.save_best_imgs('/home/liuchang/MKG/data/FB15k-pickle/FB15k_best_img.pickle')