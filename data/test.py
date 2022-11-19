# -*- coding: utf-8 -*-
"""
@Time ： 2022/10/24 10:54
@Auth ： liuchang
@File ：test.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
import torch
import numpy
from PIL import Image

sim_matrix = torch.randn([10,10])
sim_matrix = sum(sim_matrix)
print(sim_matrix)

max_matrix = sorted(enumerate(sim_matrix),key=lambda x:x[1],reverse=True)
print(max_matrix)
print(max_matrix[0][0])

full_img = Image.open("../dataset/wn18-images/n02338145/n02338145_111.JPEG").convert('RGB')
print(full_img)