import os
import sys
sys.path.append(os.getcwd())
from libx.dataio import data_divide
import torch
import pickle

meta = torch.load('./data/set/01_Meta_1.pth')
num = 1
path = './data/index/highD_01_index_'+ str(num) +'_r02_Meta_1'
dd_index = data_divide(len(meta)/num,rate=0.1,shuffle=True)
print(dd_index)
torch.save(dd_index,path+'.pth')