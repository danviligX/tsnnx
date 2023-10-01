import os
import sys
sys.path.append(os.getcwd())
from libx.dataio import data_divide
import torch
import pickle

meta = torch.load('./data/set/01_trainMeta.pth')

dd_index = data_divide(len(meta)/400,rate=0.1)
with open('./data/index/highD_01_index_400_r01.pkl','wb') as path:
    pickle.dump(dd_index,path)
    path.close()