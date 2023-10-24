import os
import sys
sys.path.append(os.getcwd())
from libx.dataio import data_divide
import torch
import pickle

meta = torch.load('./data/set/01_trainMeta.pth')
num = 400
path = './data/index/highD_01_index_'+ str(num) +'_r01'
dd_index = data_divide(len(meta)/num,rate=0.1,shuffle=True)

# with open(path+'.pkl','wb') as path:
#     pickle.dump(dd_index,path)
#     path.close()

torch.save(dd_index,path+'.pth')