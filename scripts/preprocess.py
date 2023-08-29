import os
import sys
sys.path.append(os.getcwd())
from libx.dataio import data_divide
import torch
import pickle

meta = torch.load('./data/set/01_trainMeta.pth')

dd_index = data_divide(len(meta),rate=0.1)
with open('./cache/dd_index.pkl','wb') as path:
    pickle.dump(dd_index,path)
    path.close()