import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from models.Field_nn.utils import highD,get_data,net

tracks_path = './data/raw/13_tracks.csv'
recordingMeta_path = './data/raw/13_recordingMeta.csv'
tracksMeta_path = './data/raw/13_tracksMeta.csv'

data_track = pd.read_csv(tracks_path)
data_trackMeta = pd.read_csv(tracksMeta_path)
data__recordingMeta = pd.read_csv(recordingMeta_path)

index = data_trackMeta.loc[:]['drivingDirection'] == 1
dir1_data_trackMeta = data_trackMeta[index] #upperLaneMarking

ItemMeta = []
for i in range(len(dir1_data_trackMeta)):
    item = dir1_data_trackMeta.iloc[i]
    if item['numFrames']<250: continue

    vec = np.array([0,0,0,0])
    vec[0] = item['id']
    if item['class'] == 'Car':
        vec[1] = 1
    elif item['class'] == 'Truck':
        vec[2] = 1
    vec[3] = item['numLaneChanges']

    frames = []
    temp = []
    for fid in range(item['initialFrame']+25,item['finalFrame']-25,5):
        temp.append(fid)
        if len(temp) == 40:
            frames.append(temp)
            if vec[3] != 0:
                temp = list(map(lambda x:x+2,temp))
                frames.append(temp)
                temp = list(map(lambda x:x-2,temp))
                frames.append(temp)
            temp = []

    for flist in frames:
        item = np.concatenate((vec,np.array(flist)))
        ItemMeta.append(item)

ItemMeta_np = np.stack(ItemMeta)

index_LaneC = ItemMeta_np[:,3]!=0
index_LaneK = ItemMeta_np[:,3]==0

ItemMeta_LK_np = ItemMeta_np[index_LaneK]
ItemMeta_LC_np = ItemMeta_np[index_LaneC]

ItemMeta_d = np.concatenate((ItemMeta_LK_np[:200],ItemMeta_LC_np))

cls_data = highD()
cls_data.gen_dset()

item = ItemMeta[0]
Ninfo = get_data(item,cls_data)



