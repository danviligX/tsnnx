import pandas as pd
import torch
from tqdm import trange

def track2set(path='data/raw/01_tracks.csv'):
    '''
    require:
        raw data path: track
    out:
        data set: [frame id][car id][car id, x, y, xVelocity, yVelocity]
    '''
    data = pd.read_csv(path)

    max_frame = data['frame'].max() + 1
    max_id = data['id'].max() + 1

    set = torch.zeros([max_frame,max_id,6])
    for i in trange(len(data)):
        # Setting properties
        t = [data['id'][i],data['x'][i],data['y'][i],data['xVelocity'][i],data['yVelocity'][i],data['laneId'][i]]
        value = torch.tensor(t)
        set[data['frame'][i]][data['id'][i]] = value

    return set.to_sparse()

def meta2meta(path='data/raw/01_tracksMeta.csv', frameNum=200):
    '''
    require:
        raw data path: trackMeta
    out:
        train item meta: [car id, start frame, end frame]
    '''
    data = pd.read_csv(path)
    calNum = frameNum - 2
    metaItem = []

    for i in trange(len(data)):
        sf = data.loc[i,'initialFrame']
        ef = data.loc[i,'finalFrame']
        cid = data.loc[i,'id']

        if ef - sf >= calNum:
            for j in range(sf+calNum,ef + 1):
                item = [cid, j-calNum, j]
                metaItem.append(item)

    return torch.tensor(metaItem)

class Dset(object):
    def __init__(self,path) -> None:
        self.set = torch.load(path)

    def showframe(self,frameId):
        frame = self.set[frameId].to_dense()
        return frame[frame[:,1]>0]