import pandas as pd
import torch
import sys
from tqdm import trange
import numpy as np
from torch.utils.data import Dataset,DataLoader

class Args(object):
    def __init__(self) -> None:
        pass

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
    def __init__(self,path,device) -> None:
        self.set = torch.load(path).to(device=device)

    def frame(self,frameId):
        frame = self.set[frameId].to_dense()
        return frame[frame[:,1]>0]
    
    def frange(self,begin_frame=0,end_frame=125):
        esbf = []
        for i in range(begin_frame,end_frame):
            esbf.append(self.frame(i))
        return esbf
    
    def search_track(self,target_id,begin_frame,end_frame):
        track = []
        for i in range(begin_frame,end_frame):
            if self.set[i][target_id][1] != 0:
                track.append(self.set[i][target_id].to_dense()[1:])
        return torch.stack(track,dim=0)

        
    
class vtp_dataset(Dataset):
    def __init__(self,use_index) -> None:
        super().__init__()
        Meta = torch.load('./data/set/01_trainMeta.pth')
        self.meta_info = Meta[use_index]
    def __getitem__(self, index):
        return self.meta_info[index]
    def __len__(self):
        return len(self.meta_info)
    
def vtp_dataloader(train_item_idx=None,valid_item_idx=None,test_item_idx=None,batch_size=1):
    if train_item_idx is not None:
        train_set = vtp_dataset(train_item_idx)
        valid_set = vtp_dataset(valid_item_idx)
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)
        valid_loader = DataLoader(valid_set,batch_size=1,shuffle=False,num_workers=4,drop_last=True)
        return train_loader,valid_loader
    else:
        test_set = vtp_dataset(test_item_idx)
        test_loader = DataLoader(test_set,batch_size=1,shuffle=False,num_workers=4,drop_last=True)
        return test_loader

def data_divide(index_length,rate=0.1):
    if rate<0 or rate > 1:
        print('Hould out rate error!')
        sys.exit()

    index_length = np.arange(index_length)
    np.random.shuffle(index_length)
    split_num = int(len(index_length)*rate)

    valid_set = index_length[:split_num]
    test_set = index_length[split_num+1:2*split_num]
    train_set = index_length[2*split_num+1:]
    return [train_set,valid_set,test_set]

