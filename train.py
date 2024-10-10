

import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from libx.fx import multi_proc
from dataclasses import dataclass

@dataclass
class highDConfig:
    f:int=5             # preprocessing frame rate
    input_time:int=3    # input time length
    iter_step:int=1     # iteration step
    dt:float=25/f*0.04  # time step

    ppc_cache:str='./data/processed/change_f5.pth'  # preprocessed data cache
    raw_folder:str='./data/raw/'                    # raw data folder


class highD:
    def __init__(self, config:highDConfig):
        self.config = config
        self.input_len = config.f * config.input_time
        assert type(self.input_len)==int, f'input frames number should be "int" type'
        self.iter_step = config.iter_step
        self.item_len = self.input_len + self.iter_step

        # load basic data
        self.load_files()
        
        # fileter out data with numLaneChanges>0, and split into train and test set
        id_set = self.trackMeta.loc[self.trackMeta['numLaneChanges']>0,'id'].values
        train_set = id_set[:-int(len(id_set)*0.1)]
        self.train_set = train_set
        self.test_set = id_set[-int(len(id_set)*0.1):]

        if os.path.exists(config.ppc_cache):
            self.ppc_data = torch.load(config.ppc_cache, weights_only=True)
            self.ppc_nei = torch.load(config.ppc_cache.replace('.pth','_nei.pth'), weights_only=True)
            print(f'load ppc data from {config.ppc_cache}')
            print(f'load ppc nei data from {config.ppc_cache.replace(".pth","_nei.pth")}')
        else:
            self.preprocess()
            print(f'create ppc data and save to {config.ppc_cache}')

        
    def load_files(self,
                    tracks_path = './data/raw/13_tracks.csv',
                    tracksMeta_path = './data/raw/13_tracksMeta.csv',
                    recordingMeta_path = './data/raw/13_recordingMeta.csv'                   
                   ):
        
        self.track = pd.read_csv(tracks_path)
        self.trackMeta = pd.read_csv(tracksMeta_path)
        self.recordingMeta = pd.read_csv(recordingMeta_path)

        self.upperLaneMarkings = np.array([float(x) for x in self.recordingMeta['upperLaneMarkings'][0].split(';')]) # direction = 1, [13.55, 17.45, 21.12, 24.91]
        self.lowerLaneMarkings = np.array([float(x) for x in self.recordingMeta['lowerLaneMarkings'][0].split(';')]) # direction = 2, [30.53, 34.43, 38.10, 42.11]

        indexes = [0,1,2,3,6,7,8,9,16,17,18,19,20,21,22,23,24]
        self.used_kw = [self.track.keys()[index] for index in indexes]
        self.data_fr = self.recordingMeta['frameRate'].values.item()
    
    def process_item(self, id:int):
            track_array = self.track.loc[self.track['id']==id, self.used_kw].values
            len_track_array = len(track_array)
            if len_track_array<self.div_frame: return None
            track_array = self.get_track(id=id, track_array=track_array)

            # split into item_len length
            _, n = track_array.shape
            item = np.lib.stride_tricks.sliding_window_view(track_array, (self.item_len, n)).squeeze()
            neighbors = self.get_neighbors(item)
            return item, neighbors
    
    def preprocess(self):
        assert self.data_fr%self.config.f==0, "make sure data frame rate is divisible by input frame rate"
        sample_scale = self.data_fr//self.config.f
        self.div_frame = self.data_fr * self.config.input_time + self.iter_step * sample_scale
     
        data_set = []
        nei_set = []
        
        
        res = multi_proc(self.process_item, self.train_set, core=64)

        # for id in tqdm(self.train_set):
        #     item = self.process_item(id)
        #     if item is None: continue    # skip track with length less than input_len
        #     neighbors = self.get_neighbors(item)
        #     # border = self.get_border(item)
        #     data_set.append(item)
        #     nei_set.append(neighbors)
        # data_set = [item[0] for item in res if item is not None]
        # nei_set = [item[1] for item in data_set if item is not None]
        for item in res:
            data_item, nei_item = item
            if data_item is not None:
                data_set.append(data_item)
            else:
                data_set.append(torch.zeros_like(data_set[0]))

            if nei_item is not None:
                nei_set.append(nei_item)
            else:
                nei_set.append(torch.zeros_like(nei_set[0]))

        nei_set = np.concatenate(nei_set,axis=0)
        data_set = np.concatenate(data_set,axis=0)
        
        self.ppc_data = torch.tensor(data_set).to(torch.float32)
        self.ppc_nei = torch.tensor(nei_set).to(torch.float32)
        torch.save(self.ppc_data, self.config.ppc_cache)
        torch.save(self.ppc_nei, self.config.ppc_cache.replace('.pth','_nei.pth'))
    
    def get_track(self, id:int, track_array=None):
        if track_array is None:
            track_array = self.track.loc[self.track['id']==id, self.used_kw].values
        
        sample_scale = self.data_fr//self.config.f
        track_array = track_array[::sample_scale]
        # rotation and translation, set origin to (x[0], bond_center)
        if track_array[0,3] > self.upperLaneMarkings[-1]: 
            # direction = 2
            track_array[:,2:8] = -track_array[:,2:8]
            track_array[:,3] = track_array[:,3] + self.lowerLaneMarkings[-1]
        else:
            # direction = 1
            track_array[:,3] = track_array[:,3] - self.upperLaneMarkings[0]
        track_array[:,2] = track_array[:,2] - track_array[0,2] # x = x - x[0]

        return track_array
    
    def get_neighbors(self, ego):
        nei_ids = ego[:,-1,8:-1]
        f_ids = ego[:,-1,0]

        # search for neighboring features in the track data, and pad with zeros if necessary to make it 8xnum_features
        batched_nei = []
        for ids, f in zip(nei_ids, f_ids):
            if os.path.exists(self.config.ppc_cache) is False:
                temp = self.track.loc[(self.track['id'].isin(ids))&(self.track['frame']==f),self.used_kw].values
            else:
                temp = self.track.loc[(self.track['id'].isin(ids.numpy()))&(self.track['frame']==f.numpy()),self.used_kw].values
            if len(temp)<8: temp = np.pad(temp, ((0,8-len(temp)),(0,0)), 'constant', constant_values=0)
            batched_nei.append(temp)

        batched_nei = torch.Tensor(np.array(batched_nei))
        return batched_nei

    def get_border(self, ego):
        car_id = ego[0,-1,1].int().item()
        direction = self.trackMeta.loc[self.trackMeta['id']==car_id,'drivingDirection'].values
        if direction == 1:
            return torch.Tensor(self.upperLaneMarkings)
        else:
            return torch.Tensor(self.lowerLaneMarkings)


import time
start_time = time.time()
config = highDConfig()
data = highD(config)
print("--- %s seconds ---" % (time.time() - start_time))