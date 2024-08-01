import pandas as pd
import torch
import sys
from tqdm import trange
import numpy as np
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt
from matplotlib import animation

class AniTrack():
    def __init__(self) -> None:
        self.Tracks = None
        self.ani = None
        self.p_ani = None

    def clear(self):
        self.Tracks = None
        self.ani = None
        self.p_ani = None

    def load(self,Tracks):
        self.Tracks = Tracks
    
    def init(self, ifline=False, figsize=(20,1.7), gridlim=[0,412.5,0,36]):
        self.fig = plt.figure(figsize=figsize)
        img = plt.imread('./data/raw/01_highway.png')
        plt.imshow(img,extent=gridlim)

        if ifline:
            for i in range(self.Tracks.size(1)):
                x = self.Tracks[:,i,1]
                y = self.Tracks[:,i,2]

                plt_x = x[x!=0]
                plt_y = y[y!=0]

                self.line = plt.plot(plt_x,plt_y,'--','#1f77b4')

        x = self.Tracks[0,:,1]
        y = self.Tracks[0,:,2]

        init_x = x[x!=0]
        init_y = y[y!=0]

        self.p_ani = plt.plot(init_x,init_y,'r.',markersize='3')
        plt.xlim(gridlim[:2])
        plt.ylim(gridlim[2:])
        # plt.grid(ls='--')

    def update_p(self,flid):
        # print(flid)
        x = self.Tracks[flid,:,1]
        y = self.Tracks[flid,:,2]

        fresh_x = x[y!=0]
        fresh_y = y[y!=0]
        self.p_ani[0].set_data(fresh_x, fresh_y)

        return self.p_ani

    def ani_play(self,frames=np.arange(0, 200),interval=30,ifline=True):
        if self.p_ani == None: self.init(ifline=ifline)

        self.ani = animation.FuncAnimation(self.fig, self.update_p,
                                        frames = frames,
                                        interval=interval,
                                        blit=True)
        plt.show()