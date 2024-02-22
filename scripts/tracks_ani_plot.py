from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from libx.dataio import Dset, AniTrack
import torch

dset = Dset('./data/set/01_tracks_a.pth')
meta = torch.load('./data/set/01_Meta_2.pth')

def get_Tracks(meta_id,meta,dset):

    item = meta[meta_id]
    track = dset.search_track(item[0],item[1],item[2]).transpose(0,1)

    nei_cid = dset.frame_neighbor(item[0],item[2])[:,0]
    nei_track = dset.search_track(nei_cid.long(),item[1],item[2]).transpose(0,1)

    return torch.concatenate((track.unsqueeze(1),nei_track),dim=1)
        
if __name__=='__main__':
    # Tracks = get_Tracks(12,meta,dset)
    Tracks = dset.set.to_dense()
    Tracks = Tracks[1:]
    anier = AniTrack()

    anier.load(Tracks)
    anier.init(figsize=None)
    anier.ani_play(frames=np.arange(0,len(Tracks)-1),ifline=False,interval=10)

    anier.ani.save('tracks.gif',writer='pillow')