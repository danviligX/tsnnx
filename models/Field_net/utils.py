import torch
import torch.nn as nn
import pandas as pd
from libx.dataio import Args

class Dppsx():
    def __init__(self,
                    tracks_path = './data/raw/01_tracks.csv',
                    recordingMeta_path = './data/raw/01_recordingMeta.csv',
                    tracksMeta_path = './data/raw/01_tracksMeta.csv',
                    cache = False,
                    device='cpu'
                ) -> None:
        
        self.tracks_path = tracks_path
        self.recordingMeta_path = recordingMeta_path
        self.tracksMeta_path = tracksMeta_path
        self.device=device
    
    def gen_dset(self):
        data = pd.read_csv(self.tracks_path)
        
        max_frame = data['frame'].max() + 1
        max_id = data['id'].max() + 1

        dset_s = torch.zeros([max_frame,max_id,7])
        for i in range(len(data)):
            t = [data['id'][i],data['x'][i],data['y'][i],data['xVelocity'][i],data['yVelocity'][i],data['xAcceleration'][i],data['yAcceleration'][i]]
            value = torch.tensor(t)
            dset_s[data['frame'][i]][data['id'][i]] = value
        
        self.set = dset_s.to(device=self.device)
    
    def gen_item(self, frameNum=None):
        data = pd.read_csv(self.tracksMeta_path)
        metaItem_1 = []
        metaItem_2 = []

        if frameNum:
            calNum = frameNum - 1

            for i in range(len(data)):
                sf = data.loc[i,'initialFrame']
                ef = data.loc[i,'finalFrame']
                cid = data.loc[i,'id']

                if ef - sf > calNum:
                    for j in range(sf+calNum,ef + 1):
                        item = [cid, j-calNum, j]

                    if data.loc[i,'drivingDirection'] == 1:
                        metaItem_1.append(item)
                    elif data.loc[i,'drivingDirection'] == 2:
                        metaItem_2.append(item)

        elif frameNum is None:
            for i in range(len(data)):
                sf = data.loc[i,'initialFrame']
                ef = data.loc[i,'finalFrame']
                cid = data.loc[i,'id']

                item = [cid, sf, ef]
                if data.loc[i,'drivingDirection'] == 1:
                    metaItem_1.append(item)
                elif data.loc[i,'drivingDirection'] == 2:
                    metaItem_2.append(item)
        
        self.d1_items = metaItem_1
        self.d2_items = metaItem_2
    
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
        for i in range(begin_frame,end_frame+1):
            if torch.norm(self.set[i][target_id]) != 0:
                track.append(self.set[i][target_id].to_dense())
        return torch.stack(track,dim=1)
    
    def get_Tracks(self,meta_id,meta):
        '''
        get the tracks of meta as well as the neighbors at the end frames of the meta
        '''
        item = meta[meta_id]
        track = self.search_track(item[0],item[1],item[2]).transpose(0,1)

        nei_cid = self.frame_neighbor(item[0],item[2])[:,0]
        nei_track = self.search_track(nei_cid.long(),item[1],item[2]).transpose(0,1)

        return torch.concatenate((track.unsqueeze(1),nei_track),dim=1)

    def frame_neighbor(self,center_car,frameId):
        frame = self.frame(frameId=frameId)
        return frame[frame[:,0]!=center_car]
    
    def norm_pos(self):
        idx_data = self.set[self.set[:,:,2]!=0][:,1:3]
        self.mu = idx_data.mean(0)
        self.sig = idx_data.std(0)

        # self.pos = idx_data - mu
        # self.pos = self.pos/sig

class Field_net(nn.Module):
    def __init__(self,basis_num=9) -> None:
        super(Field_net,self).__init__()
        dim_list_nei = [72,24,8,2]
        dim_list_cen = [36,12,4,2]
        dim_list_mlp = [2,4,2]
        self.mlp_nei = self.layer_stack(dim_list = dim_list_nei)
        self.mlp_cen = self.layer_stack(dim_list = dim_list_cen)
        self.mlp = self.layer_stack(dim_list=dim_list_mlp)
        self.mlp2 = self.layer_stack(dim_list=dim_list_mlp)
        self.sinl = nn.Linear(in_features=2,out_features=basis_num)

        self.layernorm = nn.LayerNorm(72)

    def forward(self, cen_pos, nei_pos):
        _cen, _nei = self.pos_argumentation(cen_pos=cen_pos,nei_pos=nei_pos)
        sf_nei = self.mlp_nei(self.layernorm(_nei))
        # mt_cen = self.mlp_cen(_cen)

        sf = (self.mlp(sf_nei) + sf_nei)/2
        # mt = (self.mlp(mt_cen) + mt_cen)/2

        sf = (self.mlp2(sf) + sf)/2
        # mt = (self.mlp2(mt) + mt)/2

        # delta = sf.mean(0) + mt
        delta = sf.mean(0)
        return delta
    
    def layer_stack(self,dim_list):
        layers = []
        for dim_in, dim_out in zip(dim_list[:-2],dim_list[1:-1]):
            layers.append(nn.Linear(in_features=dim_in,out_features=dim_out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=dim_list[-2],out_features=dim_list[-1]))
        return nn.Sequential(*layers)
    
    def pos_argumentation(self,cen_pos,nei_pos):
        rel_pos = nei_pos - cen_pos + 1e-05

        rel_arg = self.mapping(rel_pos)
        nei_arg = self.mapping(nei_pos)

        _nei = torch.concat((nei_arg,rel_arg),dim=1)
        _cen = self.mapping(cen_pos)

        return _cen, _nei

    def mapping(self,data):
        sin = torch.sin(self.sinl(data))
        sqr = data**2
        cub = data**3
        exp = torch.exp(data) 
        crs = data[:,0]*data[:,1]

        inv = 1/(data)
        inv_sin = 1/sin
        inv_sqr = 1/sqr
        inv_cub = 1/cub
        inv_exp = 1/exp
        inv_crs = 1/crs

        data_arg = [data,sin,sqr,cub,exp,crs.unsqueeze(1),inv,inv_sin,inv_sqr,inv_cub,inv_exp,inv_crs.unsqueeze(1)]

        return torch.concat(data_arg,dim=1)

def main():
    args = Args()

    if torch.cuda.is_available():
        args.device = torch.device("cuda:1")
    else:
        args.device = torch.device("cpu")
    
    args.model_name = 'Field_net'

    proser = Dppsx(device=args.device)
    proser.gen_item()
    proser.gen_dset()
    proser.norm_pos()

    args.train_Num = int(len(proser.d1_items)*0.9)
    
    args.batch_size = 20
    args.batch_num = int(args.train_Num/args.batch_size)
    args.lr = 0.001
    args.epoch_num = 200
    args.epoch = 0

    args.msdp = './models/' + args.model_name + '/trial/md_state_' + str(args.epoch) + '_.mdic'
    
    args.opt = 'Adam'

    net = Field_net().to(args.device)
    opt = getattr(torch.optim, args.opt)(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    

    for epoch in range(args.epoch_num):
        net = train(net,args,criterion,proser,opt)
        error = valid(net,args,criterion,proser)
        print('epoch:{},error:{}'.format(epoch,error.item()))
        torch.save(net.state_dict(),args.msdp)
        args.epoch = epoch

def train(net,args,criterion,proser,opt):
    net.train()
    for batch in range(args.batch_num):
        opt.zero_grad()
        for i in range(args.batch_size):
            idx = batch*args.batch_size + i
            item = proser.d1_items[idx]

            if item[2] - item[1] > 40:
                mid_fid = int(sum(item[1:])/2)
                for fid in range(mid_fid-5,mid_fid+5):
                    cen_pos,nei_pos = get_cen_nei(fid,proser,item)
                    cen_pos_in,nei_pos_in = cen_nei_norm(cen_pos,nei_pos,proser)

                    move = net(cen_pos_in,nei_pos_in)
                    cen_pos_pre = cen_pos + move

                    cen_pos_truth,_ = get_cen_nei(fid+1,proser,item)

                    loss = criterion(cen_pos_pre,cen_pos_truth)
                    loss.backward()
            print('epoch:{},batch:{},loss:{}'.format(args.epoch,batch,loss.item()))
        opt.step()
    return net

def valid(net,args,criterion,proser):
    error = []
    net.eval()
    with torch.no_grad():
        for i in range(args.train_Num,len(proser.d1_items)):
            item = proser.d1_items[i]

            if item[2] - item[1] > 40:
                mid_fid = int(sum(item[1:])/2)
                for fid in range(mid_fid-5,mid_fid+5):
                    cen_pos,nei_pos = get_cen_nei(fid,proser,item)
                    cen_pos_in,nei_pos_in = cen_nei_norm(cen_pos,nei_pos,proser)

                    move = net(cen_pos_in,nei_pos_in)
                    cen_pos_pre = cen_pos + move

                    cen_pos_truth,_ = get_cen_nei(fid+1,proser,item)

                    loss = criterion(cen_pos_pre,cen_pos_truth)
                    error.append(loss.item())
    
    return torch.tensor(error).mean()

def get_cen_nei(fid,proser,item):
    nei = proser.frame_neighbor(center_car = item[0],frameId=fid)
    frame = proser.frame(fid)
    cen = frame[frame[:,0]==item[0]]
    cen_pos = cen[:,1:3]
    nei_pos = nei[:,1:3]

    return cen_pos, nei_pos

def cen_nei_norm(cen_pos,nei_pos,proser):
    nei_pos_in = (nei_pos - proser.mu)/proser.sig
    cen_pos_in = (cen_pos-proser.mu)/proser.sig

    return cen_pos_in, nei_pos_in

if __name__ == "__main__": main()