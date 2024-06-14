import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import copy

class highD():
    def __init__(self,
                    tracks_path = './data/raw/13_tracks.csv',
                    recordingMeta_path = './data/raw/13_recordingMeta.csv',
                    tracksMeta_path = './data/raw/13_tracksMeta.csv',
                    cache = True,
                    device='cpu'
                ) -> None:
        
        self.tracks_path = tracks_path
        self.recordingMeta_path = recordingMeta_path
        self.tracksMeta_path = tracksMeta_path
        self.device = device
        self.trackMeta = pd.read_csv(self.tracksMeta_path)
        self.track = pd.read_csv(self.tracks_path)
        self.recordingMeta = pd.read_csv(self.recordingMeta_path)
        self.ifcache = cache

        index = self.trackMeta.loc[:]['drivingDirection'] == 1
        self.dir1_vid = list(self.trackMeta['id'][index])
        self.dir1_trackMeta = self.trackMeta[index]
        # self.dir1_ids = self.dir1_trackMeta['id']
        self.norm_value_center = torch.tensor([201.7, 25.5, -3, 0])
        self.norm_value_scaler = torch.tensor([435, 36.5, 85.6, 3.6])
        
    def gen_dset(self, ifnorm=True):
        if self.ifcache: return 0
        data = self.track
        
        max_frame = data['frame'].max() + 1
        max_id = data['id'].max() + 1

        dset_s = torch.zeros([max_frame,max_id,5])
        if ifnorm:
            for i in range(len(data)):
                # if data['id'][i] not in self.dir1_vid: continue
                t = [data['id'][i],data['x'][i],data['y'][i],data['xVelocity'][i],data['yVelocity'][i]]
                value = torch.tensor(t)
                value[1:] = (value[1:] - self.norm_value_center)/self.norm_value_scaler
                dset_s[data['frame'][i]][data['id'][i]] = value
        else:
            for i in range(len(data)):
                # if data['id'][i] not in self.dir1_vid: continue
                t = [data['id'][i],data['x'][i],data['y'][i],data['xVelocity'][i],data['yVelocity'][i]]
                value = torch.tensor(t)
                dset_s[data['frame'][i]][data['id'][i]] = value
        
        self.set = dset_s.to(device=self.device)
    
    def frame(self,frameId):
        frame = self.set[frameId].to_dense()
        return frame[frame[:,0]>0]
    
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

    def gen_ItemMeta(self):
        if self.ifcache: return 0
        ItemMeta = []
        for i in range(len(self.dir1_trackMeta)):
            item = self.dir1_trackMeta.iloc[i]
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

        self.ItenMeta_d = ItemMeta_d

    def get_itemdata(self,item):
        if self.ifcache: return 0
        ego_id = item[0]
        # ego_cls = item[1:3]
        Ninfo = []
        trackMeta = self.trackMeta
        for fid in item[4:]:
            frame = self.frame(fid)
            frame_nei = frame[frame[:,0]!=ego_id]
            P = frame[frame[:,0]==ego_id,1:3]
            V = frame[frame[:,0]==ego_id,3:5]
            Pn = frame_nei[:,1:3]
            Vn = frame_nei[:,3:5]
            Idn = frame_nei[:,0]
            Cn = []
            for id in Idn:
                if trackMeta['class'][trackMeta['id']==id.item()].iloc[0]=='Truck':
                    Cn.append([0,1])
                elif trackMeta['class'][trackMeta['id']==id.item()].iloc[0]=='Car':
                    Cn.append([1,0])
            Cn = torch.tensor(Cn).to(self.device)
            idx = [i.item() in self.dir1_vid for i in Idn]

            Ninfo.append((P,V,Pn[idx],Vn[idx],Cn[idx],Idn[idx]))

        return Ninfo
    
    def gen_data_iform(self):
        if self.ifcache: return 0
        info_input_net = []
        for item_meta in self.ItenMeta_d:
            info = self.get_itemdata(item_meta)
            info_input_net.append(info)
        
        self.input_info = info_input_net
    
    def gen_dataloader(self, batch_size=25, shuffle=True,ratio=[0.8,0.1]):
        if self.ifcache:
            self.train_data = torch.load('./cache/train.dloader')
            self.valid_data = torch.load('./cache/valid.dloader')
            self.test_data = torch.load('./cache/test.dloader')
            return 0
        
        if ratio[0]+ratio[1]>1: raise ValueError("Ratio should be less than 1")
        if shuffle: random.shuffle(self.input_info)
        item_num = len(self.input_info)
        div_num_train = int(item_num*ratio[0])
        div_num_valid = int(item_num*(ratio[0]+ratio[1]))
        n1 = div_num_train%batch_size
        n2 = (div_num_valid+n1)%batch_size

        self.train_data = self.input_info[:div_num_train-n1]
        self.valid_data = self.input_info[div_num_train-n1:div_num_valid-n2]
        self.test_data = self.input_info[div_num_valid-n2:]

        torch.save(self.train_data,'./cache/train.dloader')
        torch.save(self.valid_data,'./cache/valid.dloader')
        torch.save(self.test_data,'./cache/test.dloader')

class ff_net(nn.Module):
    def __init__(self):
        super(ff_net,self).__init__()
        # self.Er_net = self.gen_Er_net()
        self.dt = 0.2
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sfm = nn.Softmax(dim=0)
        self.LaneMark = torch.tensor([13.55,17.45,21.12,24.91]).cuda()
        self.hidden_size = 64
        self.de_hidden = nn.Linear(in_features=self.hidden_size,out_features=2)

        # LSTM Cell
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.cell_l1 = nn.Linear(in_features=2*self.hidden_size,out_features=self.hidden_size)
        self.cell_l2 = nn.Linear(in_features=2*self.hidden_size,out_features=self.hidden_size)
        self.cell_l3 = nn.Linear(in_features=2*self.hidden_size,out_features=self.hidden_size)
        self.cell_l4 = nn.Linear(in_features=2*self.hidden_size,out_features=self.hidden_size)

        # Er part
        Er_basis_num = 32
        self.Er_Linear_sel = nn.Linear(in_features=4,out_features=4)
        # self.Er_Linaer_map = nn.Linear(in_features=32+Er_basis_num*2,out_features=Er_basis_num)
        self.Er_Linaer_map = self.En_mlp = nn.Sequential(*[
            nn.Linear(in_features=Er_basis_num+16,out_features=1024),
            # nn.Linear(in_features=Er_basis_num*2+32,out_features=1024),
            nn.Sigmoid(),
            nn.Linear(in_features=1024,out_features=512),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=512,out_features=128),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=128,out_features=Er_basis_num)
        ])
        self.Er_Linaer_efc = nn.Linear(in_features=Er_basis_num,out_features=self.hidden_size)
        self.aug_sin0 = nn.Linear(in_features=4,out_features=Er_basis_num)
        self.aug_ensin0 = nn.Linear(in_features=Er_basis_num,out_features=Er_basis_num)

        # En part
        En_basis_num = 128
        self.aug_sin1 = nn.Linear(in_features=2,out_features=En_basis_num)
        self.aug_ensin1 = nn.Linear(in_features=En_basis_num,out_features=En_basis_num)
        self.En_mlp = nn.Sequential(*[
            # nn.Linear(in_features=En_basis_num*4+46,out_features=1024),
            nn.Linear(in_features=En_basis_num*2+30,out_features=512),
            # nn.Sigmoid(),
            # nn.Linear(in_features=1024,out_features=1024),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=512,out_features=512)
        ])

        self.En_mlp2 = nn.Sequential(*[
            # nn.Linear(in_features=1024,out_features=1024),
            # nn.Sigmoid(),
            nn.Linear(in_features=1024,out_features=1024),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=1024,out_features=1024)
        ])

        # self.En_mlp3 = nn.Sequential(*[
        #     # nn.Linear(in_features=1024,out_features=1024),
        #     # nn.Sigmoid(),
        #     nn.Linear(in_features=1024,out_features=1024),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(in_features=1024,out_features=1024)
        # ])

        # self.En_mlp4 = nn.Sequential(*[
        #     # nn.Linear(in_features=1024,out_features=1024),
        #     # nn.Sigmoid(),
        #     nn.Linear(in_features=1024,out_features=1024),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(in_features=1024,out_features=1024)
        # ])

        self.En_mlp5 = nn.Sequential(*[
            nn.Linear(in_features=512,out_features=512),
            nn.Sigmoid(),
            nn.Linear(in_features=512,out_features=256),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=256,out_features=self.hidden_size)
        ])

        # self.En_Linear_weight = nn.Linear(in_features=6,out_features=1)
        self.En_Linear_weight = nn.Sequential(*[
            nn.Linear(in_features=self.hidden_size+4,out_features=256),
            nn.Sigmoid(),
            nn.Linear(in_features=256,out_features=128),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=128,out_features=1)
        ])

    def encode(self,frame):
        ego_p, ego_v, Pn, Vn, Cn, Idn = frame
        En_density = self.En_net(Pn,ego_p,Vn,ego_v,Cn)
        Er_density = self.Er_net(ego_p[0,1])
        return Er_density,En_density

    def hidden_update(self,h_r,hm_r):
        hc = torch.concat((h_r,hm_r),dim=0)
        i = self.sig(self.cell_l1(hc))
        f = self.sig(self.cell_l2(hc))
        o = self.sig(self.cell_l3(hc))
        g = self.tanh(self.cell_l4(hc))
        c = f*hm_r + i*g
        h = o*self.tanh(c)

        return h, c
    
    def Er_net(self,ego_y):
        delta_y = torch.tensor(list(map(lambda x:x-ego_y, self.LaneMark))).cuda()
        x = self.sfm(self.Er_Linear_sel(delta_y))*delta_y
        x = torch.concat(self.auge(x))
        x = self.leaky_relu(self.Er_Linaer_map(x))
        x = self.Er_Linaer_efc(x)
        return x

    def En_net(self,Pn,Pego,Vn,Vego,Cn):
        Pego = Pego.repeat(len(Pn),1)
        Vego = Vego.repeat(len(Pn),1)
        dP = Pn - Pego
        dV = Vn - Vego
        dP_aug = torch.concat(self.auge(dP,cls=1),dim=1)
        dV_aug = torch.concat(self.auge(dV,cls=1),dim=1)

        Ninfo = torch.concat((Pego,Pn,dP,dP_aug,Vego,Vn,dV,dV_aug,Cn),dim=1)
        Ei = []
        for nei in Ninfo:
            x = self.En_mlp(nei)
            # x = self.leaky_relu(x + self.En_mlp2(x))
            # x = self.leaky_relu(x + self.En_mlp3(x))
            # x = self.leaky_relu(x + self.En_mlp4(x))
            x = self.En_mlp5(x)
            Ei.append(x)
        
        En = torch.stack(Ei)
        weight = self.sfm(self.En_Linear_weight(torch.concat((En,Pn,dP),dim=1)))
        En_out = sum(weight*En)
        
        return En_out

    def auge(self,data,cls=0):
        if cls==0:
            sin = self.aug_ensin0(torch.sin(self.aug_sin0(data)))
        elif cls==1:
            sin = self.aug_ensin1(torch.sin(self.aug_sin1(data)))
        sqr = data**2
        cub = data**3
        exp = torch.exp(data)

        inv = 1/(data + 1e-6)
        inv_sin = 1/sin
        inv_sqr = 1/sqr
        inv_cub = 1/cub
        inv_exp = 1/exp

        # data_aug = [data,sin,sqr,cub,exp,inv,inv_sin,inv_sqr,inv_cub,inv_exp]
        data_aug = [data,sin,sqr,cub,exp]

        return data_aug
     
    def layer_stack(self,dim_list):
        layers = []
        for dim_in, dim_out in zip(dim_list[:-2],dim_list[1:-1]):
            layers.append(nn.Linear(in_features=dim_in,out_features=dim_out))
            layers.append(self.leaky_relu)
        layers.append(nn.Linear(in_features=dim_list[-2],out_features=dim_list[-1]))
        return nn.Sequential(*layers)
    
    # def forward(self,data_item):
    #     ego_p, ego_v, Pn, Vn, Cn, Idn = data_item

    #     self.Er_density = self.Er_net(ego_p[0,1])
    #     self.En_density = self.En_net(Pn,ego_p,Vn,ego_v,Cn)
    #     out = self.de_hidden(self.Er_density+self.En_density)
    #     return out
    def gen_track(self,item):
        track_pre = []
        for frame in item:
            track_pre.append(torch.concat((frame[0],frame[1]),dim=1))

        Track_pre = torch.concat(track_pre,dim=0)
        return Track_pre

    def phy_cal(self,ori_pos,ori_vel,a):
        vel = ori_vel + a*self.dt
        pos = ori_pos + vel*self.dt
        pos_d = ori_pos + ori_vel*self.dt + a*self.dt**2/2
        return pos,vel,pos_d

    def pre_pv(self,ho_r,ho_n,frame,ori_frame):
        out = self.de_hidden(ho_r+ho_n)
        ori_pos = ori_frame[0]
        ori_vel = ori_frame[1]
        pos,vel,_ = self.phy_cal(ori_pos,ori_vel,out)
        frame[0][0] = pos
        frame[1][0] = vel

    def forward(self,data_item):
        
        history = data_item[:15]
        # print('copy started')
        predict = copy.deepcopy(data_item[15:])
        # print('copy complete')
        # future = data_item[15:]

        for frame in predict:
            frame[0][0] = torch.tensor([0,0])
            frame[1][0] = torch.tensor([0,0])

        gt_track = self.gen_track(data_item[15:])

        cm_r,cm_n = self.encode(history[0])
        for frame in history[1:]:
            hi_r, hi_n = self.encode(frame)

            ho_r,cm_r = self.hidden_update(hi_r,cm_r)
            ho_n,cm_n = self.hidden_update(hi_n,cm_n)
        
        self.pre_pv(ho_r,ho_n,predict[0],history[-1])
        
        for idx in range(1,len(predict)):
            ori_frame = predict[idx-1]
            frame = predict[idx]
            hi_r, hi_n = self.encode(ori_frame)
            ho_r,cm_r = self.hidden_update(hi_r,cm_r)
            ho_n,cm_n = self.hidden_update(hi_n,cm_n)

            self.pre_pv(ho_r,ho_n,frame,ori_frame)
        
        pre_track = self.gen_track(predict)


        return pre_track,gt_track