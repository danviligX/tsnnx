import torch.nn as nn
import torch

class Fmap(nn.Module):
    '''
    Fourier series for function: $R^n \to R$
    '''
    def __init__(self,in_dim,basis_dim=7) -> None:
        super(Fmap,self).__init__()
        self.L_s = nn.Linear(in_features=in_dim,out_features=basis_dim)
        self.L_m = nn.Linear(in_features=2*basis_dim,out_features=1)
    def forward(self,input):
        x = self.L_s(input)
        return self.L_m(torch.concat((torch.sin(x),torch.cos(x)),dim=-1))

class Flinear(nn.Module):
    '''
    Fourier series for function: $R^n \to R^m$
    '''
    def __init__(self,in_dim,out_dim,basis_dim=7) -> None:
        super(Flinear,self).__init__()
        self.out_dim = out_dim
        self.Fls = nn.ModuleList([Fmap(in_dim=in_dim,basis_dim=basis_dim)]*out_dim)
    def forward(self,input):
        out = []
        for idx, Fl in enumerate(self.Fls):
            out.append(Fl(input))
        out = torch.concat(out,dim=-1)
        return out

class MFLP(nn.Module):
    '''
    High order Fourier series for function: $R^n \to R^m$
    '''
    def __init__(self,dim_list,basis_list,ifrelu=False) -> None:
        super(MFLP,self).__init__()
        self.mflp = self.layer_stack(dim_list=dim_list, basis_list=basis_list, ifrelu=ifrelu)

    def forward(self,input): return self.mflp(input)
    
    def layer_stack(self,dim_list,basis_list,ifrelu=False):
        layers = []
        for dim_in, dim_out, basis_dim in zip(dim_list[:-2],dim_list[1:-1],basis_list[:-1]):
            layers.append(Flinear(in_dim=dim_in,out_dim=dim_out,basis_dim=basis_dim))
            if ifrelu: layers.append(nn.ReLU())
        layers.append(Flinear(in_dim=dim_list[-2],out_dim=dim_list[-1],basis_dim=basis_list[-1]))
        return nn.Sequential(*layers)
    
class net_motivation(nn.Module):
    def __init__(self, args) -> None:
        super(net_motivation,self).__init__()
        self.embadding = nn.Linear(in_features=2,out_features=args.embadding_size)
        self.mflp = MFLP(dim_list=[args.his_len,1],basis_list=args.basis_list)
        self.deembadding = nn.Linear(in_features=args.embadding_size,out_features=2)

    def forward(self, history):
        seq = self.embadding(history)
        seq = seq.transpose(0,1)

        seq = self.mflp(seq)
        seq = seq.transpose(0,1)

        out = self.deembadding(seq)
        return out

class net_socialforce(nn.Module):
    def __init__(self, args) -> None:
        super(net_socialforce,self).__init__()
        self.embadding = nn.Linear(in_features=2,out_features=args.embadding_size)
        self.mflp = MFLP(dim_list=[2*args.embadding_size,args.embadding_size],basis_list=args.basis_list)
        self.deembadding = nn.Linear(in_features=args.embadding_size,out_features=2)

        self.relu = nn.ReLU()
        self.zero_mflp = MFLP(dim_list=[args.embadding_size,2],basis_list=args.basis_list)
    
    def forward(self, last_position, frame):
        pos_embad = self.embadding(last_position)
        nei_embad = self.embadding(frame[:,:2])

        cn_embad = torch.concat((pos_embad.repeat([len(frame),1]),nei_embad),dim=-1)

        out = self.deembadding(self.mflp(cn_embad)) + self.relu(self.zero_mflp(nei_embad))

        return sum(out)

class net_PhyNet(nn.Module):
    def __init__(self, args) -> None:
        super(net_PhyNet,self).__init__()
        self.tiem_step = args.time_step
        self.pre_len = args.pre_len
        self.his_len = args.his_len
        self.net_mv = net_motivation(args=args)
        self.net_sf = net_socialforce(args=args)

        self.net_mv_v = net_motivation(args=args)
        self.net_sf_v = net_socialforce(args=args)

        self.p2v = Flinear(in_dim=2,out_dim=2,basis_dim=128)
        

    def forward(self, history, meta, dset):
        pre_series = torch.zeros(self.pre_len,4).to(next(self.parameters()).device)
        center_car = meta[0]
        bf = meta[1]
        ef = meta[2]

        for idx in range(self.pre_len):
            if idx<self.his_len:
                his_info = torch.concat((history[idx-self.his_len:,:],pre_series[:idx]))
            else:
                his_info = pre_series[-self.his_len:]

            frame_info = dset.frame_neighbor(center_car=center_car,frameId=bf+idx)[:,1:]

            ninfo = self.next_poi(history=his_info,frame=frame_info[:,:])
            pre_series[idx,:] = ninfo + pre_series[idx,:]

        return pre_series
    
    def next_poi(self,history,frame):
        a_mv = self.net_mv(torch.clone(history[:,:2]))
        a_sf = self.net_sf(torch.clone(history[-1,:2]), torch.clone(frame))
        a = a_mv + a_sf

        a_mv_v = self.net_mv_v(torch.clone(history[:,:2]))
        a_sf_v = self.net_sf_v(torch.clone(history[-1,:2]), torch.clone(frame))
        a_v = a_mv_v + a_sf_v
        
        position = a/2*self.tiem_step**2 + history[-1,2:]*self.tiem_step + history[-1,:2]
        a = self.p2v(a)
        velocity = self.tiem_step*a_v + history[-1,2:]
        return torch.concat((position,velocity),dim=-1)

class sp_model(nn.Module):
    def __init__(self,args) -> None:
        super(sp_model,self).__init__()
        self.d_model = args.rnn_hidden_size + 2
        self.encoder_num = 3

        self.hidden_size = args.rnn_hidden_size

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,nhead=10)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=self.encoder_num)

        self.linear = nn.Linear(in_features=self.d_model,out_features=self.hidden_size)
    def forward(self,SHidden_position):
        '''
        SHidden_position: [N,(rnn_hidden_size+2)]
        out: [N,rnn_hidden_size]
        '''
        out = self.encoder(SHidden_position)
        out = self.linear(out)
        return out

class rnn_phy(nn.Module):
    def __init__(self,args) -> None:
        super(rnn_phy,self).__init__()
        self.embedding_size = args.embedding_size
        self.hidden_size = args.rnn_hidden_size
        self.pre_len = args.pre_len
        self.dt = args.time_step

        self.embedding = nn.Linear(in_features=args.in_feature_num,out_features=self.embedding_size)
        self.rnn_cell = nn.LSTMCell(input_size=self.embedding_size,hidden_size=self.hidden_size)
        self.sp_model = sp_model(args=args)

        self.pre_unit_1 = nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.ReLU(),
        )

        self.pre_unit_2 = nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.ReLU(),
        )

        self.pre_linear = nn.Linear(self.hidden_size,2)

    def forward(self,Tracks):
        '''
        Given a Tracks of [N_neighbor, len_his, 5],
        Output future tracks
        '''
        idx = Tracks[:,0,0]!=0
        input = Tracks[idx,0,1:]
        hidden,cell = self.rnn_cell(self.embedding(input))
        Shidden = self.sp_model(torch.concat((hidden,input[:,:2]),dim=1))
        preinfo = []

        cid_o = [0]

        # encode
        for i in range(Tracks.shape[1]):
            idx = Tracks[:,i,0]!=0
            cid = Tracks[idx,i,0]
            if len(cid) > len(cid_o) and i>0:
                ztensor_hidden = torch.zeros([len(cid),Shidden.shape[1]]).to(next(self.parameters()).device)
                ztensor_cell = torch.zeros([len(cid),cell.shape[1]]).to(next(self.parameters()).device)
                for cid_idx,id in enumerate(cid_o):
                    ztensor_hidden[cid==id] = Shidden[cid_idx]
                    ztensor_cell[cid==id] = cell[cid_idx]
                Shidden = ztensor_hidden
                cell = ztensor_cell
                
            input = Tracks[idx,i,1:]
            Shidden,cell = self.rm_sp(input=input,hidden_cell=(Shidden,cell))
            cid_o = cid
        
        current_frame = Tracks[:,-1,1:]
        # decode
        for i in range(self.pre_len):
            x = self.pre_unit_1(Shidden) + Shidden
            x = self.pre_unit_2(x) + x
            out_a = self.pre_linear(x)
            
            out = self.phy_part(out_a,torch.clone(current_frame)) 
            preinfo.append(out)
            current_frame = out

            Shidden,cell = self.rm_sp(input=out,hidden_cell=(Shidden,cell))

        return torch.stack(preinfo,dim=1)

    def rm_sp(self,input,hidden_cell):
        '''
        hidden: [N,rnn_hidden_szie]
        cell: [N,rnn_hidden_size]
        input: [N,5]
        '''
        hidden,cell = self.rnn_cell(self.embedding(input),hidden_cell)
        Shidden = self.sp_model(torch.concat((hidden,input[:,:2]),dim=1))
        return Shidden,cell
    
    def phy_part(self,a,cf):
        cf[:,2:] = a*self.dt + cf[:,2:]
        cf[:,:2] = a*self.dt**2/2 + cf[:,2:]*self.dt + cf[:,:2]
        return cf