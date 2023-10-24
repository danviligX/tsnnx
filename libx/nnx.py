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

        