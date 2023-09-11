import torch.nn as nn
import torch

class Fmap(nn.Module):
    '''
    Fourier series for function: $R^n \to R$
    '''
    def __init__(self,in_dim,basis_dim=7) -> None:
        super(Fmap,self).__init__()
        # basis_dim = basis_dim**in_dim
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
        # out = torch.zeros([self.out_dim]).to(next(self.parameters()).device)
        out = []
        for idx, Fl in enumerate(self.Fls):
            # out[idx] = Fl(input)
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