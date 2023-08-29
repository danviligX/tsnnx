import torch.nn as nn
import torch

class FLinear(nn.Module):
    def __init__(self,in_feature,out_feature,basis_num) -> None:
        super(FLinear,self).__init__()

        self.L_spanning = nn.Linear(in_features=in_feature,out_features=basis_num)
        self.L_mapping = nn.Linear(in_features=2*basis_num,out_features=2*basis_num)
        self.L_combine = nn.Linear(in_features=2*basis_num,out_features=out_feature)
        self.relu = nn.ReLU()

    def forward(self,input):
        x = self.L_spanning(input)
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        x = torch.concat((x_sin,x_cos),dim=-1)
        x = self.L_mapping(x)
        x = self.relu(x)
        x = self.L_combine(x)
        return x