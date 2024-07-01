import torch
import torch.nn as nn

class SFM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.direction = torch.tensor([1,0])
        self.effective_angle = nn.Parameter(torch.sigmoid(torch.rand(1)))
        self.attr_destination_para = nn.Parameter(torch.relu(torch.rand(2)))

    def attr_destination(self, ego, desired_velocity=None):
        '''
        N: batch size
        ego:Nx16: [ 'id',
                    'x',
                    'y',
                    'xVelocity',
                    'yVelocity',
                    'xAcceleration',
                    'yAcceleration',
                    'precedingId', the vehicle in front of ego, index=7
                    'followingId', the vehicle behind the ego
                    'leftPrecedingId',
                    'leftAlongsideId',
                    'leftFollowingId',
                    'rightPrecedingId',
                    'rightAlongsideId',
                    'rightFollowingId',
                    'laneId']
        desired_velocity:Nx2: ['x', 'y']
        '''
        if desired_velocity is None:
            # Motivation: Keep in same velocity
            vx = self.attr_destination_para[1]*ego[:,2] - ego[:,1]
            return torch.stack((vx, ego[:,2]))/self.attr_destination_para[0]
        else:
            # Motivation: Next desired position
            vx = self.attr_destination_para[1]*desired_velocity - ego[:,1:3]
            return vx/self.attr_destination_para[0]

    def attr_repu_neighbors(self, ego, nei, t=None):
        '''
        nei:Nx8x16: 8xtype(ego)
        '''
        r = nei[:,:,1:3] - ego[1:3].unsqueeze(1)
        f_attr = self.attr_nei_grad_W(r, t)
        f_repu = self.repu_nei_grad_W(r, nei)

        return f_attr, f_repu

    def repu_borders(self, ego, border):
        r = ego[:,2] - border # which data structure
        f = self.repu_borders_grad_U(r)
        return f
    
    # Used function for social force
    def attr_nei_grad_W(self, r, t):
        pass
    def repu_nei_grad_W(self,r:torch.Tensor):
        '''
        r: Nx8x2
        '''
        
        pass
    def repu_borders_grad_U(self, r):
        pass
    def apro_exp(self,x: torch.Tensor):
        pass