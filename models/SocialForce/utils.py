import torch
import torch.nn as nn
import inspect
import numpy as np

class config_SFM:
    f:int=5
    input_time:int=3
    iter_step:int=1
    dt:float=25/f*0.04

    # batch_size: int=32
    # neighbors_num: int=8
    # dt: float=0.02
    # embd: int=16
    select_lane: list=[0,-1] # two borders of a highway road

    batch_size: int = 256*3*4
    max_lr: float = 0.001
    min_lr: float = 0.00001
    warmup_steps: int = 30
    max_steps: int = 1000

    mini_batch_size = 256
    val_step: int = 100
    ckp_step: int = 1000
    assert ckp_step % val_step == 0, f"ckp_step should be a multiple of val_step, but got {ckp_step} and {val_step}"

    # preprocessed data path, lane changed data, with 5Hz sampling rate
    ppc_cache: str = './cache/highD_ppc_change_5.pth'

    log_file: str = 'log2.txt'
    log_dir: str = './logs/sfm'

    neighbors_num: int = 8
    embd: int = 32

class weightConstraint(object):
    def __init__(self, min_b:float=None, max_b:float=None):       
        # self.max_b = max_b if max_b is not None else None
        # self.min_b = min_b if min_b is not None else None
        self.max_b = max_b
        self.min_b = min_b
        
    def __call__(self,module:nn.Module):
        if hasattr(module,'weight'):
            w = module.weight.data
            w = w.clamp(min=self.min_b, max=self.max_b)
            module.weight.data = w

class monotonic_fun(nn.Module):
    def __init__(self, config:config_SFM, increasing=False) -> None:
        super().__init__()
        self.in_linear = nn.Linear(1,config.embd)
        self.out_linear= nn.Linear(config.embd,1)
        self.increasing = increasing
    def forward(self,x:torch.Tensor):
        x = self.in_linear(x)
        x = torch.relu(x) if self.increasing else torch.relu(-x)
        x = self.out_linear(x)
        return x

class SFM(nn.Module):
    def __init__(self, config:config_SFM) -> None:
        super().__init__()
        '''
        requires: frames with ego and neighbors information
        outputs: the force on ego, i.e. the accerleration
        '''
        self.config = config
        self.dt = config.dt
        # clamp setting
        _clamp_non_negative = weightConstraint(min_b=0)

        # Variable setting
        self.recording_time = torch.zeros(config.batch_size, config.neighbors_num, 2)   # Memory of time delation

        # Parameters setting
        self.attr_destination_para = nn.Parameter(torch.relu(torch.rand(2)))
        self.effective_angle = nn.Parameter(torch.sigmoid(torch.rand(1)))

        # monotonic function
        self.repu_nei_f = monotonic_fun(config=config, increasing=False).apply(_clamp_non_negative)
        self.attr_nei_f = monotonic_fun(config=config, increasing=False).apply(_clamp_non_negative)
        self.repu_bor_f = monotonic_fun(config=config, increasing=False).apply(_clamp_non_negative)
        self.delation_time_f = monotonic_fun(config=config, increasing=False).apply(_clamp_non_negative)

    def attr_destination(self, ego:torch.Tensor, desired_velocity:torch.Tensor=None):
        '''
        ego: (batch_size, num_features): [ 'id',
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
        desired_velocity:Nx2: ['x', 'y'], 
            which is equal to desired position in one time step
        '''
        if desired_velocity is None: 
            dev = next(self.parameters()).device
            desired_velocity = torch.stack((ego[:,3:5].norm(dim=-1),torch.zeros(len(ego)).to(dev)),dim=1)
        vx = self.attr_destination_para[1]*desired_velocity - ego[:,3:5]
        return (vx/self.attr_destination_para[0]).unsqueeze(1)

    def attr_repu_neighbors(self, ego:torch.Tensor, nei:torch.Tensor):
        """
        Compute the reputations of neighbors of ego node based on their positions.
        ego: (batch_size, num_frames, num_features)
        nei: (batch_size, num_neighbors, num_features)
        """
        dt = self.config.dt
        dev = next(self.parameters()).device

        r = nei[:,:,2:4] - ego[:,-1,None,2:4]
        mask = nei[:,:,2:4]==0

        r_norm = r.norm(dim=-1).unsqueeze(-1)
        
        # calculate the durality of neighbors
        nei_durality = torch.zeros(8).to(dev)
        for i in ego[1,:,8:-1]:
            nei_durality += torch.isin(ego[1,-1,8:-1],i).float()

        # calculate the attraction and repulsion forces
        f_attr = self.attr_nei_f(r_norm)*r/r_norm*self.delation_time_f(nei_durality.unsqueeze(-1))
        f_attr[mask]=0

        v_nei = nei[:,:,4:6]
        b = r_norm + (r + v_nei*dt).norm(dim=-1).unsqueeze(-1)**2 - (v_nei*dt).norm(dim=-1).unsqueeze(-1)**2
        b = (b**0.5)/2
        f_repu = self.repu_nei_f(b)*r/r_norm
        f_repu[mask]=0

        return f_attr, f_repu

    def repu_borders(self, ego:torch.Tensor, border:torch.Tensor):
        '''
        A special function for HighD data. This function only calculates the force from y-axis, since HighD data is about highway.
        border: (border_num, )
        '''
        select_lane = self.config.select_lane
        r = (ego[:,3,None] - border)[:,select_lane]
        r_norm = r.abs()
        f_repu = self.repu_bor_f(r_norm.unsqueeze(-1))*(r/r_norm).unsqueeze(-1)
        f_repu = torch.concat((torch.zeros_like(f_repu),f_repu),dim=-1)
        return f_repu

    def cal_force(self, ego:torch.Tensor, nei:torch.Tensor, border:torch.Tensor, desired_velocity:torch.Tensor=None):
        f_attr_destination = self.attr_destination(ego[:,-1], desired_velocity)
        f_attr_nei, f_repu_nei = self.attr_repu_neighbors(ego, nei)
        f_repu_border = self.repu_borders(ego[:,-1], border)

        e = ego[:,-1,4:6]
        e = e/e.norm(dim=1).reshape(-1,1)
        f_attr_destination_clamped = self.angle_clamp(f_attr_destination, e)
        f_attr_nei_clamped = self.angle_clamp(f_attr_nei, e)
        f_repu_nei_clamped = self.angle_clamp(f_repu_nei, e)
        f_repu_border_clampled = self.angle_clamp(f_repu_border, e)

        f_destination = f_attr_destination_clamped
        f_neighors = f_attr_nei_clamped + f_repu_nei_clamped
        f_border = f_repu_border_clampled

        return f_destination, f_neighors, f_border

    def angle_clamp(self, vector:torch.Tensor, target:torch.Tensor):
        index = torch.cosine_similarity(target.repeat(vector.shape[1],1,1).transpose(0,1),vector,dim=-1).abs() > self.effective_angle
        batched_force = []
        for i in range(len(index)):
            force = vector[i,index[i]].sum(dim=0)
            batched_force.append(force)
        batched_force = torch.stack(batched_force)
        return batched_force
    
    def forward(self, ego:torch.Tensor, nei:torch.Tensor, border:torch.Tensor, target:torch.Tensor=None, desired_velocity:torch.Tensor=None):
        f_destination, f_neighors, f_border = self.cal_force(ego, nei, border, desired_velocity)
        a = f_destination + f_neighors + f_border

        v = ego[:,-1,2:4] + a*self.dt
        s = ego[:,-1,:2] + v*self.dt

        out = torch.concat((s,v,a),dim=1)
        # out = a

        loss = None
        if target is not None:
            # loss = nn.functional.mse_loss(out,target)
            loss = nn.functional.l1_loss(out,target)
        return out, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # Generate an optimizer with decay learning rate.
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_group = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        # print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optimizer

def get_neighbors(ego, data):
    nei_ids = ego[:,-1,8:-1]
    f_ids = ego[:,-1,0]

    # search for neighboring features in the track data, and pad with zeros if necessary to make it 8xnum_features
    batched_nei = []
    for ids, f in zip(nei_ids, f_ids):
        temp = data.track.loc[(data.track['id'].isin(ids.numpy()))&(data.track['frame']==f.numpy()),data.used_kw].values
        if len(temp)<8: temp = np.pad(temp, ((0,8-len(temp)),(0,0)), 'constant', constant_values=0)
        batched_nei.append(temp)

    batched_nei = torch.Tensor(np.array(batched_nei))
    return batched_nei

def get_border(ego, data):
    car_id = ego[0,-1,1].int().item()
    direction = data.trackMeta.loc[data.trackMeta['id']==car_id,'drivingDirection'].values
    if direction == 1:
        return torch.Tensor(data.upperLaneMarkings)
    else:
        return torch.Tensor(data.lowerLaneMarkings)
