import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm, trange

import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import inspect
import torch.distributed as dist
import time

@dataclass
class xconfig:
    f:int=5
    input_time:int=3
    iter_step:int=1
    dt:float=25/f*0.04

    ppc_cache:str='./cache/highD_ppc_change_5.pth'
    log_dir:str='logs/29_lstm-phy_f5_L1'
    log_file:str='log.txt'
    val_step:int=250
    ckp_step:int=1000
    assert ckp_step%val_step==0, f'ckp_step must can be divided by val_step'

    hidden_size:int=2**10
    num_layers:int=3
    dropout_rate:float=0.1
    fn_out:int=2
    input_size:int=14

    max_lr:float=1e-3
    min_lr:float=max_lr*0.1
    batch_size:int=2048*2*3 # 1792*4*3
    mini_batch_size:int=2048 # 1792:111104*torch.float32
    max_steps:int=34000 # 17280:1d
    warmup_steps:int=250 # 300

class highD:
    def __init__(self, config:xconfig):
        self.config = config
        self.input_len = config.f * config.input_time
        assert type(self.input_len)==int, f'input frames number should be "int" type'
        self.iter_step = config.iter_step
        self.item_len = self.input_len + self.iter_step

        self.load_files()

        id_set = self.trackMeta.loc[self.trackMeta['numLaneChanges']>0,'id'].values
        train_set = id_set[:-int(len(id_set)*0.1)]
        self.train_set = train_set
        self.test_set = id_set[-int(len(id_set)*0.1):]

        if os.path.exists(config.ppc_cache):
            self.ppc_data = torch.load(config.ppc_cache)
        else:
            self.preprocess()
        
    def load_files(self,
                    tracks_path = './data/raw/13_tracks.csv',
                    tracksMeta_path = './data/raw/13_tracksMeta.csv',
                    recordingMeta_path = './data/raw/13_recordingMeta.csv'                   
                   ):
        
        self.track = pd.read_csv(tracks_path)
        self.trackMeta = pd.read_csv(tracksMeta_path)
        self.recordingMeta = pd.read_csv(recordingMeta_path)

        self.upperLaneMarkings = np.array([float(x) for x in self.recordingMeta['upperLaneMarkings'][0].split(';')]) # direction = 1, [13.55, 17.45, 21.12, 24.91]
        self.lowerLaneMarkings = np.array([float(x) for x in self.recordingMeta['lowerLaneMarkings'][0].split(';')]) # direction = 2, [30.53, 34.43, 38.10, 42.11]

        indexes = [0,1,2,3,6,7,8,9,16,17,18,19,20,21,22,23,24]
        self.used_kw = [self.track.keys()[index] for index in indexes]
        self.data_fr = self.recordingMeta['frameRate'].values.item()
    
    def preprocess(self):
        assert self.data_fr%self.config.f==0, "make sure data frame rate is divisible by input frame rate"
        sample_scale = self.data_fr//self.config.f
        div_frame = self.data_fr * self.config.input_time + self.iter_step * sample_scale
     
        data_set = []
        for id in tqdm(self.train_set):
            track_array = self.track.loc[self.track['id']==id, self.used_kw].values
            len_track_array = len(track_array)
            if len_track_array<div_frame: continue
            track_array = self.get_track(id=id, track_array=track_array)

            # split into item_len length
            _, n = track_array.shape
            item = np.lib.stride_tricks.sliding_window_view(track_array, (self.item_len, n)).squeeze()
            data_set.append(item)
        data_set = np.concatenate(data_set,axis=0)
        
        self.ppc_data = torch.tensor(data_set).to(torch.float32)
        torch.save(self.ppc_data, self.config.ppc_cache)

    def get_track(self, id:int, track_array=None):
        if track_array is None:
            track_array = self.track.loc[self.track['id']==id, self.used_kw].values
        
        sample_scale = self.data_fr//self.config.f
        track_array = track_array[::sample_scale]
        # rotation and translation, set origin to (x[0], bond_center)
        if track_array[0,3] > self.upperLaneMarkings[-1]: 
            # direction = 2
            track_array[:,2:8] = -track_array[:,2:8]
            track_array[:,3] = track_array[:,3] + self.lowerLaneMarkings[-1]
        else:
            # direction = 1
            track_array[:,3] = track_array[:,3] - self.upperLaneMarkings[0]
        track_array[:,2] = track_array[:,2] - track_array[0,2] # x = x - x[0]

        return track_array

class DataLoaderx:
    def __init__(self, batch_size:int, process_rank:int, num_processes:int, config:xconfig, dataset:highD, split=None) -> None:
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_process = num_processes

        assert split in {"train", "val"}
        self.split = split
        self.data = dataset
        self.reset()
    
    def reset(self):
        if self.split == 'train':
            self.current_point = self.batch_size * self.process_rank
        if self.split == 'val':
            self.current_point = -len(self.data.ppc_data)%self.batch_size * self.process_rank
            
    
    def next_batch(self):
        batch_size = self.batch_size

        if self.current_point + (batch_size*self.num_process + 1) > len(self.data.ppc_data):
            self.current_point = batch_size * self.process_rank

        self.current_point += batch_size*self.num_process
        buf = self.data.ppc_data[self.current_point:self.current_point + batch_size]
        x = buf[:,:-1]
        y = buf[:,-1]

        return x, y

class net(nn.Module):
    def __init__(self, config:xconfig):
        super().__init__()
        self.config = config
        self.dt = config.dt
        self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_size, batch_first=True, num_layers=config.num_layers, dropout=config.dropout_rate)
        self.fnn = nn.Linear(in_features=config.hidden_size, out_features=config.fn_out)
        # self.criterion = nn.SmoothL1Loss(reduction='sum')

    def forward(self, x:torch.tensor, target:torch.tensor=None):
        a, (h, c) = self.lstm(x)
        a = self.fnn(a[:,-1])

        # accelration -> position
        v = x[:,-1,2:4] + a*self.dt
        s = x[:,-1,:2] + v*self.dt

        # out = torch.concat((s,v))
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

def main():
    # ============================== Mulit GPUs ==============================
    # Multi GPUs
    # torchrun --standalone --nproc_per_node=3 train_gpt2.py

    ddp = int(os.environ.get('RANK',-1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = 'mps'
        print(f"using devices:{device}")

    # threads setting
    # torch.set_num_threads(3)
    # export OMP_NUM_THREADS=6

    # ============================== Precision ==============================
    # torch.set_float32_matmul_precision('high') # set TF32

    # ============================== Batch size ==============================
    config = xconfig
    total_batch_size = config.batch_size
    m_batch_size = config.mini_batch_size # micro batch size
    assert total_batch_size%(m_batch_size*ddp_world_size) == 0, "make sure total_batch_size is divisible by batch_size*ddp_world_size"
    grad_accum_steps = total_batch_size//(m_batch_size*ddp_world_size)

    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    
    # ============================== Dataloader ==============================
    data = highD(config=config)
    train_loader = DataLoaderx(batch_size=m_batch_size, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", config=config, dataset=data)
    val_loader = DataLoaderx(batch_size=m_batch_size, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", config=config, dataset=data)

    # ============================== Model ==============================
    model = net(config=config)
    model.to(device)
    model = torch.compile(model) # pre-compile
    if ddp: model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # ============================== Learning Rate ==============================
    max_lr = config.max_lr
    min_lr = config.min_lr
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps
    def get_lr(it):
        if it<warmup_steps:
            return max_lr * (it+1)/ warmup_steps
        if it>max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps)/(max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5*(1.0 + math.cos(math.pi*decay_ratio))
        return min_lr + coeff*(max_lr - min_lr)
    
    # ============================== Optimizer ==============================
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    # ============================== Log file ==============================
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, config.log_file)
    with open(log_file,'w') as f:
        pass
    
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        # once in a while evaluate our validation loss
        if step % config.val_step == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x[:,:,2:-1].to(device), y[:,2:8].to(device)
                    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x,y)
                    loss = loss/val_loss_steps
                    val_loss_accum += loss.detach()
            
            if ddp: dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                
                # ====================== Checkpoint ================================
                if step>0 and (step%config.ckp_step == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint,checkpoint_path)

        # ============================== Training ==============================
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x[:,:,2:-1].to(device), y[:,2:8].to(device)

            # with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x,y)
            # import code; code.interact(local=locals()) # inter action
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp: model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        
        if ddp: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # control the norm of loss
        lr = get_lr(step)
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (train_loader.batch_size * grad_accum_steps * ddp_world_size) / (t1 -t0)
        if master_process:
            print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt:{dt:.2f}ms | tok/sec:{tokens_per_sec:.2f}")
            with open(log_file,"a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
            
    if ddp: destroy_process_group()

if __name__=="__main__": main()