# traing for SFM
from models.SocialForce.utils import *
import torch
import os
from torch.distributed import init_process_group
from models.LSTM_Vanilla.hdlstm import highD, DataLoaderx
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist
import time
import math

# ============================== Config =========
# parameters
config = config_SFM()

# ============================== Mulit GPUs ==============================
    # Multi GPUs
    # OMP_NUM_THREADS=6 torchrun --standalone --nproc_per_node=3 train_gpt2.py
    # nohup xxx &> nohup.out 2>&1 & disown

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

# ============================== Batch size ==============================
total_batch_size = config.batch_size    # total batch size
m_batch_size = config.mini_batch_size   # micro batch size
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
model = SFM(config=config)
model.to(device)
model = torch.compile(model) # pre-compile
if ddp: model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
raw_model = model.module if ddp else model
# print unused parameters
if master_process:
    print(f"total number of parameters: {sum(p.numel() for p in raw_model.parameters())}")
    print(f"number of trainable parameters: {sum(p.numel() for p in raw_model.parameters() if p.requires_grad)}")
    print(f"number of unused parameters: {sum(p.numel() for p in raw_model.parameters() if not p.requires_grad)}")
# torch.set_float32_matmul_precision('high')

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
                x, y = x, y[:,2:8]
                nei = get_neighbors(x, data)
                border = get_border(x, data)
                # with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x.to(device),nei.to(device),border.to(device),y.to(device))
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
        x, y = x, y[:,2:8]
        nei = get_neighbors(x, data)
        border = get_border(x, data)
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(x.to(device),nei.to(device),border.to(device),y.to(device))
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