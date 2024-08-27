# train hyperparameters
batch_size: int = 128
lr: float = 0.001
mini_batch_size = 128
val_step: int = 100
ckp_step: int = 1000
assert ckp_step % val_step == 0, f"ckp_step should be a multiple of val_step, but got {ckp_step} and {val_step}"

# preprocessed data path, lane changed data, with 5Hz sampling rate
ppc_path: str = './cache/highD_ppc_change_5.pth'

log_file: str = './cache/log.txt'

neighbors_num: int = 8
embd: int = 16