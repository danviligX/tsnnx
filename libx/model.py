import torch.nn as nn
import torch
import optuna
from dataio import Args,vtp_dataloader,Dset,data_divide

class net(nn.Module):
    def __init__(self,args) -> None:
        super(net,self).__init__()
    def forward(self,input):
        pass

def net_object(trial):
    args = Args()

    # cuda
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    # task setting
    args.model_name = 'VTPnet'
    args.pre_len = 125
    args.his_len = 75
    args.set_path = './data/set/01_tracks.pth'
    args.meta_path = './data/set/01_trainMeta.pth'

    # net initialization parameters
    args.embadding_size = trial.suggest_int("embadding_size", 8, 128,step=8)
    args.hidden_size = trial.suggest_int("hidden_size", 32, 512,step=16)
    args.te_hidden_size = trial.suggest_int("te_hidden_size", 32, 512,step=16)
    args.temproal_mlp_size = trial.suggest_int("temproal_mlp_size", 32, 512,step=16)

    # soical attention MLPs
    args.rel_mlp_hidden_size = trial.suggest_int("rel_mlp_hidden_size", 8, 128,step=8)
    args.abs_mlp_hidden_size = trial.suggest_int("abs_mlp_hidden_size", 8, 128,step=8)
    args.attention_mlp_hidden_size = trial.suggest_int("attention_mlp_hidden_size", 8, 128,step=8)

    # hyperparameters selection with optuna
    args.opt = trial.suggest_categorical("optimizer", ["RMSprop", "SGD", "Adam"])
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    args.batch_size = trial.suggest_int("batch_size", 4, 32,step=4)
    args.epoch_num = trial.suggest_int("epoch_num",5,50)

    net = net(args=args).to(device=args.device)
    opt = getattr(torch.optim, args.opt)(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # data prepare
    data = Dset(args.dataset_path)
    meta = torch.load(args.meta_path)
    train_validation_idx = data_divide(len(meta),para=10)
    train_loader,valid_loader = vtp_dataloader(train_item_idx=train_validation_idx[0],
                                                              valid_item_idx=train_validation_idx[1],
                                                              batch_size=args.batch_size)

    # initialization for optuna error
    valid_error = torch.tensor([])

    ESS = 0
    for epoch in range(args.epoch_num):
        net = train(net=net,train_loader=train_loader,criterion=criterion,
                    optimizer=opt,args=args,set_file_list=data)

        epoch_error,_ = valid(net,valid_loader,criterion,data,device=args.device)
        print('trial:{}, epoch:{}, loss:{}'.format(trial.number,epoch,epoch_error.item()))

        if epoch%5==0:
            if ESS == epoch_error.item(): raise optuna.exceptions.TrialPruned()
            ESS = epoch_error.item()

        valid_error = torch.concat((valid_error,epoch_error))
        trial.report(epoch_error.item(),epoch)
        if epoch_error > 1000: raise optuna.exceptions.TrialPruned()
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        if torch.isnan(epoch_error).any(): raise optuna.exceptions.TrialPruned()
        if torch.isinf(epoch_error).any(): raise optuna.exceptions.TrialPruned()

    optuna_error = valid_error.mean()

    torch.save(net.state_dict(),''.join(['./result/',args.model_name,'/trial/trial_',str(trial.number),'.mdic']))
    torch.save(args,''.join(['./result/',args.model_name,'/trial/args_',str(trial.number),'.marg']))
    return optuna_error

def train(net,train_loader,criterion,optimizer,args,set_file_list):
    net.train()
    for _,batched_meta in enumerate(train_loader):
        optimizer.zero_grad()
        for idx,meta_item in enumerate(batched_meta):
            input_track = set_file_list.frange(batched_meta[1],batched_meta[2])

            # forward
            out = net(input=input_track)
            loss = criterion(set_file_list.search_track[meta_item[0],meta_item[1],meta_item[2]],out)

            loss.backward()
        optimizer.step()
    return net

def valid():
    pass