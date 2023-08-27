import torch.nn as nn
import torch
import optuna
import pickle
from libx.dataio import Args,vtp_dataloader,Dset,data_divide

class vtp_net(nn.Module):
    def __init__(self,args) -> None:
        super(vtp_net,self).__init__()
        self.embadding_size = args.embadding_size
        self.hidden_size = args.hidden_size

        self.his_len = args.his_len
        self.pre_len = args.pre_len

        self.emabadding = nn.Linear(in_features=4,out_features=self.embadding_size)
        self.linear = nn.Linear(in_features=self.his_len,out_features=self.pre_len)

        self.deembadding = nn.Linear(in_features=self.embadding_size,out_features=2)

    def forward(self,dset,meta_item):
        track = dset.search_track(meta_item[0],meta_item[1],meta_item[1]+self.his_len)
        seq = self.emabadding(track)
        seq = seq.transpose(0,1)
        seq = self.linear(seq)
        out = seq.transpose(0,1)
        out = self.deembadding(out)

        return out

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
    args.dd_index_path = './cache/dd_index.pkl'

    # net initialization parameters
    args.embadding_size = trial.suggest_int("embadding_size", 8, 128,step=8)
    args.hidden_size = trial.suggest_int("hidden_size", 32, 512,step=16)
    args.te_hidden_size = trial.suggest_int("te_hidden_size", 32, 512,step=16)
    args.temproal_mlp_size = trial.suggest_int("temproal_mlp_size", 32, 512,step=16)

    # hyperparameters selection with optuna
    args.opt = trial.suggest_categorical("optimizer", ["RMSprop", "SGD", "Adam"])
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    args.batch_size = trial.suggest_int("batch_size", 4, 32,step=4)
    args.epoch_num = trial.suggest_int("epoch_num",5,50)

    net = vtp_net(args=args).to(device=args.device)
    opt = getattr(torch.optim, args.opt)(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # data prepare
    data = Dset(args.set_path,args.device)
    
    with open(args.dd_index_path,'rb') as path:
        dd_index = pickle.load(path)
        path.close()

    train_loader,valid_loader = vtp_dataloader(train_item_idx=dd_index[0],
                                                              valid_item_idx=dd_index[1],
                                                              batch_size=args.batch_size)

    # initialization for optuna error
    valid_error = torch.tensor([])

    ESS = 0
    for epoch in range(args.epoch_num):
        net = train(net=net,train_loader=train_loader,criterion=criterion,
                    optimizer=opt,args=args,dset=data)

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

def train(net,train_loader,criterion,optimizer,args,dset):
    net.train()
    for _,batched_meta in enumerate(train_loader):
        optimizer.zero_grad()
        for idx,meta_item in enumerate(batched_meta):
            # forward
            out = net(dset=dset,meta_item=meta_item)
            loss = criterion(dset.search_track(meta_item[0],meta_item[2]-args.pre_len,meta_item[2])[:,:2],out)

            loss.backward()
            print(loss)
        optimizer.step()
    return net

def valid(net,valid_loader,criterion,dset,args):
    error = torch.tensor([])
    net.eval()
    with torch.no_grad():
        for _,batched_one_meta in enumerate(valid_loader):
            meta_item = batched_one_meta[0]
            
            # forward
            out = net(dset=dset,meta_item=meta_item)
            loss = criterion(dset.search_track(meta_item[0],meta_item[2]-args.pre_len,meta_item[2])[:,:2],out)

            loss_tensor = torch.tensor([loss.item()])
            error = torch.concat((loss_tensor,error))
        epoch_error = error.mean()
        epoch_error_std = error.std()
    
    return torch.tensor([epoch_error]),epoch_error_std

def test(net,test_loader,criterion,dset,args):
    test_length = len(test_loader)
    error = torch.zeros(test_length,2)
    with torch.no_grad():
        for item_idx,batched_one_meta in enumerate(test_loader):
            meta_item = batched_one_meta[0]

            # forward
            out = net(dset=dset,meta_item=meta_item)
            loss = criterion(dset.search_track(meta_item[0],meta_item[2]-args.pre_len,meta_item[2])[:,:2],out)

            error[item_idx,0] = meta_item[0].item()
            error[item_idx,1] = loss

    return error