import os
import pickle

import optuna
import torch
import torch.nn as nn

import libx.nnx as nnx
from libx.dataio import Args, Dset, vtp_dataloader


class vtp_Flinear(nn.Module):
    def __init__(self,args) -> None:
        super(vtp_Flinear,self).__init__()
        self.embadding_size = args.embadding_size
        self.basis_num = args.basis_num

        self.his_len = args.his_len
        self.pre_len = args.pre_len

        self.emabadding = nn.Linear(in_features=4,out_features=self.embadding_size)
        
        self.flinear = nnx.MFLP(dim_list=[self.his_len,self.pre_len],basis_list=[self.basis_num])

        self.deembadding = nn.Linear(in_features=self.embadding_size,out_features=2)

    def forward(self,dset,meta_item):
        track = dset.search_track(meta_item[0],meta_item[1],meta_item[1]+self.his_len)
        seq = self.emabadding(track)
        seq = seq.transpose(0,1)

        seq = self.flinear(seq)

        out = seq.transpose(0,1)
        out = self.deembadding(out)

        return out


        

def obj_vtp_Flinear(trial):
    args = Args()

    # cuda
    if torch.cuda.is_available():
        args.device = torch.device("cuda:2")
    else:
        args.device = torch.device("cpu")

    # task setting
    args.model_name = 'vtp_Flinear'
    args.model_type = '50'
    args.pre_len = 125
    args.his_len = 75
    args.set_path = './data/set/01_tracks.pth'
    args.meta_path = './data/set/01_trainMeta.pth'
    args.dd_index_path = './data/index/highD_01_index_50_r01.pkl'
    args.checkpoint_path = './cache/ckp_' + args.model_name + '_' + args.model_type + '_trial' + str(trial.number) + '.ckp'
    args.model_state_dic_path = ''.join(['./models/',args.model_name,'/trial/',args.model_type,'_trial_',str(trial.number),'.mdic'])
    args.args_path = ''.join(['./models/',args.model_name,'/trial/',args.model_type,'_args_',str(trial.number),'.marg'])
    
    # net initialization parameters
    args.embadding_size = trial.suggest_int("embadding_size", 32, 1024,step=32)
    args.basis_num = trial.suggest_int("hidden_size", 32, 1024,step=16)

    # hyperparameters selection with optuna
    args.opt = trial.suggest_categorical("optimizer", ["RMSprop", "SGD", "Adam"])
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    args.batch_size = trial.suggest_int("batch_size", 4, 32,step=4)
    args.epoch_num = trial.suggest_int("epoch_num",1,20)
    args.ifprune = False

    # break point
    initepoch = 0
    if os.path.isfile(args.checkpoint_path):
        cpt = torch.load(args.checkpoint_path)
        initepoch = cpt['epoch_num']
        trial = cpt['trial']
        args = cpt['args']
        print('Trial '+ str(trial.number) + ' begain with epcoh '+ str(initepoch))

    net = vtp_Flinear(args=args).to(device=args.device)
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

    # break point
    if os.path.isfile(args.checkpoint_path):
        net.load_state_dict(cpt['net'])
        opt.load_state_dict(cpt['optimizer'])

    ESS = 0
    for epoch in range(initepoch,args.epoch_num):
        cpt = {
                'epoch_num':epoch,
                'net':net.state_dict(),
                'optimizer':opt.state_dict(),
                'trial':trial,
                'args':args,
            }
        torch.save(cpt,args.checkpoint_path)
        

        net = train(net=net,train_loader=train_loader,criterion=criterion,
                    optimizer=opt,args=args,dset=data)

        epoch_error = valid(net,valid_loader,criterion,data,args)
        print('trial:{}, epoch:{}, loss:{}'.format(trial.number,epoch,epoch_error.item()))

        if epoch%5==0:
            if ESS == epoch_error.item(): raise optuna.exceptions.TrialPruned()
            ESS = epoch_error.item()

        valid_error = torch.concat((valid_error,epoch_error))
        trial.report(epoch_error.item(),epoch)

        if trial.should_prune(): 
            args.ifprune = True
            raise optuna.exceptions.TrialPruned()
        if args.ifprune: raise optuna.exceptions.TrialPruned()

    optuna_error = valid_error.mean()

    torch.save(net.state_dict(),args.model_state_dic_path)
    torch.save(args,args.args_path)
    return optuna_error

def train(net,train_loader,criterion,optimizer,args,dset):
    net.train()
    args.ifprune = False
    for batch_num,batched_meta in enumerate(train_loader):
        optimizer.zero_grad()
        for _,meta_item in enumerate(batched_meta):
            # forward
            out = net(dset=dset,meta_item=meta_item)
            loss = criterion(dset.search_track(meta_item[0],meta_item[2]-args.pre_len,meta_item[2])[:,:2],out)

            if torch.isinf(loss).any():
                args.ifprune = True
                break
            if torch.isnan(loss).any(): 
                args.ifprune = True
                break
            loss.backward()
        # print('batch:{},loss:{}'.format(batch_num,loss.item()))
        if args.ifprune: break
        if args.ifprune: break
        optimizer.step()
    return net

def valid(net,valid_loader,criterion,dset,args):
    error = torch.tensor([])
    net.eval()
    if args.ifprune: return torch.tensor([9999999])
    with torch.no_grad():
        for _,batched_one_meta in enumerate(valid_loader):
            meta_item = batched_one_meta[0]
            
            # forward
            out = net(dset=dset,meta_item=meta_item)
            loss = criterion(dset.search_track(meta_item[0],meta_item[2]-args.pre_len,meta_item[2])[:,:2],out)

            loss_tensor = torch.tensor([loss.item()])
            error = torch.concat((loss_tensor,error))
        epoch_error = error.mean()
        # epoch_error_std = error.std()
    
    return torch.tensor([epoch_error])

def test(net,test_loader,criterion,dset,args):
    test_length = len(test_loader)
    error = torch.zeros(test_length)
    with torch.no_grad():
        for item_idx,batched_one_meta in enumerate(test_loader):
            meta_item = batched_one_meta[0]

            # forward
            out = net(dset=dset,meta_item=meta_item)
            loss = criterion(dset.search_track(meta_item[0],meta_item[2]-args.pre_len,meta_item[2])[-1,:2],out[-1])

            error[item_idx] = loss

    return error