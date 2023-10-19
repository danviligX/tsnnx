from libx.dataio import Args, Dset, vtp_dataloader
import torch.nn as nn
import os
import pickle
import torch

class FLinearNet(nn.Module):
    def __init__(self,args) -> None:
        super(FLinearNet,self).__init__()
        self.basis_num = args.basis_num
        self.embadding_size = args.embadding_size
        self.his_len = args.his_len
        self.pre_len = args.pre_len

        self.embad = nn.Linear(in_features=2,out_features=self.embadding_size)

        self.L_s = nn.Linear(in_features=self.his_len,out_features=self.basis_num*self.his_len)
        # self.L_s = nn.Linear(in_features=self.his_len,out_features=self.pre_len) # Linear
        self.L_m = nn.Linear(in_features=self.basis_num*self.his_len*2,out_features=self.pre_len)
        self.relu = nn.ReLU()

        self.comb = nn.Linear(in_features=self.embadding_size,out_features=2)
    
    def forward(self,input):
        x = self.embad(input)
        x = x.transpose(0,1)
        
        x = self.L_s(x)
        x = self.L_m(torch.concat((torch.sin(x),torch.cos(x)),dim=-1))
        x = self.relu(x)
        x = x.transpose(0,1)

        out = self.comb(x)

        return out

def Obj_FlinearNet():
    args = Args()
    trial = Args()
    
    trial.number = 4

    if torch.cuda.is_available():
        args.device = torch.device("cuda:2")
    else:
        args.device = torch.device("cpu")

    # task setting
    args.model_name = 'FLinear_basis_Net'
    args.data_type = '400'
    args.pre_len = 123
    args.his_len = 75
    args.in_feature_num = 2
    args.time_step = 0.04
    # args.basis_list = [7]
    args.set_path = './data/set/01_tracks.pth'
    args.meta_path = './data/set/01_trainMeta.pth'
    args.dd_index_path = './data/index/highD_01_index_' + args.data_type + '_r01.pkl'

    args.checkpoint_path = './cache/ckp_' + args.model_name + '_' + args.data_type + '_trial_' + str(trial.number) + '.ckp'
    args.model_state_dic_path = ''.join(['./models/',args.model_name,'/trial/',args.data_type,'_trial_',str(trial.number),'.mdic'])
    args.args_path = ''.join(['./models/',args.model_name,'/trial/',args.data_type,'_args_',str(trial.number),'.marg'])

    args.embadding_size = 256
    args.basis_num = 512

    args.opt = 'Adam'
    args.lr = 0.0001
    args.batch_size = 8
    args.epoch_num = 500
    args.ifprune = False
    args.ifresume = False

    initepoch = 0
    if os.path.isfile(args.checkpoint_path):
        cpt = torch.load(args.checkpoint_path)
        args.ifresume = True
        initepoch = cpt['epoch_idx']
        trial = cpt['trial']
        args = cpt['args']
        print('Trial '+ str(trial.number) + ' begain with epcoh '+ str(initepoch))

    args.epoch_num = 500
    
    net = FLinearNet(args=args).to(device=args.device)
    opt = getattr(torch.optim, args.opt)(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    data = Dset(args.set_path,args.device)
    with open(args.dd_index_path,'rb') as path:
        dd_index = pickle.load(path)
        path.close()

    train_loader,valid_loader = vtp_dataloader(train_item_idx=dd_index[0],
                                                              valid_item_idx=dd_index[1],
                                                              batch_size=args.batch_size)

    valid_error = torch.tensor([])

    # checkpoint
    if args.ifresume:
        net.load_state_dict(cpt['net'])
        opt.load_state_dict(cpt['optimizer'])
    
    ESS = 0
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(initepoch,args.epoch_num):
        cpt = {
            'epoch_idx': epoch,
            'net':net.state_dict(),
            'optimizer':opt.state_dict(),
            'trial':trial,
            'args':args,
        }
        torch.save(cpt,args.checkpoint_path)

        net = train(net=net,train_loader=train_loader,criterion=criterion,
                    optimizer=opt,args=args,dset=data,epoch_num=epoch)
        
        epoch_error = valid(net,valid_loader,criterion,data,args)
        print('trial:{}, epoch:{}, error:{}'.format(trial.number,epoch,epoch_error.item()))

        if epoch%5==0:
            if ESS == epoch_error.item(): break
            ESS = epoch_error.item()

        valid_error = torch.concat((valid_error,epoch_error))

    torch.save(net.state_dict(),args.model_state_dic_path)
    torch.save(args,args.args_path)

def train(net,train_loader,criterion,optimizer,args,dset,epoch_num):
    net.train()
    args.ifprune = False
    for batch_num,batched_meta in enumerate(train_loader):
        optimizer.zero_grad()
        for _,meta_item in enumerate(batched_meta):
            # forward
            track = dset.search_track(target_id=meta_item[0],begin_frame=meta_item[1],end_frame=meta_item[2])
            out = net(input=track[:args.his_len,:2])
            
            loss = criterion(track[args.his_len:,:2],out[:,:2])
            loss.backward()
        print('epoch:{},batch:{},loss:{}'.format(epoch_num,batch_num,loss))
        optimizer.step()
    return net

def valid(net,valid_loader,criterion,dset,args):
    error = torch.tensor([])
    net.eval()
    with torch.no_grad():
        for _,batched_one_meta in enumerate(valid_loader):
            meta_item = batched_one_meta[0]
            
            # forward
            track = dset.search_track(target_id=meta_item[0],begin_frame=meta_item[1],end_frame=meta_item[2])
            out = net(input=track[:args.his_len,:2])
            loss = criterion(track[-1,:2],out[-1,:2])

            loss_tensor = torch.tensor([loss.item()])
            error = torch.concat((loss_tensor,error))
        epoch_error = error.mean()
    return torch.tensor([epoch_error])