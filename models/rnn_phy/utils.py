from libx.dataio import Args, Dset, vtp_dataloader
from libx.nnx import rnn_phy
import torch.nn as nn
import time
import os
import torch

def Obj_rnn_phy():
    args = Args()
    trial = Args()
    
    trial.number = 5

    if torch.cuda.is_available():
        args.device = torch.device("cuda:2")
    else:
        args.device = torch.device("cpu")

    # task setting
    args.model_name = 'rnn_phy'
    args.data_type = '1'
    args.pre_len = 125
    args.his_len = 75
    args.in_feature_num = 4
    args.time_step = 0.04
    # args.basis_list = [7]
    args.set_path = './data/set/01_tracks.pth'
    args.meta_path = './data/set/01_trainMeta.pth'
    args.dd_index_path = './data/index/highD_01_index_' + args.data_type + '_r02_Meta_1.pth'

    args.checkpoint_path = './cache/ckp_' + args.model_name + '_'  + '_trial_' + str(trial.number) + '.ckp'
    args.model_state_dic_path = ''.join(['./models/',args.model_name,'/trial_p/',str(int(time.time())),'_trial_',str(trial.number),'.mdic'])
    args.args_path = ''.join(['./models/',args.model_name,'/trial_p/',str(int(time.time())),'_args_',str(trial.number),'.marg'])

    args.embedding_size = 256
    args.rnn_hidden_size = 1028

    args.opt = 'Adam'
    args.lr = 0.0001
    args.batch_size = 22
    args.epoch_num = 300
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
    
    net = rnn_phy(args=args).to(device=args.device)
    opt = getattr(torch.optim, args.opt)(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    Meta = torch.load('./d1t.pth').to(args.device)
    data = Dset(args.set_path,args.device)
    data.mod_by_direction(Meta)
    dd_index = torch.load(args.dd_index_path)

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

        if epoch_error.item() == valid_error.min().item():
            torch.save(net.state_dict(),args.model_state_dic_path)
            torch.save(args,args.args_path)
            print('Weight update with EOV:{}'.format(epoch_error.item())) # EOV: error on valid set
    
    torch.save(net.state_dict(),args.model_state_dic_path)
    torch.save(args,args.args_path)

def train(net,train_loader,criterion,optimizer,args,dset,epoch_num):
    net.train()
    args.ifprune = False
    for batch_num,batched_meta in enumerate(train_loader):
        optimizer.zero_grad()
        for _,meta_item in enumerate(batched_meta):
            # forward
            frame_info = dset.frame(meta_item[1]+args.his_len-1)
            Tracks = dset.search_track(frame_info[:,0].int().cpu().numpy(),meta_item[1],meta_item[1]+args.his_len-1)
            out = net(Tracks)
            # target_Tracks = dset.search_track(meta_item[0].item(),meta_item[2]-args.pre_len+1,meta_item[2]).transpose(0,1)
            T = dset.search_track(frame_info[:,0].int().cpu().numpy(),meta_item[2]-args.pre_len+1,meta_item[2])

            loss = criterion(T[:,:,1:],out)
            # target = target_Tracks[:,1:3]
            
            #print(target_Tracks)
            # loss = criterion(target,out[frame_info[:,0]==meta_item[0],:,:2][0])
            loss.backward()
        print('epoch:{},batch:{},loss:{},time:{}'.format(epoch_num,batch_num,loss,time.ctime()))
        optimizer.step()
    return net

def valid(net,valid_loader,criterion,dset,args):
    error = torch.tensor([])
    net.eval()
    with torch.no_grad():
        for _,batched_one_meta in enumerate(valid_loader):
            meta_item = batched_one_meta[0]
            
            # forward
            frame_info = dset.frame(meta_item[1]+args.his_len-1)
            Tracks = dset.search_track(frame_info[:,0].int().cpu().numpy(),meta_item[1],meta_item[1]+args.his_len-1)
            out = net(Tracks)
            target_Tracks = dset.search_track(meta_item[0],meta_item[2]-args.pre_len+1,meta_item[2]).transpose(0,1)
            target = target_Tracks[:,1:3]

            loss = criterion(target[-1],out[frame_info[:,0]==meta_item[0],-1,:2][0])

            loss_tensor = torch.tensor([loss.item()])
            error = torch.concat((loss_tensor,error))
        epoch_error = error.mean()
    return torch.tensor([epoch_error])