import torch
import os
import torch.nn as nn
from libx.dataio import Args
from models.Field_nn.utils import highD, ff_net
from torch.nn.parallel import DataParallel 

def main():
    args = Args()
    args.name = 'Field_nn'
    args.dt = 0.2
    args.opt = 'Adam'
    args.lr = 1e-4
    args.batch_size = 50
    args.epoch_num = 40*300
    args.state_dic_path = './models/Field_nn/trial_4/'
    if os.path.exists(args.state_dic_path) is False: os.mkdir(args.state_dic_path)
    args.device_ids = list(range(torch.cuda.device_count()))

    highD_data = highD(cache = True,device=args.device_ids[0])
    print('Generate Dset')
    highD_data.gen_dset()
    highD_data.gen_ItemMeta()
    highD_data.gen_data_iform()
    print('Generate DataLoader')
    highD_data.gen_dataloader()

    net = ff_net()
    opt = getattr(torch.optim, args.opt)(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    net = DataParallel(net, device_ids=args.device_ids).to(args.device_ids[0])  

    args.opt = opt
    args.criterion = criterion
    args.net = net
    args.data = highD_data

    for epoch in range(args.epoch_num):
        args.net = train(args,epoch)
        error = valid(args,epoch)
        print('NUM:{}, epoch:{}, error:{}, max:{}, min:{}'.format(epoch, epoch%39,error.mean().item(),error.max().item(),error.min().item()))
        torch.save(args.net.state_dict(),args.state_dic_path+'_'+str(epoch)+".mdic")
    # acc = test()
    # print(acc)

def train(args,epoch):
    data = args.data.train_data
    net = args.net
    opt = args.opt
    criterion = args.criterion
    dt = args.dt

    net.train()
    opt.zero_grad()
    batch_count = 0

    for item in data:
        # if batch_count==args.batch_size:
        #     batch_count = 0
        #     opt.step()
        #     opt.zero_grad()
        
        # batch_count = batch_count + 1
        sel_fid = (epoch+batch_count)%39
        frame_in = item[sel_fid]
        frame_out = item[sel_fid+1]
        pos_rel = frame_out[0].to(args.device_ids[0])
        vel_rel = frame_out[1].to(args.device_ids[0])

        net_out = net(frame_in)
        pos_pre = frame_in[0] + net_out*dt*dt/2 + frame_in[1].to(args.device_ids[0])*dt
        vel_pre = frame_in[1] + net_out*dt
        tpr = pos_rel*args.data.norm_value_scaler[:2].cuda()+args.data.norm_value_center[:2].cuda()
        tpp = pos_pre*args.data.norm_value_scaler[:2].cuda()+args.data.norm_value_center[:2].cuda()
        tvr = vel_rel*args.data.norm_value_scaler[2:].cuda()+args.data.norm_value_center[2:].cuda()
        tvp = vel_pre*args.data.norm_value_scaler[2:].cuda()+args.data.norm_value_center[2:].cuda()
        loss = criterion(tpp,tpr) + criterion(tvr,tvp)
        loss.backward()
        # print('epoch:{},loss:{}'.format(epoch,loss.item()))
    opt.step()
    opt.zero_grad()
    return net

def valid(args,epoch):
    error = []
    data = args.data.valid_data
    criterion = args.criterion
    dt = args.dt
    net = args.net
    net.eval()
    with torch.no_grad():
        for item in data:
            sel_fid = epoch%39
            frame_in = item[sel_fid]
            frame_out = item[sel_fid+1]
            pos_rel = frame_out[0].to(args.device_ids[0])

            net_out = net(frame_in)
            pos_pre = frame_in[0] + net_out*dt*dt/2 + frame_in[1].to(args.device_ids[0])*dt
            tpr = pos_rel*args.data.norm_value_scaler[:2].cuda()+args.data.norm_value_center[:2].cuda()
            tpp = pos_pre*args.data.norm_value_scaler[:2].cuda()+args.data.norm_value_center[:2].cuda()
            loss = criterion(tpp,tpr)
            error.append(loss.item())

    return torch.tensor(error)

def test(args):
    data = args.data.valid_data
    net = args.net
    net.eval()
    pass

if __name__=='__main__': main()
