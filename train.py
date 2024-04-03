import torch
import torch.nn as nn
from libx.dataio import Args
from models.Field_nn.utils import highD, ff_net
from torch.nn.parallel import DataParallel 

def main():
    args = Args()
    args.name = 'Field_nn'
    args.dt = 0.2
    args.opt = 'Adam'
    args.lr = 0.001
    args.batch_size = 30
    args.epoch_num = 700
    args.state_dic_path = './models/Field_nn/trial/'
    args.device_ids = list(range(torch.cuda.device_count()))  # 25-108,20min;13-78,20-40

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
        print('epoch:{}, error:{}'.format(epoch,error.item()))
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
        if batch_count==args.batch_size:
            batch_count = 0
            opt.step()
            opt.zero_grad()
        
        batch_count = batch_count + 1
        sel_fid = (epoch+batch_count)%39
        frame_in = item[sel_fid]
        frame_out = item[sel_fid+1]
        pos_rel = frame_out[0].to(args.device_ids[0])

        net_out = net(frame_in)
        pos_pre = net_out*dt*dt/2 + frame_in[1].to(args.device_ids[0])*dt
        loss = criterion(pos_pre,pos_rel)
        loss.backward()
        # print('epoch:{},loss:{}'.format(epoch,loss.item()))
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
            pos_pre = net_out*dt*dt/2 + frame_in[1].to(args.device_ids[0])*dt
            loss = criterion(pos_pre,pos_rel)
            error.append(loss.item())

    return torch.tensor(error).mean()

def test(args):
    data = args.data.valid_data
    net = args.net
    net.eval()
    pass

if __name__=='__main__': main()
