import torch.nn as nn
import pickle
import optuna
import torch

import os
import sys
sys.path.append(os.getcwd())

from libx.model import net_object,test,Dset
from libx.dataio import vtp_dataloader

def main():
    model_name = 'VTPnet'
    save_path = ''.join(['./result/',model_name,'/',model_name,'_study.pkl'])

    trial = optuna_study(save_dic=save_path,study_name=model_name,trial_num=10)
    
    dic_path = ''.join(['./result/',model_name,'/trial/trial_',str(trial.number),'.model'])
    args_path = ''.join(['./result/',model_name,'/trial/args_',str(trial.number),'.miarg'])
    
    error = model_eval(args_dic=args_path,net_dic=dic_path)
    print(error)

def optuna_study(save_dic='./study.pkl',trial_num=5,study_name='study'):
    study = optuna.create_study(direction='minimize',study_name=study_name)
    study.optimize(net_object,n_trials=trial_num)
    with open(save_dic,'wb') as path:
        pickle.dump(study,path)
        path.close()
    
    return study.best_trial

def model_eval(args_dic,net_dic):
    args = torch.load(args_dic)
    net_state = torch.load(net_dic)
    
    net = net_object(args=args).to(args.device)
    net.load_state_dict(net_state)

    criterion = nn.MSELoss()
    with open(args.dd_index_path,'rb') as path:
        dd_index = pickle.load(path)
        path.close()
    dset = Dset(args.set_path)
    test_loader = vtp_dataloader(test_item_idx=dd_index[2])

    error_table = test(net,test_loader,criterion,dset,args)

    return error_table

if __name__=='__main__':
    main()