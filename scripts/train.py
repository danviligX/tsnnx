import torch.nn as nn
import pickle
import optuna
import torch

import os
import sys
sys.path.append(os.getcwd())

from models.vtp_Flinear.utils import obj_vtp_Flinear,test,Dset,vtp_Flinear
from libx.dataio import vtp_dataloader

def main():
    model_name = 'vtp_Flinear'
    save_path = ''.join(['./models/',model_name,'/',model_name,'_study.pkl'])

    trial = optuna_study(save_dic=save_path,study_name=model_name,trial_num=30)
    
    dic_path = ''.join(['./models/',model_name,'/trial/trial_',str(trial.number),'.mdic'])
    args_path = ''.join(['./models/',model_name,'/trial/args_',str(trial.number),'.marg'])
    
    error = model_eval(args_dic=args_path,net_dic=dic_path)
    print(error)

def optuna_study(save_dic='./study.pkl',trial_num=5,study_name='study'):
    study = optuna.create_study(direction='minimize',study_name=study_name)
    study.optimize(obj_vtp_Flinear,n_trials=trial_num,n_jobs=3)
    with open(save_dic,'wb') as path:
        pickle.dump(study,path)
        path.close()
    
    return study.best_trial

def model_eval(args_dic,net_dic):
    args = torch.load(args_dic)
    net_state = torch.load(net_dic)
    
    net = vtp_Flinear(args=args).to(args.device)
    net.load_state_dict(net_state)

    criterion = nn.MSELoss()
    with open(args.dd_index_path,'rb') as path:
        dd_index = pickle.load(path)
        path.close()
    dset = Dset(args.set_path,args.device)
    test_loader = vtp_dataloader(test_item_idx=dd_index[2])

    error_list = test(net,test_loader,criterion,dset,args)

    error_table = torch.zeros(3)
    error_table[0] = error_list.mean()
    error_table[1] = error_list.std()
    error_table[2] = error_list.max()

    return error_table

if __name__=='__main__':
    main()