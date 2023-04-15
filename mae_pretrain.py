import os
import argparse
import math
import time
import itertools
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip
from tqdm import tqdm
import numpy as np
import copy

from model import *
import model as Model
from utils import setup_seed,CustomImageNet,save_all,collect_features,KNN_eval
from dataset import load_dataset

from torchvision import transforms as T
import kornia.augmentation as K
from transforms import MultiView, RandomResizedCrop, ColorJitter, GaussianBlur, RandomRotation, MultiView_oneside
from AugSelf_model import prepare_training_batch, load_ss_predictor, SSObjective

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)#400, 2000
    parser.add_argument('--warmup_epoch', type=int, default=200)#40, 200
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--data_path', type=str, default='/mnt/mlx-nfs/dongyounggo/CL_research/DCL_variation/data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--AugSelf',  action='store_true')
    parser.add_argument('--AugSelf_mode', type=str, default='MLP_light')
    parser.add_argument('--ss-crop',  type=float, default=-1)
    parser.add_argument('--ss-color', type=float, default=-1)
    parser.add_argument('--ss-flip',  type=float, default=-1)
    parser.add_argument('--ss-blur',  type=float, default=-1)
    parser.add_argument('--ss-rot',   type=float, default=-1)
    parser.add_argument('--ss-gray',   type=float, default=-1)
    parser.add_argument('--ss-sol',   type=float, default=-1)
    parser.add_argument('--ss_sim_preserve',   type=float, default=-1)
    parser.add_argument('--ss_sim_preserve_MSE',   type=float, default=-1)
    parser.add_argument('--ss_shuffle_idx',   type=float, default=-1)
    parser.add_argument('--direct_sim_preserve',  action='store_true')
    parser.add_argument('--preserve_all_feature',   action='store_true')
    parser.add_argument('--pool_feature_wocls',   action='store_true')  
    parser.add_argument('--pool_feature_wocls2cls',   action='store_true')  
    parser.add_argument('--mse_direct_sim_preserve',  action='store_true')
    parser.add_argument('--cls2preserve',  action='store_true')
    parser.add_argument('--CL_preserve',  action='store_true')
    parser.add_argument('--testing_const',   type=float, default=1)
    parser.add_argument('--preserve_mask_ratio',   type=float, default=0)
    parser.add_argument('--position_mask_ratio',   type=float, default=0)
    parser.add_argument('--fixed_position_mask',   action='store_true') 

    parser.add_argument('--each',   action='store_true') ## for load_ss_predictor for each img
    parser.add_argument('--eval_freq', type=int, default=50) 
    parser.add_argument('--save_freq', type=int, default=50)     
    parser.add_argument('--second_cls_token',   action='store_true')  
    parser.add_argument('--third_cls_token',   action='store_true')
    parser.add_argument('--concat_second_cls_token',   action='store_true') 
    parser.add_argument('--aux_task', type=str)
    
    parser.add_argument('--emb_dim', type=int, default=192)
    parser.add_argument('--load_path', type=str, default='None')
    parser.add_argument('--lam',   type=float, default=1.0)
    parser.add_argument('--annealing',   action='store_true')  
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--model_class', type=str, default='MAE_ViT') # ['MAE_ViT','hf_MAE_ViT']
    parser.add_argument('--mid_save', type=int, default=1)
    parser.add_argument('--CAN',   action='store_true')  
    parser.add_argument('--mean_pool_wo_cls',   action='store_true')  

    args = parser.parse_args()
    setup_seed(args.seed)
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    dataset_loaded = load_dataset(args)
    train_dataset,val_dataset = dataset_loaded['train_dataset'],dataset_loaded['val_dataset']
    t1,t2 = dataset_loaded['t1'],dataset_loaded['t2']
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', args.dataset, 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_class = getattr(Model,args.model_class)
    print('model_class',model_class)
    model = model_class(**vars(args)).to(device)
    ss_predictor = None
    if (args.AugSelf)|(args.each):
        ss_objective = SSObjective(
            crop  = args.ss_crop,
            color = args.ss_color,
            flip  = args.ss_flip,
            gray  = args.ss_gray,
            blur  = args.ss_blur,
            rot   = args.ss_rot,
            sol   = args.ss_sol,
            only  = False, ## we don't use this option for MAE research
            **vars(args)
        )        
        ## Since features in ViT_Encoder is (t b c), mlp needs to be (b (t:emd_dim, c:w h))
        assert args.AugSelf_mode=='MLP_light'
        ss_predictor_emb = int(model.emb_dim)
            
        if args.second_cls_token & args.concat_second_cls_token:
            ss_predictor_emb*=2   
        elif args.third_cls_token & args.concat_second_cls_token:
            ss_predictor_emb*=3
        print(f'ss_predictor_emb : {ss_predictor_emb}')
        ss_predictor = load_ss_predictor(ss_predictor_emb, ss_objective, **vars(args))
        ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])
    else:
        ss_params = []
        
    if args.load_path != 'None':
        print(f'model load from {args.load_path}')
        model_ckpt = torch.load(args.load_path)
        if type(model_ckpt)==dict:
            print('loaded with load_state')
            model.load_state_dict(model_ckpt['model']) # When torch.save({'model':model.state_dict()}, args.model_path)
        else:
            print('loaded with entire model')
            model = model_ckpt # When torch.save(model, args.model_path)
        try:
            ss_load_path = args.load_path+'.ss.logs'
            ss_ckpt = torch.load(ss_load_path)
            for ss_k in ss_ckpt.keys():
                ss_predictor[ss_k].load_state_dict(ss_ckpt[ss_k])
            ss_params = sum([list(v.parameters()) for v in ss_predictor.values()], [])
            print(f"ss load from {ss_load_path}")
        except:
            print(f"ss is not loaded")
            pass        
    optim = torch.optim.AdamW(list(model.parameters())+ss_params, lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    ## cosine scheduler
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
    losses_list = []
    e_st = -1
    to_mid_save = True if args.mid_save>1 else False

    if args.load_path != 'None':
        ckpt_load_path = args.load_path+'.logs'
        print(f"ckpt load from {ckpt_load_path}")
        ckpt = torch.load(ckpt_load_path)
        losses_list = ckpt['losses_list']
        optim.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])        
        e_st = ckpt['e']
        print(f'Start with ep {e_st}')
        
    step_count = 0
    best_eval = 0
    optim.zero_grad()
    CEloss = torch.nn.CrossEntropyLoss()
    training_st_time = time.time()
    for e in range(e_st+1,args.total_epoch):
        model.train()
        losses = {'loss':[],'loss1':[],'loss2':[],'ss_loss':[],'ss_loss2':[],'CL_loss':[],'D_loss':[],'ep_time':[],'it_time':[],'enc_time':[],'dec_time':[],'tr_time':[]}
        epoch_st_time = time.time()
        for img, label in tqdm(iter(dataloader)):
            iter_st_time = time.time()
            step_count += 1
            if args.AugSelf:
                img1, img2, d1, d2 = prepare_training_batch((img, label), t1, t2, device)
                loss1, emb_feature1, mid_feature1l, mid_feature1f = model(img1)
                loss2, emb_feature2, mid_feature2l, mid_feature2f = model(img2)
                loss = (loss1+loss2).mul(0.5)

                ## calculate auxilary loss
                mid_feature1,mid_feature2 = mid_feature1l,mid_feature2l
                ss_losses = ss_objective(ss_predictor, mid_feature1, mid_feature2, d1, d2)
                loss = loss+ss_losses['total']
                
                D_loss = 0 # D_loss for denoise_loss
                if args.CAN:
                    D_loss = (mid_feature1f+mid_feature2f)/2*0.5
                    
                ## calculate CL loss
                CL_loss = 0
                if args.aux_task=='simclr':
                    logits = torch.mm(emb_feature1,emb_feature2.T)
                    labels = torch.tensor([x for x in range(logits.shape[0])]).to(device)
                    CL_loss = F.cross_entropy(logits, labels)   
                elif args.aux_task=='simsiam':
                    z1,p1 = emb_feature1
                    z2,p2 = emb_feature2
                    loss1_ = F.cosine_similarity(p1, z2.detach(), dim=-1).mean().mul(-1)
                    loss2_ = F.cosine_similarity(p2, z1.detach(), dim=-1).mean().mul(-1)
                    CL_loss = (loss1_+loss2_).mul(0.5)
                elif args.aux_task=='barlowtwin':
                    def off_diagonal(x):
                        # return a flattened view of the off-diagonal elements of a square matrix
                        n, m = x.shape
                        assert n == m
                        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()                    
                    N = emb_feature1.size(0)
                    corr = torch.mm(emb_feature1.T,emb_feature2)/N
                    on_diag = torch.diagonal(corr).add_(-1).pow_(2).mean()
                    off_diag = off_diagonal(corr).pow_(2).mean()
                    # lambd = 0.005 ## hyper parameter for barlow twin! 
                    lambd = 10 ## To match the scale for the case of mean, 10*mean is similar with 0.005*sum, given dim=2048
                    CL_loss = on_diag + lambd * off_diag    
                elif args.aux_task is None:
                    pass
                else:
                    print(f'{args.aux_task} is not implemented yet')
                    raise NotImplementedError                
                annealing_c = 1-(e/args.total_epoch) if args.annealing else 1
                loss = loss+(annealing_c*args.lam)*CL_loss
                loss = loss + D_loss
                losses['loss'].append(loss.item())
                losses['loss1'].append(loss1.item())
                losses['loss2'].append(loss2.item())
                losses['D_loss'].append(D_loss.item() if type(D_loss)!=int else D_loss)
                losses['CL_loss'].append(CL_loss.item() if type(CL_loss)!=int else CL_loss)
                losses['ss_loss'].append(ss_losses['total'].item() if type(ss_losses['total'])!=int else ss_losses['total'])
            else:
                img, _ = img # drop DA parameter
                img = t1(img.to(device)).detach() if t1 is not None else img ## For case of DA including more than simple Normalize
                img = img.to(device)
                loss, _, mid_feature, mid_featuref = model(img)
                losses['loss'].append(loss.item())
                ## Calculate loss only for masked patches
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            iter_ed_time = time.time()
            losses['it_time'].append(iter_ed_time-iter_st_time)
        epoch_ed_time = time.time()
        losses['ep_time'].append(epoch_ed_time-epoch_st_time)
        lr_scheduler.step()
        losses = {k_:sum(v_)/len(v_) for k_,v_ in losses.items() if len(v_)!=0}
        avg_loss = losses['loss']
        writer.add_scalar('mae_loss', avg_loss, global_step=e)

        if (e % args.eval_freq==0)|(e==(args.total_epoch-1)):
            print('run eval')
            eval_st_time = time.time()
            model.eval()
            backbone = lambda x:model(x.to(device))[2] ## this should get [cls,emb] features, e.g. [512,192]
            with torch.no_grad():
                features, labels = collect_features(backbone, dataloader, device,t1, t2, args)
                corrects = KNN_eval(backbone,testloader,device,features,labels)
                losses['knn'] = corrects
            eval_ed_time = time.time()
            losses['ev_time']=eval_ed_time-eval_st_time
            training_ed_time = time.time()
            losses['tr_time']=training_ed_time-training_st_time
        print(f'In epoch {e}, average traning loss is {losses}.')
        losses_list.append(losses)

        if (e % args.save_freq==0)|(e==(args.total_epoch-1)):
            knn_k = max(losses['knn'].keys()) # knn_k==20 for accuracy_knn, knn_k=5 for dino_knn
            if (best_eval<losses['knn'][knn_k]):
                ## Let's use previous 5 acc for early stop criteria
                # best_eval = losses['knn'][knn_k]
                best_eval = np.mean([x['knn'][knn_k] for x in losses_list if 'knn' in x][-5:])
                save_all(model,args,ss_predictor,losses_list,optim,lr_scheduler,e)
                print(f'Model saved to {args.model_path}')
                if (to_mid_save)&(e>=args.mid_save):
                    args_mid = copy.deepcopy(args)
                    args_mid.model_path += f'.mid_save{e}'
                    save_all(model,args_mid,ss_predictor,losses_list,optim,lr_scheduler,e)
                    to_mid_save = False # one time operation
                    del args_mid
            else:
                print(f'Worse than Best eval, preserving {args.model_path}')
            if (e==(args.total_epoch-1)):
                args.model_path = args.model_path.replace('.pt','')+'.last.pt'  
                save_all(model,args,ss_predictor,losses_list,optim,lr_scheduler,e)
                print(f'Model at last epoch is saved to {args.model_path}')
                
