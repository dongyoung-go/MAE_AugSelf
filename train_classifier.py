import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop
import torchvision.transforms as transforms

from tqdm import tqdm
from transformers import ViTForImageClassification

from model import *
from utils import setup_seed
from dataset import load_downstream_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)#1024,128
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--use_other_cls', type=str, default=False)
    parser.add_argument('--use_mean_features',   action='store_true')    
    parser.add_argument('--use_mean_features_official',   action='store_true')    
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-from_scratch.pt')
    parser.add_argument('--data_path', type=str, default='/mnt/mlx-nfs/dongyounggo/CL_research/DCL_variation/data')  
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--two_layer',   action='store_true')  
    parser.add_argument('--two_layer_from_AugSelf',   action='store_true') 
    parser.add_argument('--direct_from_encoder',   action='store_true') 
    parser.add_argument('--add_DA',   action='store_true') 
    parser.add_argument('--linear_probe',   action='store_true') 
    parser.add_argument('--load_linear_prob_initialization', type=str, default=False)      
    parser.add_argument('--model_class', type=str, default='MAE_ViT') # ['MAE_ViT','hf_MAE_ViT']
    parser.add_argument('--pretrain_dataset', type=str, default='cifar100')

    args = parser.parse_args()
    assert '.pt' in args.output_model_path
    args.output_model_path = args.output_model_path[:-3] + str(args.base_learning_rate) + '.pt'
    setup_seed(args.seed)
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    dataset_loaded = load_downstream_dataset(args)
    train_dataset,val_dataset,num_classes = dataset_loaded['train_dataset'],dataset_loaded['val_dataset'],dataset_loaded['num_classes']
    print(f'train_dataset: {len(train_dataset)}')
    print(f'val_dataset: {len(val_dataset)}')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu')
        if args.model_class=='MAE_ViT':
            model = ViT_Classifier(model.encoder, num_classes=num_classes, **vars(args)).to(device)
        elif args.model_class=='hf_MAE_ViT':
            model = torch.load(args.pretrained_model_path, map_location='cpu')
            args.pretrained_model_path_hf = args.pretrained_model_path+'_hf'
            model.model.save_pretrained(args.pretrained_model_path_hf) ## save only the HF_model part since only this will be used for downstream classifier            
            model = ViTForImageClassification.from_pretrained(args.pretrained_model_path_hf,num_labels=num_classes).to(device)        
        else:
            raise NotImplementedError
    else:
        model = MAE_ViT()
        model = ViT_Classifier(model.encoder, num_classes=num_classes, **vars(args)).to(device)
    
    if args.load_linear_prob_initialization:
        ckpt = torch.load(args.load_linear_prob_initialization)
        model.load_state_dict(ckpt['model'])
        print(f'Initialization loaded from {args.load_linear_prob_initialization}')
        del ckpt
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'# of params: {pytorch_total_params}') 
    if args.linear_probe:
        print('Linear probe, Freeze encoder...')
        for name, param in model.named_parameters():
            if 'classifier' not in name: ## unified classification layer from self.head to self.classifier
                param.requires_grad = False
    # Count the number of learnable parameters
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# of learnable parameters: {num_learnable_params}")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader),desc='Training'):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            logits = logits.logits if hasattr(logits,'logits') else logits ## To match with hf_MAE_ViT
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        train_acc_list.append(avg_train_acc)
        train_loss_list.append(avg_train_loss)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader),desc='Validation'):
                img,label = img.to(device),label.to(device)
                logits = model(img)
                logits = logits.logits if hasattr(logits,'logits') else logits ## To match with hf_MAE_ViT
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            val_acc_list.append(avg_val_acc)
            val_loss_list.append(avg_val_loss)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch, {args.output_model_path}')
            torch.save({'model':model.state_dict(),
            'best_val_acc':best_val_acc,
            'train_acc_list':train_acc_list,
            'train_loss_list':train_loss_list,
            'val_acc_list':val_acc_list,
            'val_loss_list':val_loss_list,}, args.output_model_path)