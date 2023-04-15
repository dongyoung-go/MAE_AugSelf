import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from model import *
from utils import setup_seed, Flowers102

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

    args = parser.parse_args()
    assert '.pt' in args.output_model_path
    args.output_model_path = args.output_model_path[:-3] + str(args.base_learning_rate) + '.pt'
    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(args.data_path, train=True, download=False, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR10(args.data_path, train=False, download=False, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean=[0.5071, 0.4865, 0.4409]
        std=[0.2009, 0.1984, 0.2023]
        train_dataset = torchvision.datasets.CIFAR100(args.data_path, train=True, download=False, transform=Compose([ToTensor(), Normalize(mean, std)]))
        val_dataset = torchvision.datasets.CIFAR100(args.data_path, train=False, download=False, transform=Compose([ToTensor(), Normalize(mean, std)]))
        num_classes = 100
    elif args.dataset == 'flowers':
        '''
        Assume we used cifar100 for pretraining
        '''
        # assert pretrain_data == 'cifar100_for_MAE':
        mean = torch.tensor([0.5071, 0.4865, 0.4409])
        std  = torch.tensor([0.2009, 0.1984, 0.2023])
        transform = Compose([Resize(32),
                            CenterCrop(32),
                            ToTensor(),
                            Normalize(mean, std)])   
        train    = Flowers102(root=args.data_path, split='train', transform = transform)
        val      = Flowers102(root=args.data_path, split='val', transform = transform)
        train_dataset = ConcatDataset([train, val])
        val_dataset     = Flowers102(root=args.data_path, split='test', transform = transform)
        num_classes = 102
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu')
        writer = SummaryWriter(os.path.join('logs', args.dataset, 'pretrain-cls'))
    else:
        model = MAE_ViT()
        writer = SummaryWriter(os.path.join('logs', args.dataset, 'scratch-cls'))
    model = ViT_Classifier(model.encoder, num_classes=num_classes, **vars(args)).to(device)

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
        if args.get('linear_probing',False):
            model.eval()
            model.encoder.eval()
            model.head.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
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
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
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
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
            # torch.save(model, args.output_model_path)
            torch.save({'model':model.state_dict(),
            'best_val_acc':best_val_acc,
            'train_acc_list':train_acc_list,
            'train_loss_list':train_loss_list,
            'val_acc_list':val_acc_list,
            'val_loss_list':val_loss_list,}, args.output_model_path)

        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train' : avg_train_acc, 'val' : avg_val_acc}, global_step=e)