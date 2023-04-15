import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, CenterCrop
from torchvision import transforms as T
import kornia.augmentation as K
from PIL import Image

from transforms import MultiView, RandomResizedCrop, ColorJitter, GaussianBlur, RandomRotation, MultiView_oneside
from utils import CustomImageNet, Flowers102, CUB

def load_downstream_dataset(args):
    ## Define transform used in Pretrain 
    if args.pretrain_dataset =='cifar100':
        mean = torch.tensor([0.5071, 0.4865, 0.4409])
        std  = torch.tensor([0.2009, 0.1984, 0.2023])
        if args.add_DA:
            print('add DA!')
            train_transform = Compose([
                T.RandomResizedCrop(32,interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                ToTensor(), Normalize(mean, std)])    
        else:
            train_transform = Compose([Resize(32,interpolation=Image.BICUBIC),CenterCrop(32),ToTensor(),Normalize(mean, std)])
        val_transform = Compose([Resize(32,interpolation=Image.BICUBIC),CenterCrop(32),ToTensor(),Normalize(mean, std)])
    elif args.pretrain_dataset=='imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        train_transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean, std)])
        val_transform = T.Compose([T.Resize(224, interpolation=Image.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean, std)])

    ## Load Downstream dataset with pretrain_transform    
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(args.data_path, train=True, download=False, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(args.data_path, train=False, download=False, transform=val_transform)
        num_classes = 10
    elif 'cifar100' in args.dataset:
        train_dataset = torchvision.datasets.CIFAR100(args.data_path, train=True, download=False, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(args.data_path, train=False, download=False, transform=val_transform)
        num_classes = 100        
    elif args.dataset=='imagenet':
        from datasets import load_dataset
        print('Loading imagenet from cache...')
        data = load_dataset("imagenet-1k", cache_dir='/mnt/mlx-nfs/dongyounggo/message_generation/dataset')        
        train_dataset = CustomImageNet(data['train'],  transform=train_transform)
        val_dataset   = CustomImageNet(data['test'], transform=val_transform)
        num_classes = 1000
    elif args.dataset == 'flowers':                     
        train    = Flowers102(root=args.data_path, split='train', transform = train_transform)
        val      = Flowers102(root=args.data_path, split='val', transform = train_transform)
        train_dataset = ConcatDataset([train, val])
        val_dataset     = Flowers102(root=args.data_path, split='test', transform = val_transform)
        num_classes = 102
    elif args.dataset == 'cub200':
        ROOT = '/mnt/mlx-nfs/dongyounggo/message_generation/dataset/CUB200/CUB_200_2011/'
        train = CUB(ROOT, 'train', transform=train_transform)
        val = CUB(ROOT, 'valid', transform=train_transform)
        train_dataset = ConcatDataset([train, val])
        val_dataset = CUB(ROOT, 'test', transform=val_transform)
        num_classes = 200
        
    return {'train_dataset':train_dataset,'val_dataset':val_dataset,'num_classes':num_classes}


def load_dataset(args):
    t1 = None
    t2 = None
    if args.dataset == 'cifar10':
        '''
        Just same with original repo
        '''
        train_dataset = torchvision.datasets.CIFAR10(args.data_path, train=True, download=False, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR10(args.data_path, train=False, download=False, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    elif 'cifar100' in args.dataset:
        mean = torch.tensor([0.5071, 0.4865, 0.4409])
        std  = torch.tensor([0.2009, 0.1984, 0.2023])
        train_transform = MultiView(RandomResizedCrop(32, scale=(0.2, 1.0)),num_views=2 if args.AugSelf else 1)
        test_transform = Compose([T.Resize(32),T.CenterCrop(32),T.ToTensor(),T.Normalize(mean, std)])
        augmentations = []
        if args.dataset == 'cifar100_CropFlip':
            augmentations.append(K.RandomHorizontalFlip())
        elif args.dataset == 'cifar100_CropFlipColor':
            s = 0.5
            augmentations.append(K.RandomHorizontalFlip())
            augmentations.append(ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8))
        elif args.dataset == 'cifar100_CL':
            s = 0.5
            s_k = 1/3
            augmentations.append(K.RandomHorizontalFlip())
            augmentations.append(ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8))
            augmentations.append(K.RandomGrayscale(p=0.2))
            augmentations.append(GaussianBlur(int(9*s_k), (0.1, 2.0)))
        elif args.dataset == 'cifar100_AugSelf':
            s = 1.0
            augmentations.append(K.RandomHorizontalFlip())
            augmentations.append(ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8))
            augmentations.append(K.RandomGrayscale(p=0.2))
            augmentations.append(GaussianBlur(9, (0.1, 2.0)))
        else: ##args.dataset == 'cifar100':
            pass ## only Normalize
        augmentations.append(K.Normalize(mean, std))
        t1 = torch.nn.Sequential(*augmentations)
        t2 = torch.nn.Sequential(*augmentations)

        train_dataset = torchvision.datasets.CIFAR100(args.data_path, train=True, download=False, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(args.data_path, train=False, download=False, transform=test_transform)        
    elif 'imagenet' in args.dataset:
        s = 0.5 
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        train_transform = MultiView(RandomResizedCrop(224, scale=(0.2, 1.0)),num_views=2 if args.AugSelf else 1)
        test_transform = T.Compose([T.Resize(224),
                                    T.CenterCrop(224),
                                    T.ToTensor(),
                                    T.Normalize(mean, std)])
        if args.dataset == 'imagenet_CropFlip':
            t1 = torch.nn.Sequential(K.RandomHorizontalFlip(),
                            K.Normalize(mean, std))
            t2 = torch.nn.Sequential(K.RandomHorizontalFlip(),
                            K.Normalize(mean, std))
        elif args.dataset == 'imagenet_CropFlipColor':
            t1 = torch.nn.Sequential(K.RandomHorizontalFlip(),
                            ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8),
                            K.Normalize(mean, std))
            t2 = torch.nn.Sequential(K.RandomHorizontalFlip(),
                            ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8),
                            K.Normalize(mean, std))
        elif args.dataset == 'imagenet_all':
            t1 = torch.nn.Sequential(K.RandomHorizontalFlip(),
                            ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8),
                            K.RandomGrayscale(p=0.2),
                            GaussianBlur(23, (0.1, 2.0)),
                            K.Normalize(mean, std))
            t2 = torch.nn.Sequential(K.RandomHorizontalFlip(),
                            ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s, p=0.8),
                            K.RandomGrayscale(p=0.2),
                            GaussianBlur(23, (0.1, 2.0)),
                            K.Normalize(mean, std))
        
        print('Loading imagenet from cache...')
        ## if we use imagenet-100 from raw data
        data_path = "/mnt/mlx-nfs/dongyounggo/message_generation/dataset/1N/"
        train_dataset = ImageFolder(data_path+'train',train_transform)
        val_dataset = ImageFolder(data_path+'val',test_transform)
        ## if we use imagenet-1k from huggingface
        # from datasets import load_dataset
        # data = load_dataset("imagenet-1k", cache_dir='/mnt/mlx-nfs/dongyounggo/message_generation/dataset')        
        # train_dataset = CustomImageNet(data['train'],  transform=train_transform)
        # val_dataset   = CustomImageNet(data['validation'], transform=test_transform)
        # test_dataset  = CustomImageNet(data['test'], transform=test_transform)
        # del data
        
    return {'train_dataset':train_dataset,"val_dataset":val_dataset,'t1':t1,'t2':t2}