import random
import torch
import numpy as np

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def save_all(model,args,ss_predictor,losses_list,optim,lr_scheduler,e):
    ''' save model '''
    # torch.save({'model':model.state_dict()}, args.model_path)
    torch.save(model, args.model_path)
    try:
        torch.save({ss_k:ss_v.state_dict() for ss_k,ss_v in ss_predictor.items()}, args.model_path+'.ss.logs')
    except:pass
    torch.save({'losses_list':losses_list,
                'optimizer':optim.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'e':e,
                }, args.model_path+'.logs')
        
from AugSelf_model import prepare_training_batch        
import torch.nn.functional as F
def collect_features(backbone,
                     dataloader,
                     device,
                     t1, t2, args,
                     normalize=True,
                     verbose=False):
    with torch.no_grad():
        features = []
        labels   = []
        for i, (x, y) in enumerate(dataloader): ## x should be img. Since we used MultiView in kornia, additional indexing is needed
            if args.AugSelf:
                x, _, _, _ = prepare_training_batch((x, y), t1, t2, device)
            else:
                # drop DA parameter
                img = x[0] if len(x)==2 else x        
                img = t1(img.to(device)).detach() if t1 is not None else img ## For case of DA including more than simple Normalize
                x = img
            if x.ndim == 5:
                _, n, c, h, w = x.shape ## [512, 3, 32, 32]
                x = x.view(-1, c, h, w)
                y = y.view(-1, 1).repeat(1, n).view(-1)
            z = backbone(x.to(device))
            if normalize:
                z = F.normalize(z, dim=-1)
            features.append(z.to(device).detach())
            labels.append(y.to(device).detach())
            if verbose and (i+1) % 10 == 0:
                print(i+1)
        features = torch.cat(features, 0).detach()
        labels   = torch.cat(labels, 0).detach()

    return features, labels        

def KNN_eval(backbone,testloader,device,features,labels,knn_type='dino_knn'):
    if knn_type=='accuracy_knn':
        '''
        preds=True if at least one of neighbors within k has same class (i.e. highly coarse evaluation)
        '''
        topk = [1,5,10,20]
        maxk = max(topk)
        corrects = {k_:0 for k_ in topk}
        total = 0
        for x, y in testloader:
            z = F.normalize(backbone(x.to(device)), dim=-1)
            scores = torch.einsum('ik, jk -> ij', z, features)
            _, pred = scores.topk(maxk, 1, True, True)
            pred = labels[pred]
            correct = pred.eq(y.to(device).view(-1, 1).expand_as(pred))
            preds = labels[scores.argmax(1)]
            
            res = dict()
            for k in topk:
                correct_k = (correct[:,:k].sum(dim=-1)>0).float().sum().item()
                res[k] = correct_k
            total += y.shape[0]
            corrects = {k_:v_+res[k_] for k_, v_ in corrects.items()}
        corrects = {k_:v_/total for k_, v_ in corrects.items()}                
        return corrects
    elif knn_type=='dino_knn':
        '''
        standard knn
        '''
        num_classes = labels.unique().size(0)
        k=20
        T=0.07
        retrieval_one_hot = torch.zeros(k, num_classes).to(features.device)
        top1,top5,total = 0.0,0.0,0
        for x, y in testloader:
            z = F.normalize(backbone(x.to(device)), dim=-1)

            # calculate the dot product and compute top-k neighbors
            load_batch_size_ = y.size(0)
            similarity = torch.mm(z, features.T)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
            candidates = labels.view(1, -1).expand(load_batch_size_, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(load_batch_size_ * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(T).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(load_batch_size_, -1, num_classes),
                    distances_transform.view(load_batch_size_, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(y.to(device).data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
            total += load_batch_size_ 
                
        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
        print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
        corrects = {5:top5,1:top1}        
        return corrects
#### Additional dataset
import os
import random
import json
from scipy.io import loadmat
from PIL import Image
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import random_split, ConcatDataset, Subset

# from transforms import MultiView, RandomResizedCrop, ColorJitter, GaussianBlur, RandomRotation
from torchvision import transforms as T
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder, ImageNet, Caltech101, Caltech256

import kornia.augmentation as K
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets import VisionDataset
class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)
            
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageNet(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        def _convert2rgb(self,examples):
            examples["image_rgb"] = [image.convert("RGB") for image in examples["image"]]
            return examples
        print('Converting to RGB. This takes times...')
        self.data = self.data.map(_convert2rgb, remove_columns=["image"], batched=True)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image_rgb']
        label = sample['label']
        if self.transform:
            image = self.transform(image)
        return image,label
    
import pandas as pd    
class CUB():
    def __init__(self, root, dataset_type='train', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        print('Load CUB datasets...')
        trn_indices, val_indices = torch.load(os.path.join(root, 'cub200.pth'))
        df_img = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
        df_label = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
        df_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
        df = pd.concat([df_img, df_label, df_split], axis=1)
        # relabel
        df['Label'] = df['Label'] - 1

        print('Split CUB datasets...')
        # split data
        if dataset_type == 'test':
            df = df[df['Train'] == 0]
        elif dataset_type == 'train' or dataset_type == 'valid':
            df = df[df['Train'] == 1]
            # split train, valid
            if dataset_type == 'train':
                df = df.iloc[trn_indices]
            else:       # dataset_type == 'valid'
                df = df.iloc[val_indices]
        else:
            raise ValueError('Unsupported dataset_type!')
        self.img_name_list = df['Image'].tolist()
        self.label_list = df['Label'].tolist()
        print('Convert2RGB scale...')
        # Convert greyscale images to RGB mode
        self._convert2rgb()
        print('Load Done...')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.img_name_list[idx])
        image = Image.open(img_path)
        target = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def _convert2rgb(self):
        for i, img_name in enumerate(self.img_name_list):
            img_path = os.path.join(self.root, 'images', img_name)
            image = Image.open(img_path)
            color_mode = image.mode
            if color_mode != 'RGB':
                # image = image.convert('RGB')
                # image.save(img_path.replace('.jpg', '_rgb.jpg'))
                self.img_name_list[i] = img_name.replace('.jpg', '_rgb.jpg')    
                
import math

import numpy as np
import torch

def get_1d_sincos_pos_embed(x: torch.Tensor, dim: int):
    """From: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py"""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb