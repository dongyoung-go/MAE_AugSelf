import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from transforms import extract_diff, extract_each

class SSObjective:
    def __init__(self, crop=-1, color=-1, flip=-1, gray=-1, blur=-1, rot=-1, sol=-1, only=False, each=False, **add_params):
        self.only = only
        self.params = [
            ('crop',  crop,  4, 'regression'),
            ('color', color, 4, 'regression'),
            ('flip',  flip,  1, 'binary_classification'),
            ('gray',  gray,  1, 'binary_classification'),
            ('blur',  blur,  1, 'regression'),
            ('rot',    rot,  4, 'classification'),
            ('sol',    sol,  1, 'regression'),
            ('sim_preserve', add_params.get('ss_sim_preserve',-1), 1, 'regression'),
            ('sim_preserve_MSE', add_params.get('ss_sim_preserve_MSE',-1), 1, 'regression'),
            ('shuffle_idx', add_params.get('ss_shuffle_idx',-1), 64, 'regression'),
        ]
        self.diff_num = sum([int(weight>0) for _, weight, _, _ in self.params])
        self.each = each
        if self.each:
            print('self.each is ON')
        print('<Params>\n',self.params)

        '''
        change_mode for designating version
        {
            1 : cos*d_aug in simsiam, CE(logit,d_aug*label) in moco.
                can result in just weighting (implemented in simsiam, moco)
            1.5 : focal loss in CE, just for check (implemented in moco)
            2 : focal loss with d_aug in CE, (implemented in moco)
        }
        '''
        print(f"<add_params>\n{add_params}")
        self.diff_aug = add_params.get('diff_aug',False)
        self.change_mode = add_params.get('change_mode',-1)
        self.focal_gamma = add_params.get('focal_gamma',0)
        self.dimension_wise = add_params.get('dimension_wise',False)
        if self.focal_gamma!=0:
            print(f'self.focal_gamma : {self.focal_gamma}')
        if self.diff_aug:
            self.diff_aug_weight = add_params.get('diff_aug_weight',1)
            print('self.diff_aug is ON')
            print(f'self.diff_aug_weight is {self.diff_aug_weight}')
            self.diff_aug_params = [
                ('crop',  add_params.get('diff_crop',-1),  4, 'regression'),
                ('color', add_params.get('diff_color',-1), 4, 'regression'),
                ('flip',  add_params.get('diff_flip',-1),  1, 'binary_classification'),
                ('blur',  add_params.get('diff_blur',-1),  1, 'regression'),
                ('rot',    add_params.get('diff_rot',-1),  4, 'classification'),
                ('sol',    add_params.get('diff_sol',-1),  1, 'regression'),
            ]
            
            ## Need to remove hard coded cuda to device
            self.W = torch.tensor(sum([[x[1]]*x[2] for x in self.diff_aug_params if x[1]>0],[])).cuda()
            self.params_range = [ ## will be used for ad hoc diff SSL
                ('crop',  torch.tensor([0.0]*4).cuda(),  torch.tensor([0.62]*4).cuda(), 'regression'),
                ('color', torch.tensor([0.0]*4).cuda(), torch.tensor([1.0]*4).cuda(), 'regression'),
                ('flip',  torch.tensor([0]).cuda(),  torch.tensor([1]).cuda(), 'binary_classification'),
                ('blur',  torch.tensor([-2.0]).cuda(),  torch.tensor([2.0]).cuda(), 'regression'),
                ('rot',    rot,  4, 'classification'),
                ('sol',    sol,  1, 'regression'),
            ]
        self.diff_aug_inv = add_params.get('diff_aug_inv',False)
        if self.diff_aug_inv:
            print('self.diff_aug_inv is ON')        
        
    def cal_diff(self, d):
        '''
        difference score of augmentation parameter. It is weighted mean with its weight on self.diff_aug_params. 
        Should return [batchsize x 1] score vector, with its range [0,1]
        Other possible options would be 
        1. Use predicted value
        2. Use predicted value of hidden&concated layer of Aug_pred layer
        3. Use harmonic mean
        4. Use different augmentation set with Aug_pred
        5. Just for experimental purpose, use sim_score, i.e. -1*(diff_score-1)
        '''
        assert(self.diff_aug)
        d_norm = dict()
        for idx, (name, weight, _, _) in enumerate(self.diff_aug_params):
            if weight <= 0:
                continue
            pmin, pmax = self.params_range[idx][1], self.params_range[idx][2]
            
            value = d[name].abs() # if name=='crop' else d[name]
            norm_value_ = (value-pmin)/(pmax-pmin)
            norm_value_[norm_value_<0.0] = 0.0
            norm_value_[norm_value_>1.0] = 1.0
            d_norm[name] = norm_value_.pow(self.diff_aug_weight)

        d_norm = torch.cat([x for x in d_norm.values()],1)
        diff_score = torch.mul(d_norm,self.W).sum(dim=1)/torch.sum(self.W)

        if self.diff_aug_inv:
            diff_score = -1*(diff_score-1)
        
        return diff_score

    def __call__(self, ss_predictor, z1, z2, d1, d2, symmetric=True):
        ## if self.aus_self: 
        ## else
        if (z1.dim()==3)&(z2.dim()==3):
            z1 = rearrange(z1, 't b c -> b (t c)')
            z2 = rearrange(z2, 't b c -> b (t c)') 

        if self.each:
            z = torch.cat([z1, z2], 0) ## 512,2048
            d = { k: torch.cat([d1[k], d2[k]], 0) for k in d1.keys() } ## 512,4
        elif symmetric:
            z = torch.cat([torch.cat([z1, z2], 1), ## 256,4096
                           torch.cat([z2, z1], 1)], 0) ## 256,4096
            ## => 512,4096
            d = { k: torch.cat([d1[k], d2[k]], 0) for k in d1.keys() } ## 512,4
        else:
            z = torch.cat([z1, z2], 1) ## 256,4096
            d = d1 ## 256,4

        losses = { 'total': 0 }
        for param_idx, (name, weight, n_out, loss_type) in enumerate(self.params):
            if weight <= 0:
                continue
            if self.dimension_wise:
                assert(symmetric)
                in_dim = z.size(1)//2
                z1_s,z1_e = (in_dim//2 + (in_dim//2//self.diff_num)*(param_idx),in_dim//2 + (in_dim//2//self.diff_num)*(param_idx+1))
                z2_s,z2_e = (in_dim + in_dim//2 +(in_dim//2//self.diff_num)*(param_idx),in_dim + in_dim//2 +(in_dim//2//self.diff_num)*(param_idx+1))
                z_ = torch.cat([z[:,z1_s:z1_e],z[:,z2_s:z2_e]],1)
            else:
                z_ = z
                
            p = ss_predictor[name](z_)
            if loss_type == 'regression':
                losses[name] = F.mse_loss(torch.tanh(p), d[name])
            elif loss_type == 'binary_classification':
                losses[name] = F.binary_cross_entropy_with_logits(p, d[name])
            elif loss_type == 'classification':
                losses[name] = F.cross_entropy(p, d[name])
            losses['total'] += losses[name] * weight

        return losses


def prepare_training_batch(batch, t1, t2, device):
    ((x1, w1), (x2, w2)), _ = batch
    with torch.no_grad():
        x1 = t1(x1.to(device)).detach()
        x2 = t2(x2.to(device)).detach()
        diff1 = { k: v.to(device) for k, v in extract_diff(t1, t2, w1, w2).items() }
        diff2 = { k: v.to(device) for k, v in extract_diff(t2, t1, w2, w1).items() }

    return x1, x2, diff1, diff2



def load_mlp(n_in, n_hidden, n_out, num_layers=3, last_bn=True):
    layers = []
    for i in range(num_layers-1):
        layers.append(nn.Linear(n_in, n_hidden, bias=False))
        layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_hidden, n_out, bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(n_out))
    mlp = nn.Sequential(*layers)
    reset_parameters(mlp)
    return mlp


def load_ss_predictor(n_in, ss_objective, n_hidden=512, **add_params):
    each = add_params.get('each',False)
    dimension_wise = add_params.get('dimension_wise',False)
    ss_predictor = {}
    diff_num = sum([int(weight>0) for _, weight, _, _ in ss_objective.params])
    for name, weight, n_out, _ in ss_objective.params:
        if weight > 0:
            if each:
                ss_predictor[name] = load_mlp(n_in, n_hidden, n_out, num_layers=3, last_bn=False).to('cuda')
            elif dimension_wise:
                '''
                Leave n_in/2 dimensions for global features
                Make the oeter (n_in/2)/diff_num dimensions for each DA features
                '''
                ss_predictor[name] = load_mlp((n_in*2//2)//(diff_num), n_hidden, n_out, num_layers=3, last_bn=False)
            else:
                ss_predictor[name] = load_mlp(n_in*2, n_hidden, n_out, num_layers=3, last_bn=False).to('cuda')
    print('<Structure in ss_predictor>\n',ss_predictor)
    return ss_predictor    

def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()

        if isinstance(m, nn.Linear):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -bound, bound)    