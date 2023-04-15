import torch
import torch.nn.functional as F
import timm
import time
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from scipy.stats import multivariate_normal

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from AugSelf_model import load_mlp

from hf_model import *
from utils import get_1d_sincos_pos_embed

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle_from_normal(torch.nn.Module):
    def __init__(self, ratio,grid_n=16) -> None:
        super().__init__()
        self.ratio = ratio
        self.grid_n = grid_n # 16
        x, y = np.mgrid[0:1:1/grid_n, 0:1:1/grid_n]
        self.pos_grid = np.dstack((x, y))        

    def random_indexes_from_normal(self,size : int):
        ## size arg is just used for compatability
        mu = np.random.uniform(0,1,2)
        cov = np.random.rand(2, 2)*0.5
        cov = np.dot(cov, cov.T)
        cov[0, 1] = cov[1, 0] = cov[0, 0] * cov[1, 1]

        ## make grid pdfs
        rv = multivariate_normal(mu, cov,allow_singular=False)
        N_pdf = rv.pdf(self.pos_grid)    

        ## make masking_mat
        p = np.power(N_pdf.flatten(),1/4)+1e-6
        mask_mat = np.zeros((self.grid_n,self.grid_n)).flatten()
        rand_idx = np.random.choice(np.arange(self.grid_n**2),size=int(self.ratio*self.grid_n**2),replace=False,p = p/sum(p))
        mask_mat[rand_idx] = 1    

        ## make indexes
        forward_indexes = np.argsort(mask_mat)
        backward_indexes = np.argsort(forward_indexes)    

        dist_params = np.hstack([mu,np.delete(cov.flatten(),2)])
        return forward_indexes, backward_indexes, dist_params        
        
    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [self.random_indexes_from_normal(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        dist_params = torch.as_tensor(np.stack([i[2] for i in indexes], axis=-1), dtype=torch.float).to(patches.device).T

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes, dist_params

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio, patch_preserve=False) -> None:
        super().__init__()
        self.ratio = ratio
        self.patch_preserve = patch_preserve

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        if hasattr(self,'patch_preserve'):
            if self.patch_preserve:
                return patches[:remain_T], forward_indexes, backward_indexes, patches
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 **add_params
                 ) -> None:
        super().__init__()

        # if self.aus_self: ## temporal condition line
        # emb_dim += 1

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        if add_params.get('second_cls_token',False):
            self.second_cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        elif add_params.get('third_cls_token',False):
            self.second_cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
            self.third_cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.position_mask_ratio = add_params.get('position_mask_ratio',0)
        self.fixed_position_mask = add_params.get('fixed_position_mask',False)        
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        if self.fixed_position_mask:
            position_mask = torch.ones(self.pos_embedding.size())*(1-self.position_mask_ratio)
            self.position_mask = torch.bernoulli(position_mask).cuda()           
        self.add_params = add_params
        self.ss_sim_preserve = add_params.get('ss_sim_preserve',-1)
        self.direct_sim_preserve = add_params.get('direct_sim_preserve',False)
        self.mse_direct_sim_preserve = add_params.get('mse_direct_sim_preserve',False)
        self.CL_preserve = add_params.get('CL_preserve',False)
        self.add_params = add_params
        self.patch_preserve = True if ((self.ss_sim_preserve!=-1)|(self.direct_sim_preserve)|
                                       (self.mse_direct_sim_preserve)|(self.CL_preserve)|self.add_params.get('embed_mask',False)|
                                       (self.add_params.get('ss_sim_preserve_MSE',-1)!=-1)) else False
        self.shuffle = PatchShuffle(mask_ratio,patch_preserve=self.patch_preserve)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        if self.add_params.get('second_cls_token',False):
            trunc_normal_(self.second_cls_token, std=.02)
        elif self.add_params.get('third_cls_token',False):
            trunc_normal_(self.second_cls_token, std=.02)
            trunc_normal_(self.third_cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        if hasattr(self,'add_params') is False:
            self.add_params = dict()
        patches = self.patchify(img) ## why we firstly apply conv2d? just to compress img to 192x16x16?
        '''
        b : batchsize
        c : channel num, i.e. emb_dim, i.e. 192
        t : token
        h,w : height, width, i.e. 16 for image_size 32
        '''
        patches = rearrange(patches, 'b c h w -> (h w) b c') ## bx192x16x16->256xbx192
        if hasattr(self,'fixed_position_mask'):
            if self.fixed_position_mask is False: ## Other option might be just multiplying at the start
                position_mask = torch.ones(self.pos_embedding.size())*(1-self.position_mask_ratio)
                self.position_mask = torch.bernoulli(position_mask).cuda()        
        patches = patches + self.pos_embedding ## pos_embedding : (h w) 1 c ## 256x1x192
        patches, forward_indexes, backward_indexes = self.shuffle(patches) ## (h w) b c -> (h w 0.75) b c

        self.idx_target = rearrange(forward_indexes[:patches.size(0)],'c b -> (b c)') ## 64,512 -> 512*64
        cls_token_pathces = self.cls_token.expand(-1, patches.shape[1], -1)
        if self.add_params.get('second_cls_token',False):
            second_cls_token_pathces = self.second_cls_token.expand(-1, patches.shape[1], -1)
            patches = torch.cat([cls_token_pathces,second_cls_token_pathces, patches], dim=0)
        elif self.add_params.get('third_cls_token',False):
            second_cls_token_pathces = self.second_cls_token.expand(-1, patches.shape[1], -1)
            third_cls_token_pathces = self.third_cls_token.expand(-1, patches.shape[1], -1)
            patches = torch.cat([cls_token_pathces,second_cls_token_pathces,third_cls_token_pathces, patches], dim=0)
        else:
            patches = torch.cat([cls_token_pathces, patches], dim=0) ## (h w 0.75)+1 b c
        patches = rearrange(patches, 't b c -> b t c') ## b (h w 0.75)+1 c
        features = self.layer_norm(self.transformer(patches)) ## b (h w 0.75)+1 c
        features = rearrange(features, 'b t c -> t b c') ## (h w 0.75)+1 b c
        if self.add_params.get('second_cls_token',False):
            if self.add_params.get('concat_second_cls_token',False):
                special_features = torch.cat([features[0],features[1]],1) ## 2*b c, with 2nd_cls_token 
            else:
                special_features = features[1,:,:] ## b c, with 2nd_cls_token 
        elif self.add_params.get('third_cls_token',False):
            if self.add_params.get('concat_second_cls_token',False): ## assert concat_second_cls_token
                special_features = torch.cat([features[0],features[1],features[2]],1) ## 3*b c, with 2nd,3rd_cls_token 
            else:
                special_features = None
        elif self.add_params.get('mean_pool_wo_cls',False):
            special_features = features[1:,:,:].mean(axis=0) ## b c 
        else:
            special_features = features[0,:,:] ## b c 

        return features, backward_indexes, special_features


class MAE_position_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(int((image_size // patch_size) ** 2 * (1-mask_ratio) + 1), 1, emb_dim))
        ## Should I give pos_embedding here???

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, (image_size // patch_size) ** 2)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features):
        T = features.shape[0]
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c') # 512,65,192
        features = self.transformer(features)
        features = features[:,1:,:] # remove global feature # 512,64,192 

        patches = self.head(features) # 512,64,256 
        patches = rearrange(patches, 'b t c -> (b t) c') # 512*64,256

        return patches

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 **add_params
                 ) -> None:
        super().__init__()

        self.add_params = add_params
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.position_mask_ratio = add_params.get('position_mask_ratio',0)
        self.fixed_position_mask = add_params.get('fixed_position_mask',False)        
        if self.add_params.get('second_cls_token',False): 
            self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 2, 1, emb_dim))
        elif self.add_params.get('third_cls_token',False): 
            self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 3, 1, emb_dim))
        else:
            self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes, p_i=None):
        T = features.shape[0]
        ## because of the cls_token
        if self.add_params.get('second_cls_token',False): 
            backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                                          torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                                          backward_indexes + 2], dim=0)
        elif self.add_params.get('third_cls_token',False): 
            backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                                          torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                                          torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                                          backward_indexes + 3], dim=0)
        else:
            backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), 
                                          backward_indexes + 1], dim=0)
        ## add masked token behind features (since features are defined with patches[:remain_T] in PatchShuffle)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        if self.fixed_position_mask is False: ## Other option might be just multiplying at the start
            position_mask = torch.ones(self.pos_embedding.size())*(1-self.position_mask_ratio)
            self.position_mask = torch.bernoulli(position_mask).cuda()          
        if p_i is not None:
            assert self.add_params.get('CAN',False)
            features = features + p_i[None,:,:]
        else:
            features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        if self.add_params.get('second_cls_token',False):
            features = features[2:] # remove global feature
        elif self.add_params.get('third_cls_token',False):
            features = features[3:] # remove global feature
        else:
            features = features[1:] # remove global feature

        patches = self.head(features) ## t b 3*patch_size**2
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        if self.add_params.get('second_cls_token',False):
            mask = take_indexes(mask, backward_indexes[2:] - 2)
        elif self.add_params.get('third_cls_token',False):
            mask = take_indexes(mask, backward_indexes[3:] - 3)
        else:
            mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 **add_params
                 ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.add_params = add_params
        intermediate_size = 768

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, **add_params)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, **add_params)
        if self.add_params.get('aux_task',False)=='simclr':
            self.head = load_mlp(self.emb_dim, intermediate_size, self.emb_dim, num_layers=2, last_bn=False).to('cuda')
        elif self.add_params.get('aux_task',False)=='simsiam':
            self.projector = load_mlp(self.emb_dim, self.emb_dim, self.emb_dim, num_layers=2, last_bn=True).to('cuda')
            self.predictor = load_mlp(self.emb_dim, self.emb_dim//4, self.emb_dim, num_layers=2, last_bn=False).to('cuda')
        elif self.add_params.get('aux_task',False)=='barlowtwin':
            nout = 2048 ## barlow-twin has extremely large nout
            self.projector = load_mlp(self.emb_dim, intermediate_size, nout, num_layers=3, last_bn=False).to('cuda')
            self.bn = torch.nn.BatchNorm1d(nout, affine=False)       
            
        if self.add_params.get('CAN',False):
            self.noise_embed = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_dim, self.emb_dim),
            )
            
    @torch.no_grad()
    def add_noise(self, x: torch.Tensor):
        # Sample std uniformly from [0, self.noise_std_max]
        std = torch.rand(x.size(0), device=x.device) * self.add_params.get('noise_std_max',0.05)

        # Sample noise
        noise = torch.randn_like(x) * std[:, None, None, None]

        # Add noise to x
        x_noise = x + noise

        return x_noise, x, noise, std  

    def forward(self, img):
        if self.add_params.get('CAN',False):
            img, img_org, noise, std = self.add_noise(img)
        features, backward_indexes, special_features = self.encoder(img)
        p_i = None
        if self.add_params.get('CAN',False):
            img = img_org # target is original image
            p_i = self.noise_embed(get_1d_sincos_pos_embed(std, dim=self.emb_dim))
        predicted_img, mask = self.decoder(features,  backward_indexes,p_i=p_i)
        recon_loss = torch.mean((predicted_img - img) ** 2 * mask) / self.add_params.get('mask_ratio',0.75)
        
        proj_emb = None
        if self.add_params.get('aux_task',False)=='simclr':
            proj_emb = F.normalize(self.head(special_features), dim=-1)
        elif self.add_params.get('aux_task',False)=='simsiam':
            z1 = self.projector(special_features)
            p1 = self.predictor(z1)
            proj_emb = (z1, p1)
        elif self.add_params.get('aux_task',False)=='barlowtwin':
            z1 = self.projector(special_features)
            proj_emb = self.bn(z1)
            
        if self.add_params.get('CAN',False):
            denoise_mask = 1-mask
            denoise_loss = torch.mean((predicted_img - noise) ** 2 * denoise_mask) / (1-self.add_params.get('mask_ratio',0.75))
            return recon_loss, proj_emb, special_features, denoise_loss
            
        return recon_loss, proj_emb, special_features, features

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10, **add_params) -> None:
        super().__init__()
        print(f"<add_params>\n{add_params}")
        self.add_params = add_params
        if (encoder.add_params.get('mean_pool_wo_cls',False))&(self.add_params.get('direct_from_encoder',False)):
            ## This is what is done in CAN
            ## When direct_from_encoder is not given, by default ViT_Classifier build its own architecture with encoder's parts and use cls token, rather than using encoder's forward fn
            print('Rollback mean_pool_wo_cls to use cls token for classifier')
            encoder.add_params['mean_pool_wo_cls'] = False        
        if add_params.get('use_other_cls',False)=='second_cls_token':
            self.cls_token = encoder.second_cls_token
        elif add_params.get('use_other_cls',False)=='second_cls_token_merge':
            self.cls_token = torch.nn.Parameter(torch.stack([encoder.cls_token,encoder.second_cls_token]).mean(axis=0))
        elif add_params.get('use_other_cls',False)=='both_tokens':
            self.cls_token = encoder.cls_token
            self.second_cls_token = encoder.second_cls_token
        elif add_params.get('use_other_cls',False)=='both_third_tokens': ## for 3rd tokens, we will only test concat version for classifier
            self.cls_token = encoder.cls_token
            self.second_cls_token = encoder.second_cls_token
            self.third_cls_token = encoder.third_cls_token
        else:
            self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        # if self.add_params.get('direct_from_encoder',False):
        self.encoder = encoder

        n_hidden = 512
        if self.add_params.get('use_other_cls',False)=='both_tokens':
            input_dim = self.pos_embedding.shape[-1]*2
        elif self.add_params.get('use_other_cls',False)=='both_third_tokens':
            input_dim = self.pos_embedding.shape[-1]*3        
        else:
            input_dim = self.pos_embedding.shape[-1]

        if self.add_params.get('two_layer',False):
            self.classifier = TwoLayerNet(input_dim, n_hidden, num_classes)
        elif self.add_params.get('two_layer_from_AugSelf',False):
            self.classifier = load_mlp(input_dim, n_hidden, num_classes, num_layers=3, last_bn=False).to('cuda')                
        else:
            self.classifier = torch.nn.Linear(input_dim, num_classes)
        print('self.classifier\n',self.classifier) ## unified classification layer from self.head to self.classifier

    def forward(self, img):
        if self.add_params.get('direct_from_encoder',False):
            self.encoder.shuffle.ratio = 0.0
            head_input = self.encoder(img)[2] ## this should get [cls,emb] features, e.g. [512,192]
        else:
            patches = self.patchify(img)
            patches = rearrange(patches, 'b c h w -> (h w) b c')
            patches = patches + self.pos_embedding
            cls_token_pathces = self.cls_token.expand(-1, patches.shape[1], -1)
            if self.add_params.get('use_other_cls',False)=='both_tokens':
                second_cls_token_pathces = self.second_cls_token.expand(-1, patches.shape[1], -1)
                patches = torch.cat([cls_token_pathces,second_cls_token_pathces, patches], dim=0)            
            elif self.add_params.get('use_other_cls',False)=='both_third_tokens':
                second_cls_token_pathces = self.second_cls_token.expand(-1, patches.shape[1], -1)
                third_cls_token_pathces = self.third_cls_token.expand(-1, patches.shape[1], -1)
                patches = torch.cat([cls_token_pathces,second_cls_token_pathces,third_cls_token_pathces, patches], dim=0)
            else:
                patches = torch.cat([cls_token_pathces, patches], dim=0)
            patches = rearrange(patches, 't b c -> b t c')
            features = self.layer_norm(self.transformer(patches))
            features = rearrange(features, 'b t c -> t b c')
            if self.add_params.get('use_mean_features',False):
                head_input = features.mean(axis=0)
            elif self.add_params.get('use_mean_features_official',False):
                ## official repo did global pooling without cls
                head_input = features[1:].mean(axis=0)
            elif self.add_params.get('use_other_cls',False)=='both_tokens':            
                # head_input = features[:2].mean(axis=0)
                head_input = torch.cat([features[0],features[1]],1)
            elif self.add_params.get('use_other_cls',False)=='both_third_tokens':            
                # head_input = features[:2].mean(axis=0)
                head_input = torch.cat([features[0],features[1],features[2]],1)
            else:
                head_input = features[0]
                
        logits = self.classifier(head_input)
        return logits
    
class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)