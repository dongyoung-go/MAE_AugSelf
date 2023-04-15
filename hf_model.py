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
from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining

from AugSelf_model import load_mlp

class hf_MAE_ViT(torch.nn.Module):
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
        
        configuration = ViTMAEConfig()

        configuration.image_size = self.image_size
        configuration.patch_size = self.patch_size
        configuration.hidden_size = self.emb_dim ## 768 in default config
        configuration.mask_ratio = mask_ratio
        # encoder
        configuration.num_hidden_layers = encoder_layer ## 12
        configuration.num_attention_heads = encoder_head ## 12
        configuration.intermediate_size = 768
        # decoder
        configuration.decoder_hidden_size = self.emb_dim ## 512 in default config
        configuration.decoder_num_hidden_layers = decoder_layer ## 8
        configuration.decoder_num_attention_heads = decoder_head ## 16
        configuration.decoder_intermediate_size = 768 ## 2048 in default config

        # model = ViTMAEModel(configuration)
        self.model = ViTMAEForPreTraining(configuration)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f'# hf_param:{pytorch_total_params}')
        
        if self.add_params.get('aux_task',False)=='simclr':
            self.head = load_mlp(self.emb_dim, configuration.intermediate_size, self.emb_dim, num_layers=2, last_bn=False).to('cuda')
        elif self.add_params.get('aux_task',False)=='simsiam':
            self.projector = load_mlp(self.emb_dim, self.emb_dim, self.emb_dim, num_layers=2, last_bn=True).to('cuda')
            self.predictor = load_mlp(self.emb_dim, self.emb_dim//4, self.emb_dim, num_layers=2, last_bn=False).to('cuda')
        elif self.add_params.get('aux_task',False)=='barlowtwin':
            nout = 2048 ## barlow-twin has extremely large nout
            self.projector = load_mlp(self.emb_dim, configuration.intermediate_size, nout, num_layers=3, last_bn=False).to('cuda')
            self.bn = torch.nn.BatchNorm1d(nout, affine=False)

    def forward(self, img):
        output = self.model(pixel_values=img,output_hidden_states=True)
        hidden_states = output.hidden_states[-1] ## b t c
        recon_loss = output.loss
        features = hidden_states
        special_features = hidden_states[:,0,:]
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

        return recon_loss, proj_emb, special_features, features

if __name__ == '__main__':
    img = torch.rand(2, 3, 32, 32)
    loss,_,feature_s,feature_f = hf_MAE_ViT(img)
    print(feature_s.shape)
    print(feature_f.shape)
    print(loss)