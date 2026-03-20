
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        # timm 0.9 expects global_pool as a string ('token', 'avg'), not a bool.
        # We pop it so we use the default timm behaviour ('token') and handle our own pooling.
        kwargs.pop('global_pool', None)
        super(VisionTransformer, self).__init__(**kwargs)

        self.custom_global_pool = global_pool
        if self.custom_global_pool:
            norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            if hasattr(self, 'norm'):
                del self.norm  # remove the original norm

    def forward_features(self, x, **kwargs):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.custom_global_pool:
            x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            if hasattr(self, 'norm'):
                x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        """Override forward completely to avoid timm 0.9.x forward_head calling timm's global_pool"""
        x = self.forward_features(x)
        if hasattr(self, 'head'):
            x = self.head(x)
        return x


def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def Dinov2(args, **kwargs):
    
    if args.model_arch == 'dinov2_vits14':
        arch = 'vit_small_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitb14':
        arch = 'vit_base_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitl14':
        arch = 'vit_large_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitg14':
        arch = 'vit_giant_patch14_dinov2.lvd142m'
    else:
        raise ValueError(f"Unknown model_arch '{args.model_arch}'. "
                         f"Expected one of: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14")
        
    model = timm.create_model(
        arch,
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model



def RETFound_dinov2(args, **kwargs):
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model


def Dinov3(args, **kwargs):
    # Load ViT-L/16 backbone (hub model has `head = Identity` by default)
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=args.model_arch,
        pretrained=False,   # main() will load your checkpoint
        trust_repo=True,
    )

    # Figure out feature dimension for the probe
    feat_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    model.head = nn.Linear(feat_dim, args.nb_classes)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    return model
