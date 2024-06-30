from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple, Union
from math import isqrt

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from transformers import ViTConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


layer_scale = False
init_value = 1e-6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.module_1(x)
        x = x + self.module_2(x)
        return x

    def module_1(self, x):
        x = self.norm1(x.to(dtype=self.norm1.weight.dtype))  # Won't autocast to the dtype of the parameters of nn.BatchNorm2d.
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        x = self.drop_path(x)
        return x
    
    def module_2(self, x):
        x = self.norm2(x.to(dtype=self.norm2.weight.dtype))  # Won't autocast to the dtype of the parameters of nn.BatchNorm2d.
        x = self.mlp(x)
        x = self.drop_path(x)
        return x

class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x        
   

class HeadEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadEmbedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class MiddleEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleEmbedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches_height = image_size[0] // patch_size[0]
        num_patches_width = image_size[1] // patch_size[1]
        num_patches = num_patches_height * num_patches_width
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.proj(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    
    
class UniFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], image_size=224, in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, patch_size=[4, 2, 2, 2],
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., conv_stem=False, layer_norm_eps=1e-6, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps) 
        if conv_stem:
            self.patch_embed1 = HeadEmbedding(in_channels=in_chans, out_channels=embed_dim[0])
            self.patch_embed2 = MiddleEmbedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = MiddleEmbedding(in_channels=embed_dim[1], out_channels=embed_dim[2])
            self.patch_embed4 = MiddleEmbedding(in_channels=embed_dim[2], out_channels=embed_dim[3])
        else:
            self.patch_embed1 = PatchEmbed(
                image_size=image_size, patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                image_size=image_size // patch_size[0], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                image_size=image_size // (patch_size[0]*patch_size[1]), patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
            self.patch_embed4 = PatchEmbed(
                image_size=image_size // (patch_size[0]*patch_size[1]*patch_size[2]), patch_size=patch_size[3], in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(dim=embed_dim[0], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i])
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(dim=embed_dim[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i+depth[0]])
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            SABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
        for i in range(depth[3])])
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x.to(dtype=self.norm.weight.dtype))  # Won't autocast to the dtype of the parameters of nn.BatchNorm2d.
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class UniFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class UniFormerProjectionHead(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        # Layer normalisation before projection:
        self.layer_norm = torch.nn.LayerNorm(config.embed_dim[-1], eps=config.layer_norm_eps)

        # No bias as following layer normalisation with bias:
        self.projection = torch.nn.Linear(config.embed_dim[-1], config.projection_size, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class UniFormerModel(UniFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.uniformer = UniFormer(**vars(config))

        # Initialize weights and apply final processing:
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        last_hidden_state = self.uniformer(pixel_values)

        # Flatten h x w:
        last_hidden_state = torch.flatten(last_hidden_state, 2)

        # Permute last hidden state:
        last_hidden_state = torch.permute(last_hidden_state, [0, 2, 1])

        # return last_hidden_state
        if not return_dict:
            return last_hidden_state

        return ModelOutput(last_hidden_state=last_hidden_state)


class MultiUniFormerWithProjectionHead(UniFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.uniformer = UniFormer(**vars(config))
        self.projection_head = UniFormerProjectionHead(config)

        # Initialize weights and apply final processing:
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Flatten the batch and study_id dimensions:
        assert len(pixel_values.shape) == 5, 'pixel_values must be B, S, C, H, W, where S is the max number of images for a study in the batch.'
        last_hidden_state = self.uniformer(pixel_values.view(-1, *pixel_values.shape[2:]))
        # last_hidden_state = self.uniformer(pixel_values.flatten(start_dim=0, end_dim=1))

        # Flatten h x w:
        last_hidden_state = torch.flatten(last_hidden_state, 2)

        # Project the features for each spatial position to the decoder's hidden size:
        projection = self.projection_head(torch.permute(last_hidden_state, [0, 2, 1]))

        # Concatenate the features for each chest X-ray:
        projection = projection.view(pixel_values.shape[0], -1, projection.shape[-1])

        # Derive the attention mask from the pixel values:
        mask = (pixel_values[:, :, 0, 0, 0] != 0.0)[:, :, None]
        attention_mask = torch.ones(
            [projection.shape[0], pixel_values.shape[1], projection.shape[1] // pixel_values.shape[1]], 
            dtype=torch.long,
            device=mask.device,
        )
        attention_mask = attention_mask * mask
        attention_mask = attention_mask.view(attention_mask.shape[0], -1)

        if not return_dict:
            return projection

        return ModelOutput(last_hidden_state=projection, attention_mask=attention_mask)
    

if __name__ == '__main__':
    y = PatchEmbed()
    y(torch.randn(2, 3, 224, 224))
