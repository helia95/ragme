from typing import Optional, Dict, Any, Tuple, Union
import torch.nn as nn
import torch
from diffusers.models.resnet import Downsample2D, ResnetBlock2D
from diffusers.models.attention import BasicTransformerBlock
from einops import rearrange, repeat
import pdb
from controller.architecture.infuser import Infuser


class TemporalEncoder(nn.Module):
    def __init__(self, cin=512, num_layers=8, infuser_config : Dict = None, **kwargs):
        super().__init__()

        self.body = nn.ModuleList([])
        for _ in range(num_layers):
            self.body.append(BasicTransformerBlock(
                                dim=cin,
                                num_attention_heads=8,
                                attention_head_dim=cin // 8,
                                cross_attention_dim=None,
                                only_cross_attention=False))

        self.infuser = Infuser(**infuser_config)

        self.use_cls_token = kwargs.get('use_cls_token', False)
        self.use_mean = kwargs.get('use_mean', False)
        # check cls and mean are not active at the same time
        assert not (self.use_cls_token and self.use_mean), "Both cls token and mean pooling cannot be active at the same time"
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, cin))
        self.spatial_average = kwargs.get('spatial_average', True)
        self.temporal_average = kwargs.get('temporal_average', True)


    def process_input(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = rearrange(x, 'b c r f -> (b r) f c')
        attn_mask = rearrange(attn_mask, 'b r f -> (b r) f') if attn_mask is not None else None
        b, num_frames, _ = x.shape
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '() f c -> b f c', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            attn_mask = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=attn_mask.device), attn_mask), dim=1) if attn_mask is not None else None
        for block in self.body:
            x = block(x, attention_mask = attn_mask)
        if self.use_cls_token:
            x = x[:, [0]]
        if self.use_mean:
            x = torch.mean(x, dim=1, keepdim=True)
        return x


    def infuse(self, x, cond, iterator, num_frames, num_rag, **kwargs):
        if self.infuser.mode == 'add':
            return self.infuser(x, cond, iterator, num_frames=num_frames, num_rag=num_rag, **kwargs)
        elif self.infuser.mode == 'crossattn':
            # Pool the latents temporally
            x_pooled = torch.mean(x, dim=[-1, -2], keepdim=True) if self.spatial_average else x
            kwargs.update({'average_time' : self.temporal_average})
            res = self.infuser(x_pooled, cond, iterator, num_frames=num_frames, num_rag=num_rag, return_tuple=True, **kwargs)[1]
            return x + res
        else:
            raise NotImplementedError(f"Infuser mode {self.infuser.mode} not implemented")




