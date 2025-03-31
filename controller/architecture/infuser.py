import torch
import torch.nn as nn
import pdb
from typing import Optional, Dict, Any, Tuple, Union
from diffusers.models.attention import BasicTransformerBlock
from einops import rearrange, repeat


class MyAdaLayerNorm(nn.Module):
    r"""
    Copied from: https://github.com/huggingface/diffusers/blob/a9a5b14f3560596e34ce960e74ff29cc6b6a22e1/src/diffusers/models/normalization.py#L28
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, timestep_dim: Optional[int] = None, num_embeddings: Optional[int] = None):
        super().__init__()
        self.has_embedding_layer = num_embeddings is not None
        if self.has_embedding_layer:
            timestep_dim = embedding_dim
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
            self.silu = nn.SiLU()

        self.linear = nn.Linear(timestep_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep))) if self.has_embedding_layer else self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=-1) # only difference, add dim=-1
        x = self.norm(x) * (1 + scale) + shift
        return x

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, num_layers=2, norm_type='layer_norm', bias=True, timestep_dim: Optional[int] = None, num_embeddings: Optional[int] = None):
        super(MyLinear, self).__init__()
        self.in_layer = nn.Linear(in_features, out_features, bias)

        norm_layer = MyAdaLayerNorm(embedding_dim=out_features,
                                      timestep_dim=timestep_dim,
                                      num_embeddings=num_embeddings if norm_type == 'ada_norm' else nn.LayerNorm(out_features))


        self.norm_in = norm_layer
        self.body = nn.ModuleList([])
        for _ in range(num_layers):
            self.body.append(nn.Linear(out_features, out_features, bias))
            self.body.append(nn.GELU())
            self.body.append(norm_layer)


    def forward(self, x: torch.Tensor, timestep: torch.LongTensor, sig: torch.Tensor, **kwargs):
        '''
        x: input tensor, coming form the conditioner
        timestep: timestep tensor
        sig: signal tensor, coming from the main network
        '''
        b, f, c = x.shape
        bs, _, hs, ws = sig.shape
        reps = bs // b

        x = self.norm_in(self.in_layer(x), timestep[::reps])
        x = repeat(x, 'b f c -> (b t) f c', t=reps)
        sig = rearrange(sig, 'b c h w -> b (h w) c')
        sig = torch.mean(sig, dim=1, keepdim=True)
        x = x+sig

        for layer in self.body:
            if isinstance(layer, MyAdaLayerNorm):
                x = layer(x, timestep)
            else:
                x = layer(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=1, w=1)
        return x


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# Infusers

class Infuser(nn.Module):
    def     __init__(self, mode='', channels=[320, 640, 1280, 1280], num_layers=[4, 3, 3, 2], **kwargs):
        super(Infuser, self).__init__()

        mode = mode.split('-')
        self.mode = mode[0]
        self.num_layers = sum(num_layers)

        self.zero_convs = nn.ModuleList([])
        for ch, n_blocks in zip(channels, num_layers):
            for _ in range(n_blocks):
                self.zero_convs.append(zero_module(nn.Conv2d(ch, ch, 1)))

        self.rag_mode = 'mean'
        self.blocks = nn.ModuleList([])
        if self.mode == 'add':
            for ch, n_blocks in zip(channels, num_layers):
                for _ in range(n_blocks):
                    self.blocks.append(MyLinear(in_features=kwargs.get('cross_attention_dim', ch),
                                                out_features=ch,
                                                norm_type=kwargs.get('norm_type', 'layer_norm'),
                                                timestep_dim=kwargs.get('timestep_dim', None),
                                                num_embeddings=kwargs.get('num_embeds_ada_norm', None)))


        elif self.mode == 'crossattn':
            for ch, n_blocks in zip(channels, num_layers):
                for _ in range(n_blocks):
                    block = BasicTransformerBlock(
                            dim=ch,
                            num_attention_heads=8,
                            attention_head_dim=ch // 8,
                            cross_attention_dim=kwargs.get('cross_attention_dim', ch),
                            only_cross_attention=kwargs.get('only_cross_attention', True),
                            norm_type=kwargs.get('norm_type', 'layer_norm'),
                            num_embeds_ada_norm=1, # hack, set to one and reset later if needed
                        )
                    if kwargs.get('norm_type', 'layer_norm') != 'layer_norm':
                        # Hack to change the huggingface implementation
                        block.norm1 = MyAdaLayerNorm(embedding_dim=ch, num_embeddings=kwargs.get('num_embeds_ada_norm', None), timestep_dim=kwargs.get('timestep_dim', None))
                        block.norm2 = MyAdaLayerNorm(embedding_dim=ch, num_embeddings=kwargs.get('num_embeds_ada_norm', None), timestep_dim=kwargs.get('timestep_dim', None))
                    self.blocks.append(block)

            # Other hparams
            self.attn_mode = mode[1] if len(mode) > 1 and (mode[1] in ['spatial', 'temporal']) else 'spatial'
            self.rag_mode = mode[2] if len(mode) > 2 and (mode[2] in ['mean', 'loop']) else 'mean'

    def forward(self, x, cond, iterator, timestep, return_tuple=False, **kwargs):
        idx = next(iterator)

        if self.mode == '':
            return x
        # Reshape the conditioning tensor
        num_rag = kwargs.pop('num_rag', 1)
        num_frames = kwargs.pop('num_frames', 1)
        if self.rag_mode == 'mean':
            cond = rearrange(cond, '(b r) f c -> b r f c', r=num_rag)
            if kwargs.get('encoder_attention_mask', None) is not None:
                # average only over the valid conditioning frames
                enc_mask = kwargs.pop('encoder_attention_mask').to(cond.dtype)
                cond = torch.sum(cond * enc_mask[..., None, None], dim=1) / torch.sum(enc_mask, dim=1)
            else:
                # average over all
                cond = torch.mean(cond, dim=1)

        # Actual forward
        if self.mode == 'add':
            timestep = timestep.unsqueeze(1)
            if len(self.blocks):
                cond = self.blocks[idx](cond, timestep, x)
            res = self.zero_convs[idx](cond)
            if return_tuple:
                return (x, res)
            return x + res

        elif self.mode == 'concat':
            return torch.cat((x, self.zero_convs[idx](cond)), dim=1)

        elif self.mode == 'crossattn':
            average_time = kwargs.get('average_time', False)
            b, c, h, w = x.shape
            x = rearrange(x, '(b f) c h w -> (b h w) f c', f=num_frames)
            cond = repeat(cond, '(b r) f c -> (b h w) (r f) c', r=1 if self.rag_mode == 'mean' else num_rag, h=h, w=w)
            encoder_attention_mask = kwargs.get('encoder_attention_mask', None)
            encoder_attention_mask = repeat(encoder_attention_mask, 'b r -> (b h w) (r f)', h=h, w=w, f=1 if average_time else num_frames) if encoder_attention_mask is not None else None
            timestep = repeat(timestep, '(b t) ... -> (b h w) t ...', h=h, w=w, t=num_frames)

            if average_time:
                x = torch.mean(x, dim=1, keepdim=True)
                timestep  = timestep[:, [0]]

            res = self.blocks[idx](hidden_states=x, encoder_hidden_states=cond, timestep=timestep, encoder_attention_mask=encoder_attention_mask,)
            res = rearrange(res, '(b h w) f c -> (b f) c h w', h=h, w=w)
            res = self.zero_convs[idx](res)
            if average_time:   res = repeat(res, 'b c h w -> (b f) c h w', f=num_frames)
            if return_tuple:
                return (x, res)
            x = rearrange(x, '(b h w) f c -> (b f) c h w', f=num_frames, h=h, w=w)
            return x + res
        else:
            raise ValueError(f"Invalid mode: {self.mode}")





