import torch
import torch.nn as nn
from diffusers.models.transformer_temporal import TransformerTemporalModel
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import pdb
from models.unet_3d_blocks import transformer_g_c
from typing import Optional, Dict, Any, Tuple, Union, List
from models.unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
    transformer_g_c
)

from utils import instantiate_from_config

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from einops import rearrange, repeat

class Controller(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 controller_network_config : Dict[str, Any] = None,
                 learnable_timestep: bool = True,
                 control_mode : str = 'residual',
                 **kwargs):
        super().__init__()

        self.learnable_timestep = learnable_timestep
        if self.learnable_timestep:
            controller_network_config['params']['infuser_config']['num_embeds_ada_norm'] = 1000
            controller_network_config['params']['infuser_config']['timestep_dim'] = None
        else:
            controller_network_config['params']['infuser_config']['num_embeds_ada_norm'] = None
            controller_network_config['params']['infuser_config']['timestep_dim'] = 1280

        self.controller = instantiate_from_config(controller_network_config)

        # 1. Control Mode: where to apply the controlling mechanism
        assert control_mode in ['residual', 'encoder', 'encoder-decoder']
        self.control_mode = control_mode

    def forward(self, base_model: torch.nn.Module,
                sample: torch.Tensor,
                timesteps: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                cond_latents: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                no_control: bool = False,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                cond_attn_mask: Optional[torch.Tensor] = None,
                cond_encoder_attn_mask: Optional[torch.Tensor] = None,
                **kwargs):

        if no_control:
            return base_model(sample,
                            timesteps,
                            encoder_hidden_states,
                            class_labels,
                            timestep_cond,
                            attention_mask,
                            cross_attention_kwargs,
                            down_block_additional_residuals,
                            mid_block_additional_residual,
                            return_dict=True).sample

        # 0. Go through the network
        default_overall_up_factor = 2**base_model.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        # 1. time
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        num_frames = sample.shape[2]
        num_rag = cond_latents.shape[2]
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = base_model.time_proj(timesteps)

        # Iterators for the infusers
        ctrl2sample_idx = iter(range(0, self.controller.infuser.num_layers)) # 0 to 11 iterator

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=base_model.dtype)

        t_emb = base_model.time_embedding(t_emb, timestep_cond)
        emb = t_emb.repeat_interleave(repeats=num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=num_frames, dim=0)

        timesteps_cond = timesteps.repeat_interleave(repeats=num_frames, dim=0) if self.learnable_timestep else emb

        # 2. pre-process
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])
        sample = base_model.conv_in(sample)


        ctrl = self.controller.process_input(cond_latents, cond_attn_mask)

        if num_frames > 1:
            if base_model.gradient_checkpointing:
                sample = transformer_g_c(base_model.transformer_in, sample, num_frames)
            else:
                sample = base_model.transformer_in(sample, num_frames=num_frames).sample

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in base_model.down_blocks:
            # -- Base model
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controller=self.controller,
                    controller_kwargs=dict(cond=ctrl, iterator=ctrl2sample_idx, num_frames=num_frames, num_rag=num_rag, timestep=timesteps_cond, encoder_attention_mask=cond_encoder_attn_mask)
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)


            down_block_res_samples += res_samples

        # 4. mid
        if base_model.mid_block is not None:
            sample = base_model.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                controller=self.controller,
                controller_kwargs=dict(cond=ctrl, iterator=ctrl2sample_idx, num_frames=num_frames, num_rag=num_rag, timestep=timesteps_cond, encoder_attention_mask=cond_encoder_attn_mask)
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(base_model.up_blocks):
            is_final_block = i == len(base_model.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                    controller=self.controller,
                    controller_kwargs=dict(cond=ctrl, iterator=ctrl2sample_idx, num_frames=num_frames, num_rag=num_rag, timestep=timesteps_cond, encoder_attention_mask=cond_encoder_attn_mask)
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )
        # 6. post-process
        if base_model.conv_norm_out:
            sample = base_model.conv_norm_out(sample)
            sample = base_model.conv_act(sample)

        sample = base_model.conv_out(sample)

        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)
        return  sample


