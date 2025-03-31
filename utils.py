import importlib
from inspect import isfunction
from pathlib import Path

import imageio
from einops import rearrange
from diffusers import DDIMScheduler, TextToVideoSDPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from models.unet_3d_condition import UNet3DConditionModel

import numpy as np


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_primary_models(pretrained_model_path):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet


def export_to_video(video_frames, output_video_path, fps=None):
    # remove
    video_frames = video_frames[0].clamp(-1., 1.)
    video_frames = (video_frames + 1.) / 2.
    video_frames = (video_frames.permute(1, 2, 3, 0) * 255.).byte().cpu()

    video = [np.array(img) for img in video_frames]
    if output_video_path.suffix == '.gif':   imageio.mimsave(output_video_path, video, duration=125, loop=25)
    if output_video_path.suffix == '.mp4':   imageio.mimsave(output_video_path, video, fps=8)


def tensor_to_vae_latent(t, vae, need_vae_encoding=True):
    # TODO: Hack to support RAGDataset, fix this.
    if not need_vae_encoding:
        if t.ndim == 4:   return rearrange(t, 'b r f c -> b c r f')
        if t.ndim == 3:   return rearrange(t, 'b r c -> b c r 1')

    if t.ndim == 5:
        video_length = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    elif t.ndim == 6:
        rag_samples, video_length = t.shape[1], t.shape[2]
        t = rearrange(t, "b r f c h w -> (b r f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b r f) c h w -> b c r f h w", r=rag_samples, f=video_length)
    else:
        raise NotImplementedError("Only 5D and 6D tensors are supported.")

    latents = latents * 0.18215

    return latents