from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
from omegaconf import OmegaConf

import argparse
import random
import json
from tqdm.auto import tqdm
from pathlib import Path
from natsort import natsorted


import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import TextToVideoSDPipeline, DDPMScheduler
from faiss.utils import l2_normalise

from utils import load_primary_models, tensor_to_vae_latent, export_to_video
from utils import instantiate_from_config
from tqdm import tqdm
from safetensors.torch import load_file

from faiss.dataset import process_video
import open_clip
from torchvision.transforms.functional import to_pil_image

class LoadEmbedding(torch.utils.data.Dataset):
    def __init__(self, txt_path: str,
                 data_path: str,
                 k : int = 5,
                 rag_n_sample_frames: int = 16,
                 rag_skip_frames: int = 1,
                 N: int = None,
                 shuffle: bool = False,
                 height: int = 256,
                 width: int = 448,
                 n_sample_frames: int = 12,
                 **kwargs):
        """
        Load the embeddings from the txt_path and data_path
        :param txt_path: Path to the precomputed text embeddings and nearest neighbors
        :param data_path: Path to the precomputed video embeddings
        :param k: Number of nearest neighbors to retrieve
        :param rag_n_sample_frames: Number of frames to sample from the retrieved videos
        :param rag_skip_frames: Number of frames to skip when sampling
        """

        super().__init__()

        self.txt_path = Path(txt_path)
        self.data_path = Path(data_path)
        # Read all prompts
        self.prompts = {}
        for file in tqdm(list(sorted((self.txt_path / "embeddings").glob("*.json"))), desc="reading embeddings..."):
            with open(file, "r") as f:
                self.prompts[file.stem] = json.load(f)["txt"].strip()
        self.file_ids = list(self.prompts.keys())
        self.file_ids = self.file_ids[:N] if N is not None else self.file_ids
        if shuffle:
            random.shuffle(self.file_ids)

        #
        self._k = k
        self.rag_skip_frames = rag_skip_frames
        self.rag_n_sample_frames = rag_n_sample_frames
        self.embedding_dim = 512

        self.N = N
        self.n_sample_frames = n_sample_frames
        self.height = height
        self.width = width
        self.fps = 8
        self.use_bucketing = False

    def __len__(self):
        return len(self.file_ids)

    def get_frame_buckets(self, vr):
        h, w, c = vr[0].shape
        width, height = T.transforms.functional.sensible_crop_size(self.width, self.height, w, h)
        resize = T.transforms.Resize((height, width), antialias=True)
        return resize

    def get_frame_batch(self, vr, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()
        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)
        effective_length = len(vr) // every_nth_frame
        n_sample_frames = min(n_sample_frames, effective_length)
        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)
        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")
        if resize is not None:
            video = resize(video)
        return video, vr

    def process_video_wrapper(self, vid_path):
        video, vr = process_video(vid_path, self.use_bucketing, self.width, self.height, self.get_frame_buckets,
                                  self.get_frame_batch)
        return video, vr

    def _load_video(self, path: str, normalize: bool = True):
        """
        Load a video from a given path and return it as a tensor. Optionally, normalize between -1, +1.
        Args
        path: str: Path to the video file.
        normalize: bool: Whether to normalize the video between -1, +1.
        """
        video, _ = self.process_video_wrapper(path)
        pixel_values = video[0] / 127.5 - 1.0
        pixel_values = F.pad(pixel_values, (0, 0, 0, 0, 0, 0, 0, self.n_sample_frames - pixel_values.shape[0]),
                             value=-1)
        return pixel_values


    def __getitem__(self, idx):

        txt = self.prompts[self.file_ids[idx]]
        nn_info = self.txt_path / "nearest_neighbors" / f"nn_{self._k}-rad_1.01-dedup_0.98"

        with open((nn_info / self.file_ids[idx]).with_suffix(".json"), "r") as f:
            nearest_neighbors_info = json.load(f)

        nearest_neighbors_ids = nearest_neighbors_info["videoID"][:self._k]

        # Load the embeddings
        nn_txt = nearest_neighbors_info["txt"][:self._k]
        nn_embeddings = torch.zeros((self._k, self.rag_n_sample_frames, self.embedding_dim))
        attn_mask = torch.ones((self._k, self.rag_n_sample_frames), dtype=torch.bool)
        encoder_attn_mask = torch.zeros((self._k,), dtype=torch.bool)
        for i, nn in enumerate(nearest_neighbors_ids):
            emb = np.load((self.data_path / str(nn)).with_suffix(".npy"))
            emb = l2_normalise(emb)
            emb = torch.from_numpy(emb)
            f = emb.shape[0]
            start = random.randint(0, max(f - self.rag_n_sample_frames * self.rag_skip_frames, 0))
            emb = emb[start:start + self.rag_n_sample_frames * self.rag_skip_frames:self.rag_skip_frames]
            # update the attention mask, then pad the embeddings
            attn_mask[i, emb.shape[0]:] = False
            encoder_attn_mask[i] = True
            nn_embeddings[i] = F.pad(emb, (0, 0, 0, self.rag_n_sample_frames - emb.shape[0]))

        nn_videos = torch.zeros((self._k, self.n_sample_frames, 3, self.height, self.width))
        for i, videoLoc in enumerate(nearest_neighbors_info["videoLoc"][:self._k]):
            # Load the caption
            nn_videos[i] = self._load_video(videoLoc)

        return txt, nn_txt, nn_videos, nn_embeddings, attn_mask, encoder_attn_mask

@torch.no_grad()
def inference_loop(
        output_dirs: List[str],
        pipeline,
        unet,
        text_encoder,
        vae,
        controller,
        cond_video,
        cond_attn_mask,
        cond_encoder_attn_mask,
        prompt_list: list,
        height_in: int,
        width_in: int,
        num_inference_steps: int,
        batch_size: int,
        num_frames: int,
        guidance_scale: float,
        num_videos_per_prompt: int,
        device:torch.device,
        init_latents: Optional[torch.Tensor] = None,
        no_control: bool = False,
        seed: int = 42,
        need_vae_encoding: bool = True,
        nn_text: Optional[List[str]] = None,):

    set_seed(seed)
    if not no_control:
        assert cond_video is not None, "Conditional video is required for controlled generation"

    controller.eval()

    scheduler = pipeline.scheduler
    ddpm_scheduler = DDPMScheduler.from_config(scheduler.config)
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator=None,
                                                           eta=0.0)

    do_classifier_free_guidance = guidance_scale > 1.0

    # Prepare cond latents
    if cond_video is not None:
        cond_latents = tensor_to_vae_latent(cond_video, vae, need_vae_encoding=need_vae_encoding)
        cond_latents = torch.cat([cond_latents] * 2, dim=0) if do_classifier_free_guidance else cond_latents
        # print(cond_latents.shape)
        if cond_attn_mask is not None:
            cond_attn_mask = torch.cat([cond_attn_mask] * 2, dim=0) if do_classifier_free_guidance else cond_attn_mask
        if cond_encoder_attn_mask is not None:
            cond_encoder_attn_mask = torch.cat([cond_encoder_attn_mask] * 2, dim=0) if do_classifier_free_guidance else cond_encoder_attn_mask
    else:
        cond_latents = None

    # 3. Encode input prompt
    prompt_embeds = pipeline._encode_prompt(
        list(prompt_list),
        device,
        1, # num_images_per_prompt hardcoded to 1
        do_classifier_free_guidance)

    # 4. Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Encode the cond latents if are provided
    shape = (
        1, # set the batch size always to 1, and repeat if needed
        unet.config.in_channels,
        num_frames,
        height_in // pipeline.vae_scale_factor,
        width_in // pipeline.vae_scale_factor,
    )

    init_latents_values = tensor_to_vae_latent(init_latents, vae) if init_latents is not None else None


    for video_idx in range(num_videos_per_prompt):
        noise = torch.randn(shape, device=device).repeat(batch_size, 1, 1, 1, 1)
        if init_latents is not None:
            latents = torch.mean(init_latents_values, dim=2)
            # latents = scheduler.add_noise(latents, noise, timesteps=timesteps[0])
            latents = ddpm_scheduler.add_noise(latents, noise, timesteps=timesteps[0])
        else:
            latents = noise

        # 5. Prepare latent variables
        num_channels_latents = unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            num_frames,
            height_in,
            width_in,
            prompt_embeds.dtype,
            device,
            None,
            latents
        )

        # 7. Denoising loop
        for i, t in tqdm(enumerate(timesteps), desc="Denoising loop", leave=False):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = controller(base_model=unet,
                                sample=latent_model_input,
                                timesteps=t,
                                cond_latents=cond_latents,
                                no_control=no_control,
                                encoder_hidden_states=prompt_embeds,
                                cond_attn_mask=cond_attn_mask,
                                cond_encoder_attn_mask=cond_encoder_attn_mask,
                                )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

            # reshape latents
            bsz, channel, frames, width, height = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width,
                                                             height)
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width,
                                                                   height)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents,
                                                **extra_step_kwargs).prev_sample

            # reshape latents back
            latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0,
                                                                                            2,
                                                                                            1,
                                                                                            3,
                                                                                            4)

        # 8. Post processing
        video_tensor = pipeline.decode_latents(latents)
        for i, output_dir in enumerate(output_dirs):
            if num_videos_per_prompt > 1:
                output_dir = output_dir.parent / f"{output_dir.stem}-{video_idx}.mp4"
            export_to_video(video_tensor[[i]], output_dir)
        torch.cuda.empty_cache()

def main(checkpoint_path : Path,
         opt: argparse.Namespace,
         accelerator : Accelerator,
         dataloader,
         tokenizer,
         text_encoder,
         vae,
         unet,
         use_ema: bool = False
         ):

    checkpoint_path = Path(checkpoint_path)
    config = OmegaConf.load(checkpoint_path / "config.yaml")
    checkpoint_path = natsorted(list((checkpoint_path / "checkpoints").glob("*")))[-1]

    # Load the controller
    controller = instantiate_from_config(config.controller_config)
    
    print(f"-> Loading Controller Weights...")
    weights = load_file(checkpoint_path / "controller.safetensors")
    controller.load_state_dict(weights, strict=True)
    del weights
    print("Controller weights loaded!")

    # Create the output path
    output_dir = Path(opt.output_dir) / f"{checkpoint_path.parent.parent.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "generated").mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieved").mkdir(parents=True, exist_ok=True)

    # Load the pipeline
    controller = accelerator.prepare(controller)

    # Load the pipeline
    pipeline = TextToVideoSDPipeline.from_pretrained(
        opt.zeroscope_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet
    )

    # Load CLIP 
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=accelerator.device)

    for idx, batch in tqdm(enumerate(dataloader), desc="Generating videos ...", total=len(dataloader)):
        prompt_list, nn_captions, nn_videos, nn_embeddings, cond_attn_mask, cond_encoder_attn_mask = batch
        curr_bs = len(prompt_list)
        if opt.save_with_prompt:
            output_fnames = prompt_list
        else:
            output_fnames = [str(idx*opt.batch_size+i).zfill(4) for i in range(curr_bs)]


        # Save the prompt in a txt file
        for i, fname in enumerate(output_fnames):
            with open(output_dir / "prompts" / f"{fname}.txt", "w") as f:
                f.write(prompt_list[i])

        # Save the retrieved videos
        for i, fname in enumerate(output_fnames):
            for j, nn_video in enumerate(nn_videos[i]):
                _tmp_path = output_dir / "retrieved" / fname
                _tmp_path.mkdir(parents=True, exist_ok=True)
                export_to_video(nn_video[None].permute(0, 2, 1, 3, 4), _tmp_path / f"{j:03d}.mp4")

        # Inference loop
        videos_for_clip = nn_videos[:, :, ::3]
        videos_for_clip_shape = videos_for_clip.shape
        videos_for_clip = rearrange(videos_for_clip, "b k f c h w -> (b k f) c h w")
        videos_for_clip = torch.stack([clip_preprocess(to_pil_image(frame)) for frame in videos_for_clip])
        with torch.no_grad():
            nn_embeddings = clip_model.encode_image(videos_for_clip.to(accelerator.device))
        nn_embeddings = nn_embeddings.reshape(videos_for_clip_shape[:3] + (-1,))
        nn_embeddings = nn_embeddings.to(torch.float32)
        cond_attn_mask = torch.ones(videos_for_clip_shape[:3]).bool().to(device=nn_embeddings.device)

        inference_loop(
            output_dirs=[output_dir / "generated" / f"{fname}.mp4" for fname in output_fnames],
            pipeline=pipeline,
            unet=unet,
            text_encoder=text_encoder,
            vae=vae,
            controller=controller,
            cond_video=nn_embeddings,
            cond_attn_mask=cond_attn_mask,
            cond_encoder_attn_mask=None,
            prompt_list=prompt_list,
            height_in=opt.height,
            width_in=opt.width,
            num_inference_steps=opt.num_inference_steps,
            batch_size=opt.batch_size,
            num_frames=opt.num_frames,
            guidance_scale=opt.guidance_scale,
            num_videos_per_prompt=opt.num_videos_per_prompt,
            device=accelerator.device,
            no_control=False,
            seed=opt.seed,
            init_latents = nn_videos,
            need_vae_encoding=False,
            nn_text=nn_captions)


if __name__ == "__main__":
    import argparse
    argparse = argparse.ArgumentParser()
    # Model to evaluate
    argparse.add_argument("--checkpoint", type=str, required=True)
    argparse.add_argument("--output_dir", type=str, required=True)
    argparse.add_argument("--zeroscope_path", type=str, required=True)
    argparse.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to retrieve.")
    argparse.add_argument("--seed", type=int, default=42)

    # Data loading of the prompts / embeddings
    argparse.add_argument("--shuffle", action="store_true", default=False, help="Shuffle the order of the prompts.")
    argparse.add_argument("-n", "--n_samples", type=int, default=None, help="Set if want to evaluate a subset of prompts.")
    argparse.add_argument("--data_path", type=str, required=True, help="Path of the retrieved videos.")
    argparse.add_argument("--txt_path", type=str, required=True, help="Path to the prompt dataset.")

    # DDIM Options
    argparse.add_argument("--num_inference_steps", type=int, default=50)
    argparse.add_argument("--batch_size", type=int, default=1)
    argparse.add_argument("--num_frames", type=int, default=12)
    argparse.add_argument("--guidance_scale", type=float, default=10.0)
    argparse.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    argparse.add_argument("--height", type=int, default=256)
    argparse.add_argument("--width", type=int, default=448)

    # Misc
    argparse.add_argument("--precision", type=str, default="fp16")
    argparse.add_argument("--save_with_prompt", default=False, action="store_true",
                          help="Save the video with the prompt string")

    opt = argparse.parse_args()

    set_seed(opt.seed)

    dataset = LoadEmbedding(N=opt.n_samples,
                            shuffle=opt.shuffle,
                            data_path=opt.data_path,
                            txt_path=opt.txt_path,
                            n_sample_frames=opt.num_frames,
                            rag_n_sample_frames=opt.num_frames,
                            k=opt.k)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    # Load the primary models once
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(opt.zeroscope_path)

    # Accelerator
    accelerator = Accelerator(mixed_precision=opt.precision)
    noise_scheduler, tokenizer, text_encoder, vae, unet, dataloader = accelerator.prepare(noise_scheduler, tokenizer, text_encoder, vae, unet, dataloader)
    text_encoder.eval()
    vae.eval()
    unet.eval()
    vae.enable_slicing()

    # Run the inference
    with accelerator.autocast():
        main(opt.checkpoint, opt, accelerator, dataloader, tokenizer, text_encoder, vae, unet)