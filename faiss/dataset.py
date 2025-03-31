import os
from pathlib import Path
from typing import Tuple
import faiss


import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
from faiss.utils import l2_normalise, sensible_buckets
import pdb
import random
from einops import repeat, rearrange
# import open_clip
import time
import shutil
from collections import defaultdict
import glob
import json

import imageio
import decord
decord.bridge.set_bridge('torch')


def export_to_video(video_frames, output_video_path, fps=None):
    # remove
    video_frames = video_frames[0].clamp(-1., 1.)
    video_frames = (video_frames + 1.) / 2.
    video_frames = (video_frames.permute(1, 2, 3, 0) * 255.).byte().cpu()

    video = [np.array(img) for img in video_frames]
    if output_video_path.suffix == '.gif':   imageio.mimsave(output_video_path, video, duration=125, loop=25)
    if output_video_path.suffix == '.mp4':   imageio.mimsave(output_video_path, video, fps=8)

def get_text_prompt(
        text_prompt: str = '',
        fallback_prompt: str = '',
        file_path: str = '',
        ext_types=['.mp4'],
        use_caption=False
):
    try:
        if use_caption:
            if len(text_prompt) > 1: return text_prompt
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file):
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)

            # Return fallback prompt if no conditions are met.
            return fallback_prompt

        return text_prompt
    except:
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt


def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices


def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)

    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr)

    return video, vr

class RetrievalDatabase(torch.utils.data.Dataset):
    def __init__(self, data_path: str,
                 index_path: str,
                 file_path: str,
                 tokenizer=None,
                 k: int = 10, width: int = 256, height: int = 256, n_sample_frames: int = 16,
                 use_bucketing: bool = False, self_conditioning: bool = False, p_cfg: float = 0.2,
                 fps: int = 8, sample_start_idx: int = 1, frame_step: int = 1, rag_n_sample_frames: int = 16,
                 rag_skip_frames: int = 1, validation: bool = False, overfit: bool = False, **kwargs):
        super().__init__()
        """
        Args:
        data_path: str: Path to the embeddings and json files.
        index_config: str: Path to the precomputed nearest neighbors. 
        """

        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.file_path = file_path
        self._make_dataset()

        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.fps = fps
        self.self_conditioning = self_conditioning
        self._k = k if not (self_conditioning and k > 1) else 1
        self.rag_n_sample_frames = rag_n_sample_frames
        self.rag_skip_frames = rag_skip_frames
        self.embedding_dim = 512
        self.p_cfg = p_cfg
        self.validation = validation
        self.overfit = overfit
        self.n_overfit_samples = kwargs.get("n_overfit_samples", None)

    def __len__(self):
        if self.overfit:
            return self.n_overfit_samples
        return len(self.file_ids)

    def _make_dataset(self):
        with open(self.file_path) as f:
            self.file_ids = f.read().splitlines()

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

    def get_prompt_ids(self, prompt):
        return self.tokenizer(prompt, truncation=True, padding="max_length",
                              max_length=self.tokenizer.model_max_length, return_tensors="pt").input_ids

    def _load_video(self, path: str, normalize: bool = True):
        """
        Load a video from a given path and return it as a tensor. Optionally, normalize between -1, +1.
        Args
        path: str: Path to the video file.
        normalize: bool: Whether to normalize the video between -1, +1.
        """
        try:
            video, _ = self.process_video_wrapper(path)
        except:
            print("loading issue")
            video = torch.zeros((1, self.n_sample_frames, 3, self.height, self.width))
        pixel_values = video[0] / 127.5 - 1.0
        pixel_values = F.pad(pixel_values, (0, 0, 0, 0, 0, 0, 0, self.n_sample_frames - pixel_values.shape[0]),
                             value=-1)
        return pixel_values

    def __getitem__(self, idx):

        file_id = self.file_ids[idx]
        file_id = Path(file_id).stem

        with open((self.data_path / file_id).with_suffix('.json'), "r") as f:
            sample_info = json.load(f)
        # assert file_id == sample_info["videoID"]

        # Load text
        use_cfg = random.uniform(0, 1) < self.p_cfg if not self.validation else False
        txt = "" if use_cfg else sample_info["txt"]
        prompt_ids = self.get_prompt_ids(txt)[0] if self.tokenizer is not None else torch.zeros((1, 77))

        # Load video
        videoLoc = sample_info["videoLoc"]
        pixel_values = self._load_video(videoLoc)

        # Load RAG embeddings
        with open((self.index_path / file_id).with_suffix('.json'), "r") as f:
            nearest_neighbors_info = json.load(f)
        nearest_neighbors_ids = nearest_neighbors_info["videoID"][:self._k] if not self.self_conditioning else [file_id]

        # Load the embeddings
        nn_embeddings = torch.zeros((self._k, self.rag_n_sample_frames, self.embedding_dim))
        attn_mask = torch.ones((self._k, self.rag_n_sample_frames), dtype=torch.bool)
        encoder_attn_mask = torch.zeros((self._k, ), dtype=torch.bool)
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

        # === Validation ===
        if self.validation:
            # Replace tokenized prompt with the original text
            prompt_ids = txt
            # Load the CLIP emebedding of the sample
            sample_emb = torch.from_numpy(np.load((self.data_path / file_id).with_suffix('.npy')))
            # Return the retrieved videos
            nn_captions = nearest_neighbors_info["txt"][:self._k]
            nn_distances = np.array([float(s) for s in nearest_neighbors_info["dist"][:self._k]])
            nn_videos = torch.zeros((self._k, self.n_sample_frames, 3, self.height, self.width))

            for i, videoLoc in enumerate(nearest_neighbors_info["videoLoc"][:self._k]):
                # Load the caption
                nn_videos[i] = self._load_video(videoLoc)
        # ===================
        # Prepare the output tuple
        output_tuple = (pixel_values, prompt_ids, nn_embeddings, attn_mask, encoder_attn_mask)
        if self.validation:
            output_tuple += (nn_videos, nn_captions, nn_distances, sample_emb, file_id)

        return output_tuple



###
class OverfitDataset(torch.utils.data.Dataset):
    def __init__(self,
                 video_path: str,
                 txt_prompt: str = "",
                 tokenizer=None,
                 k: int = 10, width: int = 448, height: int = 256, n_sample_frames: int = 16,
                 use_bucketing: bool = False, self_conditioning: bool = False, p_cfg: float = 0.2,
                 fps: int = 8, sample_start_idx: int = 1, frame_step: int = 1, rag_n_sample_frames: int = 16,
                 rag_skip_frames: int = 1, **kwargs):
        super().__init__()
        """
        Args:
        data_path: str: Path to the embeddings and json files.
        index_config: str: Path to the precomputed nearest neighbors. 
        """

        self.video_path = video_path
        self.txt_prompt = txt_prompt

        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.fps = fps
        self.rag_n_sample_frames = rag_n_sample_frames
        self.rag_skip_frames = rag_skip_frames
        self.embedding_dim = 512
        self.p_cfg = p_cfg
        self.use_bucketing = use_bucketing

    def __len__(self):
        return 1

    def get_prompt_ids(self, prompt):
        return self.tokenizer(prompt, truncation=True, padding="max_length",
                              max_length=self.tokenizer.model_max_length, return_tensors="pt").input_ids

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
        effective_idx = 0 # Always sample from the first frame. random.randint(0, (effective_length - n_sample_frames))
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
        try:
            video, _ = self.process_video_wrapper(path)
        except:
            print("loading issue")
            video = torch.zeros((1, self.n_sample_frames, 3, self.height, self.width))
        pixel_values = video[0] / 127.5 - 1.0
        pixel_values = F.pad(pixel_values, (0, 0, 0, 0, 0, 0, 0, self.n_sample_frames - pixel_values.shape[0]),
                             value=-1)
        return pixel_values

    def __getitem__(self, idx):

        # Load text
        prompt_ids = self.get_prompt_ids(self.txt_prompt)[0] if self.tokenizer is not None else torch.zeros((1, 77))

        # Load video
        pixel_values = self._load_video(self.video_path)

        return pixel_values, prompt_ids
    
    