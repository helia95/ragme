import fire
from pathlib import Path
import torch
import open_clip
import pyarrow.parquet as pq
import itertools
import pdb
from tqdm import tqdm
import accelerate
import pyarrow as pa
from utils import l2_normalise
import itertools
import numpy as np
import json
import pyarrow.parquet as pq

# README: Use the first function to encode from a list of promtps at inference, use the second to encode a bunch of parquet files

def clip_text_encode(txt_file: str, output_path: str, batch_size: int = 32, full_precision: bool = False):
    """
    Given a list of prompts in a txt file, return the embeddings of each of them and the corresponding txt.
    Args:
        txt_file (str): Path to the txt file with the prompts.
        output_path (str): Path to the output folder.
        batch_size (int): Batch size for the encoding.
        full_precision (bool): Whether to use full precision or not.
    """
    # ==== Read the file
    if txt_file.endswith(".parquet"):
        table = pq.read_table(txt_file)
        file_names = table["videoID"].to_pandas().tolist()
        prompts = table["txt"].to_pandas().tolist()
    else:
        prompts = open(txt_file).readlines()
        file_names = [f"{str(i).zfill(3)}" for i in range(len(prompts))]

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    accelerator = accelerate.Accelerator(mixed_precision="full" if full_precision else "fp16")
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=accelerator.device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model, tokenizer = accelerator.prepare([model, tokenizer])

    n_files = len(prompts)
    for start in tqdm(range(0, n_files, batch_size)):
        batch_prompts = prompts[start:start + batch_size]
        batch_names = file_names[start:start + batch_size]
        with accelerator.autocast():
            with torch.no_grad():
                text_features = tokenizer(batch_prompts).to(accelerator.device)
                text_features = model.encode_text(text_features)
                text_features = text_features.cpu().numpy()

        for name, txt, feat in zip(batch_names, batch_prompts, text_features):
            np.save(output_path / f"{name}.npy", feat[None])  # if having problems with .txt files remove the None
            with open(output_path / f"{name}.json", "w") as f:
                json.dump({"txt": txt.strip()}, f)

if __name__ == "__main__":
    fire.Fire(clip_text_encode)