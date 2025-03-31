import numpy as np
from pathlib import Path
import pdb
import json
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import os
import fire
from utils import l2_normalise, batched, get_file_list
from multiprocessing.pool import ThreadPool

def preprocess_video_embedding_parquet(embeddings_paths: str,
                               output_path: str,
                               pre_norm: bool = True,
                               post_norm: bool = True,
                               files_per_chunk: int = 10000):
    """
    Preprocess the video embeddings, compute the average on the temporal dimension and store the results.
    """

    # Emb path
    embeddings_paths = Path(embeddings_paths)
    embeddings_paths = sorted(list(embeddings_paths.glob("*.npy")))

    iterator = batched(embeddings_paths, files_per_chunk)

    # Create output folder
    output_path = Path(output_path)
    (output_path / "embeddings").mkdir(exist_ok=True, parents=True)
    (output_path / "metadata").mkdir(exist_ok=True, parents=True)

    counter = 0
    n_shards = len(embeddings_paths) // files_per_chunk
    for shard in tqdm(iterator, desc="Processing embeddings"):
        print(f"Processing {counter} / {n_shards}")

        embeddings_data = dict(embeddings=[])
        metadata_data = dict(videoLoc=[], videoID=[], txt=[])
        for path in shard:
            # Load the embeddings
            embeddings = np.load(path)

            # Normalise embeddings
            embeddings = l2_normalise(embeddings) if pre_norm else embeddings

            # Compute the mean and normalise
            embeddings = np.mean(embeddings, axis=0, keepdims=False, dtype=np.float32)
            embeddings = l2_normalise(embeddings) if post_norm else embeddings

            with open(path.with_suffix(".json"), "r") as f:
                json_data = json.load(f)

            # Store the data
            embeddings_data["embeddings"].append(embeddings)
            metadata_data["videoLoc"].append(json_data["videoLoc"])
            metadata_data["videoID"].append(json_data["videoID"])
            metadata_data["txt"].append(json_data["txt"])

        # Save the data
        embeddings_table = pa.Table.from_pydict(embeddings_data)
        pq.write_table(embeddings_table, output_path / "embeddings" / f"{counter:04d}.parquet")

        metadata_table = pa.Table.from_pydict(metadata_data)
        pq.write_table(metadata_table, output_path / "metadata" / f"{counter:04d}.parquet")

        # Increase the counter
        counter += 1


def _compute_mean_and_save(input_path, output_path, pre_norm=True, post_norm=True):
    """
    Compute the mean of the embeddings and save the results.
    """

    # Load the embeddings
    embeddings = np.load(input_path)
    dtype = embeddings.dtype

    # Normalise embeddings
    embeddings = l2_normalise(embeddings) if pre_norm else embeddings

    # Compute the mean and normalise
    embeddings = np.mean(embeddings, axis=0, keepdims=True, dtype=np.float32)
    embeddings = l2_normalise(embeddings) if post_norm else embeddings

    # Save the data
    np.save(output_path, embeddings.astype(dtype))

def preprocess_video_embedding_numpy(embeddings_paths: str,
                                       output_path: str,
                                       pre_norm: bool = True,
                                       post_norm: bool = True):
    """
    Preprocess the video embeddings, compute the average on the temporal dimension and store the results.
    """

    # Emb path
    # embeddings_paths = Path(embeddings_paths)
    embeddings_paths = list(Path(embeddings_paths).glob("*.npy")) # get_file_list(embeddings_paths, "npy")

    # Create output folder
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    def process_file(file_path):
        output_file_path = output_path / file_path.name
        _compute_mean_and_save(file_path, output_file_path, pre_norm, post_norm)

    # Use a ThreadPool to process files concurrently
    with ThreadPool(128) as pool:
        list(tqdm(pool.map(process_file, embeddings_paths), total=len(embeddings_paths)))

if __name__ == "__main__":
    fire.Fire(preprocess_video_embedding_numpy)