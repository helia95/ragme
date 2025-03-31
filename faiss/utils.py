import numpy as np
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import fire
import math
from itertools import islice
import fsspec
from typing import List, Tuple, Union

# Copied from https://github.com/rom1504/embedding-reader/blob/main/embedding_reader/get_file_list.py
def make_path_absolute(path: str) -> str:
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return os.path.abspath(p)
    return path

def get_file_list(
    path: str, file_format: str, sort_result: bool = True
) -> Tuple[fsspec.AbstractFileSystem, List[str]]:
    """Get the file system and all the file paths that matches `file_format` given a single path."""
    path = make_path_absolute(path)
    fs, path_in_fs = fsspec.core.url_to_fs(path)
    prefix = path[: -len(path_in_fs)]
    glob_pattern = path.rstrip("/") + f"/**/*.{file_format}"
    file_paths = fs.glob(glob_pattern)
    if sort_result:
        file_paths.sort()
    file_paths_with_prefix = [prefix + file_path for file_path in file_paths]
    return fs, file_paths_with_prefix


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def file_to_count(filename):
    with open(filename, "rb") as f:
        parquet_file = pq.ParquetFile(f, memory_map=True)
        return parquet_file.metadata.num_rows


def count_samples(files):
    total_count = 0
    with ThreadPool(10) as p:
        for c in tqdm(p.imap(file_to_count, files), total=len(files)):
            total_count += c
    return total_count


def parquet_to_arrow(parquet_folder, output_arrow_folder, columns_to_return):
    """convert the parquet files into arrow files"""
    os.makedirs(output_arrow_folder, exist_ok=True)
    data_dir = Path(parquet_folder)
    files = sorted(data_dir.glob("*.parquet"))
    number_samples = count_samples(files)
    print("There are {} samples in the dataset".format(number_samples))  # pylint: disable=consider-using-f-string

    schema = pq.read_table(files[0], columns=columns_to_return).schema
    sink = None
    current_batch_count = 0
    batch_counter = 0
    key_format = max(0, int(math.log10(number_samples / 10**10))) + 1
    for parquet_files in tqdm(files):
        if sink is None or current_batch_count > 10**10:
            if sink is not None:
                writer.close()
                sink.close()
            file_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
                key_format=key_format, true_key=batch_counter
            )
            file_name = f"{output_arrow_folder}/{file_key}.arrow"
            print(f"Writing to {file_name}")
            sink = pa.OSFile(file_name, "wb")
            writer = pa.ipc.new_file(sink, schema)
            current_batch_count = 0
            batch_counter += 1

        print("going to read parquet file: ", parquet_files)
        for i in range(2):
            try:
                table = pq.read_table(parquet_files, columns=columns_to_return, use_threads=False)
            except Exception as e:  # pylint: disable=broad-except
                if i == 1:
                    raise e
                print("Error reading parquet file: ", e)
                print("Retrying once...")
                continue
        writer.write_table(table)
        current_batch_count += table.num_rows
    if sink is not None:
        writer.close()
        sink.close()



def l2_normalise(x):
    l2 = np.linalg.norm(x, 2, axis=-1, keepdims=True)
    l2[l2 == 0] = 1
    return x / l2


from PIL import Image

def min_res(size, min_size): return 192 if size < 192 else size

def up_down_bucket(m_size, in_size, direction):
    if direction == 'down': return abs(int(m_size - in_size))
    if direction == 'up': return abs(int(m_size + in_size))

def get_bucket_sizes(size, direction: 'down', min_size):
    multipliers = [64, 128]
    for i, m in enumerate(multipliers):
        res =  up_down_bucket(m, size, direction)
        multipliers[i] = min_res(res, min_size=min_size)
    return multipliers

def closest_bucket(m_size, size, direction, min_size):
    lst = get_bucket_sizes(m_size, direction, min_size)
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i]-size))]

def resolve_bucket(i,h,w): return  (i / (h / w))

def sensible_buckets(m_width, m_height, w, h, min_size=192):
    if h > w:
        w = resolve_bucket(m_width, h, w)
        w = closest_bucket(m_width, w, 'down', min_size=min_size)
        return w, m_height
    if h < w:
        h = resolve_bucket(m_height, w, h)
        h = closest_bucket(m_height, h, 'down', min_size=min_size)
        return m_width, h

    return m_width, m_height