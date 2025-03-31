import os
from pathlib import Path
from typing import Tuple
import faiss


import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from utils import l2_normalise
import json
from collections import defaultdict

class RetrievalDatabaseMetadataProvider:
    """Metadata provider for the retrieval database.

    Args:
        metadata_dir (str): Path to the metadata directory.
    """

    def __init__(self, metadata_dir: str):
        with open(metadata_dir) as f:
            self.metadata_list = f.readlines()

    def __len__(self):
        return len(self.metadata_list)

    def get(self, ids, columns=None):
        """Get the metadata for the given ids.

        Args:
            ids (list): List of ids.
        """
        metadata = {key: [] for key in columns}
        for id in ids:
            with open(Path(self.metadata_list[id]).with_suffix(".json"), "r") as f:
                data = json.load(f)
                for col in columns:
                    metadata[col].append(data[col])
        return tuple(metadata.values())


class RetrievalDatabase:
    """Retrieval database.

    Args:
        database_dir (str): Path to the index directory.
    """

    def __init__(self,
                 file_path: str,
                 video_index_fp: str,
                 radious: float = 0.98,
                 deduplication_threshold: float = 0.94,
                 k : int = 10):

        # RAG
        self._k = k

        video_index = faiss.read_index(str(video_index_fp), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        metadata_provider = RetrievalDatabaseMetadataProvider(file_path)

        self._video_index = video_index
        self._metadata_provider = metadata_provider
        self._radious = radious
        self._deduplication_threshold = deduplication_threshold

    def _get_connected_components(self, neighbors):
        """Find connected components in a graph.

        Args:
            neighbors (dict): Dictionary of neighbors.
        """
        seen = set()

        def component(node):
            r = []
            nodes = {node}
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in neighbors:
            if node not in seen:
                u.append(component(node))
        return u

    def _deduplicate_embeddings(self, embeddings, threshold=0.92):
        """Deduplicate embeddings.

        Args:
            embeddings (np.matrix): Embeddings to deduplicate.
            threshold (float): Threshold to use for deduplication. Default is 0.94.
        """
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        l, _, indices = index.range_search(embeddings, threshold)

        same_mapping = defaultdict(list)

        for i in range(embeddings.shape[0]):
            start = l[i]
            end = l[i + 1]
            for j in indices[start:end]:
                same_mapping[int(i)].append(int(j))

        groups = self._get_connected_components(same_mapping)
        non_uniques = set()
        for g in groups:
            for e in g[1:]:
                non_uniques.add(e)

        return set(list(non_uniques))

    def __call__(self, query, k=10):

        # Load embeddings
        # query = l2_normalise(query)
        # query = np.mean(query, axis=0, keepdims=True, dtype=np.float32)
        # query = l2_normalise(query)
        index = self._video_index
        # Search in the index
        distances, indices, embeddings = index.search_and_reconstruct(query, k)
        results = [indices[i] for i in range(len(indices))]

        # nb_results = [np.where(r == -1)[0] for r in results]
        total_distances = []
        total_indices = []
        total_embeddings = []
        for i in range(len(results)):
            # num_res = nb_results[i][0] if len(nb_results[i]) > 0 else len(results[i])

            valid_indices = distances[i] < self._radious
            result_indices = results[i][valid_indices]
            result_distances = distances[i][valid_indices]
            result_embeddings = embeddings[i][valid_indices]

            # normalise embeddings
            l2 = np.atleast_1d(np.linalg.norm(result_embeddings, 2, -1))
            l2[l2 == 0] = 1
            result_embeddings = result_embeddings / np.expand_dims(l2, -1)

            # deduplicate embeddings
            local_indices_to_remove = self._deduplicate_embeddings(result_embeddings, self._deduplication_threshold)
            indices_to_remove = set()
            for local_index in local_indices_to_remove:
                indices_to_remove.add(result_indices[local_index])

            curr_indices = []
            curr_distances = []
            curr_embeddings = []
            for ind, dis, emb in zip(result_indices, result_distances, result_embeddings):
                if ind not in indices_to_remove:
                    indices_to_remove.add(ind)
                    curr_indices.append(ind)
                    curr_distances.append(dis)
                    curr_embeddings.append(emb)

            total_indices.append(curr_indices)
            total_distances.append(curr_distances)
            total_embeddings.append(curr_embeddings)

        if len(total_distances) == 0:
            return []

        total_results = []
        for i in range(len(total_indices)):
            metadata = self._metadata_provider.get(total_indices[i], columns=["videoID", "txt", "videoLoc"])
            metadata += (total_distances[i],)
            total_results.append(metadata)

        return total_results

def main(
         txt: str,
         output_path: str,
         metadata_dir: str,
         video_index_fp: str,
         k: int = 5,
         radious: float = 1.01,
         deduplication_threshold: float = 0.98,
         bs: int = 16,
         ):

    output_path = Path(output_path) / f"nn_{k}-rad_{radious}-dedup_{deduplication_threshold}"
    (output_path).mkdir(exist_ok=True, parents=True)

    database = RetrievalDatabase(file_path=metadata_dir,
                                 video_index_fp=video_index_fp,
                                 radious=radious,
                                 deduplication_threshold=deduplication_threshold)
    # Read the text emebddings
    txt = Path(txt)
    txt = list(sorted(txt.glob("*.npy")))

    for query_videoid in tqdm(txt, desc="Processing embeddings"):
        # load embedding
        query = np.load(query_videoid)[None]
        query = l2_normalise(query)
        with open(query_videoid.with_suffix(".json"), "r") as f:
            data = json.load(f)
            original_caption = data["txt"]

        # retrieve until we have k elements or max_iters is reached
        if query.ndim == 3:
            query = query.squeeze(0)
        all_data = database(query, k=k * 10)

        # save results
        for data in all_data:
            ids, caption, videoloc, distances = data
            distances = [str(s) for s in distances]
            with open(output_path / f"{query_videoid.stem}.json", "w") as f:
                json.dump({"videoID": ids[:k], "txt": caption[:k], "videoLoc": videoloc[:k], "dist": distances[:k]}, f)

if __name__ == '__main__':
    import fire
    fire.Fire(main)