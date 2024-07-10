from tqdm import tqdm
import faiss
import numpy as np
import torch
from src.utils.utils import *

class DBSCANFAISSClustering:
    """
    Description here.

    Parameters
    ----------
    parameter: Type
        Description

    Attributes
    ----------
    attribute : Type
        Description
    """

    def __init__(
        self,
        eps,
        emb_num,
        emb_size=128,
        batch_queries=512,
        k=20
    ):
        self.emb_size = emb_size
        self.index = faiss.IndexFlatL2(emb_size)  # build an index based on Euclidean distance (L2)
        self.n_batches = batch_queries
        self.k = k
        self.eps = eps
        self.clusters = np.full(emb_num, np.inf)

    def fit(self, X):
        num_nodes = X.shape[0]
        print(f"Loaded {num_nodes} embeddings with size {self.emb_size}")
        self.index.add(X)
        print(f"{self.index.ntotal} embeddings indexed")
        print(f"Retrieving recommendations with batches of {self.n_batches} queries")
        for i in tqdm(range(0, num_nodes, self.n_batches)):
            D, I = self.index.search(X[i:i + self.n_batches], self.k)
            indexes = get_indexes_larger_than_threshold(D, self.eps)
            for j in range(indexes.shape[0]):
                if indexes[j] > 1:
                    old_cluster_index = np.min(self.clusters[I[j, :int(indexes[j])]])  # minimum value
                    cluster_index = np.min(I[j, :int(indexes[j])])
                    self.clusters[I[j, :int(indexes[j])]] = min(old_cluster_index, cluster_index)

        self.clusters[self.clusters == np.inf] = -1

