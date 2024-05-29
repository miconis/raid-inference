# FAISS-based recommendation system:
# 1 - it creates an index of the N-dimensional embedding space
# 2 - it retrieves the K nearest neighbors for each node

import faiss
import numpy
import torch
from tqdm import tqdm

k = 20  # number of nearest neighbors to retrieve
n_batches = 100000  # number of batch queries

embeddings = torch.load("../dataset/AttentiveRAiDWalk_embeddings.pt")

num_nodes = embeddings.shape[0]
emb_size = embeddings.shape[1]
print(f"Loaded {num_nodes} embeddings with size {emb_size}")

# 1 - Create the index
print("Indexing in progress")
index = faiss.IndexFlatL2(emb_size)  # build an index based on euclidean distance (L2)
index.add(embeddings)
print(f"{index.ntotal} nodes indexed")

# 2 - Retrieve K nearest neighbors
print(f"Retrieving recommendations with batches of {n_batches} queries")
recommendations = []
for i in tqdm(range(0, num_nodes, n_batches)):
    D, I = index.search(embeddings[i:i+n_batches], k)  # distances, indexes
    recommendations.append(I)

recommendations = numpy.concatenate(recommendations, 0)
recommendations = torch.from_numpy(recommendations)
torch.save(recommendations, "../dataset/AttentiveRAiDWalk_recommendations.pt")
print("Recommendations saved")
