# DBSCAN CLUSTERING:
# 1 - it computes the k-nn distance graph to estimate the max distance for 2 nodes to be in same cluster (eps)
# 2 - it computes the dbscan clusters using the estimated parameter

import torch
from sklearn.cluster import DBSCAN, OPTICS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

k = 20  # number of nearest neighbors
embeddings = torch.load("../dataset/AttentiveRAiDWalk_embeddings.pt").numpy()

# # 1 - Compute KNN
# print("Computing KNN")
# knn = NearestNeighbors(n_neighbors=k).fit(embeddings)
# distances, indices = knn.kneighbors(embeddings)
# # sort in descending order avg distance in the neighborhood (excluding reference node with distance 0)
# distance_desc = sorted(np.mean(distances[:, 0:k], axis=1), reverse=True)
#
# # plot k-distance graph
# print("Plotting K-distance graph")
# plt.title("K-distance plot")
# plt.ylabel("eps")
# plt.plot(list(range(1, len(distance_desc)+1)), distance_desc, "b", label="normalized curve")
#
# # plot elbow
# print("Computing elbow point")
# kneedle = KneeLocator(range(1, len(distance_desc)+1),
#                       distance_desc,
#                       S=1.0,
#                       curve="convex",
#                       direction="decreasing")
# print("Plotting elbow graph")
# kneedle.plot_knee_normalized()
# plt.show()
#
# # compute eps by measuring the elbow point
# eps = kneedle.knee_y  # 3.619 elbow
# print("Estimated eps with elbow at ", eps)

# 2 - Compute DBSCAN clustering
print("Computing DBSCAN clustering")
clustering = (DBSCAN(eps=3.619,  # max distance between two samples to be considered as in the neighborhood of the other
                     min_samples=k,  # minimum number of points for each cluster
                     metric='euclidean',  # metric to be used when calculating distance
                     algorithm='kd_tree',  # for the nearest neighbors
                     leaf_size=50,  # it affects the speed and the memory (higher=less memory usage and less speed)
                     p=2,  # the power of the minkowski metric (2=euclidean)
                     n_jobs=-1)  # number of parallel jobs to run
              .fit(embeddings))

cluster_labels = torch.from_numpy(clustering.labels_)

print("Saving data")
torch.save(cluster_labels, "../dataset/AttentiveRAiDWalk_DBSCAN_labels.pt")