import torch
from algorithms import DBSCANFAISSClustering
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import numpy as np

# parameters
k = 20  # number of nearest neighbors
embeddings_path = "../../../dataset/AttentiveRAiDWalk128_embeddings.pt"
clusters_path = "test_clusters.pt"

embeddings = torch.load(embeddings_path)

print("Computing KNN")
knn = NearestNeighbors(n_neighbors=k).fit(embeddings)
distances, indices = knn.kneighbors(embeddings)
# sort in descending order avg distance in the neighborhood (excluding reference node with distance 0)
distance_desc = sorted(np.mean(distances[:, 1:k], axis=1), reverse=True)

print("Plotting K-distance graph")
plt.title("K-distance plot")
plt.ylabel("eps")
plt.plot(list(range(1, len(distance_desc)+1)), distance_desc, "b", label="normalized curve")

print("Computing elbow point")
kneedle = KneeLocator(range(1, len(distance_desc)+1),
                      distance_desc,
                      S=1.0,
                      curve="convex",
                      direction="decreasing")
print("Plotting elbow graph")
kneedle.plot_knee_normalized()
plt.show()

eps = kneedle.knee_y
print("Estimated eps with elbow at ", eps)

print("Starting clustering")
model = DBSCANFAISSClustering(eps=eps, emb_num=embeddings.shape[0], emb_size=embeddings.shape[1])

model.fit(embeddings)

print(f"Saving clusters at {clusters_path}")
torch.save(torch.from_numpy(model.clusters), clusters_path)

