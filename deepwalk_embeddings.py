import dgl
from torch.utils.data import DataLoader
from torch.optim import SparseAdam
from dgl.sampling import pack_traces
from dgl.sampling import random_walk
from dataset import *
from models import DeepWalk

dataset = OpenAIRESubgraph(raw_dir="dataset/raid_test_dump2/", save_dir="dataset/raid_test_dump2/")

graph = dataset.get_graph()

print(graph)

exit()

# Edges
# Project <- Produces/IsProducedBy -> ResearchProduct
# coproduced_edges = dgl.metapath_reachable_graph(graph, ["IsProducedBy", "Produces"]).edges()

# ResearchProduct <- HasPartOf/IsPartOf -> ResearchProduct
parts_edges = graph.edges(etype="HasPartOf")

# ResearchProduct <- HasVersion/IsVersionOf -> ResearchProduct
versions_edges = graph.edges(etype="HasVersion")

# ResearchProduct <- IsSupplementTo/IsSupplementedBy -> ResearchProduct
supplements_edges = graph.edges(etype="IsSupplementTo")

# ResearchProduct <- References/IsReferencedBy -> ResearchProduct
references_edges = graph.edges(etype="References")

# All edges
edges_src = torch.cat((parts_edges[0], versions_edges[0], supplements_edges[0], references_edges[0]), dim=0)
edges_dst = torch.cat((parts_edges[1], versions_edges[1], supplements_edges[1], references_edges[1]), dim=0)

graph = dgl.graph((edges_src, edges_dst), num_nodes=graph.num_nodes("research_product"))

model = DeepWalk(graph)
dataloader = DataLoader(torch.arange(graph.num_nodes()), batch_size=128, shuffle=True, collate_fn=model.sample)
optimizer = SparseAdam(model.parameters(), lr=0.01)
num_epochs = 400
train_size = len(dataloader.dataset)
early_stopping = 20
min_loss = np.inf

print("Starting training process")
counter = early_stopping
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_walk in dataloader:
        loss = model(batch_walk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_walk.batch_size
    epoch_loss /= train_size
    print(type(model.node_embed))
    print(f"Epoch {epoch:03d} - Loss: {epoch_loss:.4f}")
    # early stopping
    if min_loss >= epoch_loss:
        counter = early_stopping
        min_loss = epoch_loss
    counter -= 1
    if counter <= 0:
        print("Early stopping!")
        break

# torch.save(model.node_embed, "dataset/deepwalk_embeddings.pt")

