import dgl
from torch.utils.data import DataLoader
from torch.optim import SparseAdam
from dataset import *
from dgl.nn.pytorch import MetaPath2Vec

dataset = OpenAIRESubgraph(raw_dir="dataset/raid_test_dump/", save_dir="dataset/raid_test_dump/")
graph = dataset.get_graph()

model = MetaPath2Vec(graph, ["IsProducedBy", "Produces"], window_size=1)
dataloader = DataLoader(torch.arange(graph.num_nodes("research_product")), batch_size=128, shuffle=True, collate_fn=model.sample)
optimizer = SparseAdam(model.parameters(), lr=0.025)
num_epochs = 400
train_size = len(dataloader.dataset)
early_stopping = 20
min_loss = np.inf

print("Starting training process")
counter = early_stopping
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for (pos_u, pos_v, neg_v) in dataloader:
        loss = model(pos_u, pos_v, neg_v)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(pos_u)
    epoch_loss /= train_size
    print(f"Epoch {epoch:03d} - Loss: {epoch_loss:.4f}")
    # early stopping
    if min_loss >= epoch_loss:
        counter = early_stopping
        min_loss = epoch_loss
    counter -= 1
    if counter <= 0:
        print("Early stopping!")
        break

rp_nids = torch.LongTensor(model.local_to_global_nid['research_product'])
rp_emb = model.node_embed(rp_nids)
torch.save(rp_emb, "dataset/mp2v_rp2p_embeddings.pt")

