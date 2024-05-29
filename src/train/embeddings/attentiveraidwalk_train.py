from tqdm import tqdm
from models import RAiDAttentiveWalk
from torch.utils.data import DataLoader
from torch.optim import SparseAdam
from src.utils.dataset import *

# parameters
dataset_base_dir = "../../dataset/raid_test_dump2/"
embeddings_path = "../../dataset/AttentiveRAiDWalk128_embeddings.pt"
num_epochs = 400
early_stopping = 20
batch_size = 512
embedding_dim = 128
random_walk_length = 20
window_size = 5

dataset = OpenAIRESubgraph(raw_dir=dataset_base_dir, save_dir=dataset_base_dir)

graph = dataset.get_graph()
print(graph)

model = RAiDAttentiveWalk(graph, walk_length=random_walk_length, window_size=window_size, emb_dim=embedding_dim)
dataloader = DataLoader(torch.arange(graph.num_nodes('research_product')), batch_size=batch_size, shuffle=True, collate_fn=model.sample)
optimizer = SparseAdam(model.parameters(), lr=0.01)

train_size = len(dataloader.dataset)
min_loss = np.inf

print("Starting training process")
counter = early_stopping
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_walk in tqdm(dataloader):
        loss = model(batch_walk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_walk)
    epoch_loss /= train_size
    print(f"Epoch {epoch:03d} - Loss: {epoch_loss:.4f}")
    # early stopping
    if min_loss >= epoch_loss:
        counter = early_stopping
        min_loss = epoch_loss
        # save embeddings
        torch.save(model.node_embed.weight.detach(), embeddings_path)
    counter -= 1
    if counter <= 0:
        print("Early stopping!")
        break
