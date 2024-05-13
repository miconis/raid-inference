from dgl.sampling import *
from tqdm import tqdm
from dataset import *
from models import RAiDAttentiveWalk
from dgl.nn.pytorch import DeepWalk
from torch.utils.data import DataLoader
from torch.optim import SparseAdam

dataset = OpenAIRESubgraph(raw_dir="dataset/raid_test_dump2/", save_dir="dataset/raid_test_dump2/")

graph = dataset.get_graph()
print(graph)

model = RAiDAttentiveWalk(graph, walk_length=20, window_size=5)
dataloader = DataLoader(torch.arange(graph.num_nodes('research_product')), batch_size=512  , shuffle=True, collate_fn=model.sample)

optimizer = SparseAdam(model.parameters(), lr=0.01)
num_epochs = 400
train_size = len(dataloader.dataset)
early_stopping = 20
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
    counter -= 1
    if counter <= 0:
        print("Early stopping!")
        break

torch.save(model.node_embed.weight.detach(), "dataset/AttentiveRAiDWalk_embeddings.pt")