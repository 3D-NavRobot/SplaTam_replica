# train_pruner.py
from utils.pruning_mlp import GaussianImportanceNet
import torch
from torch.utils.data import DataLoader, TensorDataset

# Dummy training data for illustration — you will replace this with real features + labels
x = torch.rand(1000, 4)  # features: [opacity, radius, lifespan, rel_radius]
y = (x[:, 0] > 0.5).float().unsqueeze(1)  # label: keep if opacity > 0.5

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GaussianImportanceNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

for epoch in range(10):
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "models/pruning_mlp.pt")
