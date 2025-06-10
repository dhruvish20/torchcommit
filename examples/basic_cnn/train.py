import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# from torchcommit.tracker import ExperimentTracker  # Adjust import based on your structure

# Simple model definition
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Dummy config
config = {
    "epochs": 2,
    "batch_size": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "dataset": "MNIST"
}

# Init tracker
# tracker = ExperimentTracker(
#     model=model,
#     optimizer=optimizer,
#     loss_fn=loss_fn,
#     config=config,
#     project_name="torchcommit-test"
# )

# tracker.start()

# Data
train_loader = DataLoader(
    datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=config["batch_size"],
    shuffle=True
)

# Training loop
for epoch in range(config["epochs"]):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    # tracker.log(epoch=epoch, metrics={"loss": avg_loss})

# tracker.end()

print("Training complete.")
print(f"Final loss: {avg_loss:.4f}")
