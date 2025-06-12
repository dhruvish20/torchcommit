import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchcommit.tracker import ExperimentTracker

class ConvolutionLayer(nn.Module):
    def __init__(self):
        super(ConvolutionLayer, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionLayer().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader = DataLoader(
    datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=32,
    shuffle=True
)

tracker = ExperimentTracker(
    model=model,
    optimizer=optimizer,
    loss_function=loss_fn,
    dataloader=train_loader,
    checkpoints=True
)

tracker.start()

num_epochs = 2
for epoch in range(num_epochs):
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

    tracker.log(epoch=epoch, metrics={"avg_loss": avg_loss})
    print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

tracker.end()

print("✅ Training complete.")
print(f"📦 Final loss: {avg_loss:.4f}")
