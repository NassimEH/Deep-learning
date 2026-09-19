import os, random, datetime, torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

hparams = dict(model="MLP", batch_size=32, lr=1e-2, seed=0, weight_decay=0.0)

run_name = f"{hparams['model']}/bs{hparams['batch_size']}_lr{hparams['lr']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
logdir = os.path.join("runs", run_name)
print("Logdir:", logdir)

writer = SummaryWriter(log_dir=logdir)

N = len(trainset)
val_size = int(0.1 * N)
train_size = N - val_size

train_subset, val_subset = random_split(
    trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(0)
)

trainloader = DataLoader(
    train_subset,
    batch_size=hparams["batch_size"],
    shuffle=True,
    pin_memory=True
)

valloader = DataLoader(
    val_subset,
    batch_size=hparams["batch_size"],
    shuffle=False,
    pin_memory=True
)

@torch.no_grad()
def epoch_metrics(loader, model, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    loss_avg = loss_sum / total
    acc = correct / total
    return loss_avg, acc

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

global_step = 0
EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss_sum, running_total = 0.0, 0

    for b, (x, y) in enumerate(trainloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if b % 10 == 0:
            writer.add_scalar("Loss/train_step", loss.item(), global_step)

        optimizer.step()
        running_loss_sum += loss.item() * y.size(0)
        running_total += y.size(0)
        global_step += 1

    train_loss = running_loss_sum / running_total
    val_loss, val_acc = epoch_metrics(valloader, model, criterion, device)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    print(
        f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
    )

writer.close()
