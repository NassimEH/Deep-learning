import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import CardioDataset


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Chargement du dataset
dataset = CardioDataset("data/cardio_train.csv")

# Découpage 80% train, 10% validation, 10% test
generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    dataset,
    [0.8, 0.1, 0.1],
    generator=generator
)

# Création des DataLoaders
train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)


# Initialisation du device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device utilisé : {device}")


# Initialisation du modèle
batch = next(iter(train_loader))

model = MLP(
    input_size=batch["features"].shape[1],
    hidden_size=128
).to(device)


# Fonction de perte
criterion = nn.BCELoss()

# Optimiseur
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)


# Coefficients de régularisation
l1_lambda = 1e-4
l2_lambda = 1e-3


# Boucle d'entraînement
for epoch in range(10):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:

        inputs = batch["features"].to(device)
        targets = batch["labels"].to(device)

        # Remise à zéro des gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)

        # Loss de base
        base_loss = criterion(outputs, targets)

        # Pénalité L1
        l1_penalty = sum(
            p.abs().sum()
            for p in model.parameters()
        )

        # Pénalité L2
        l2_penalty = sum(
            (p ** 2).sum()
            for p in model.parameters()
        )

        # Loss totale
        loss = (
            base_loss
            + l1_lambda * l1_penalty
            + l2_lambda * l2_penalty
        )

        # Backpropagation
        loss.backward()

        # Mise à jour des poids
        optimizer.step()

        # Statistiques
        total_loss += loss.item()

        predictions = (outputs >= 0.5).float()

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    average_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/10 "
        f"- Loss: {average_loss:.4f} "
        f"- Accuracy: {accuracy:.2f}%"
    )