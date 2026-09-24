import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

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

# DataLoaders
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


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device utilisé : {device}")


def train_model(opt_name, learning_rate=0.001, epochs=30):

    # Création du modèle
    model = MLP(
        input_size=12,
        hidden_size=128
    ).to(device)

    # Fonction de perte
    criterion = nn.BCELoss()

    # Choix de l'optimiseur
    if opt_name == "SGD":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Momentum":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    elif opt_name == "RMSprop":

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    else:
        raise ValueError(f"Optimiseur inconnu : {opt_name}")

    # TensorBoard
    writer = SummaryWriter(
        f"runs/cardio_{opt_name}_lr{learning_rate}"
    )

    # Boucle d'entraînement
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)

            # Remise à zéro des gradients
            optimizer.zero_grad()

            # Forward + loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()

            # Mise à jour des poids
            optimizer.step()

            running_loss += loss.item()

        # Loss moyenne de l'époque
        average_loss = running_loss / len(train_loader)

        # Logging dans TensorBoard
        writer.add_scalar(
            "Training Loss",
            average_loss,
            epoch
        )

        print(
            f"[{opt_name}] "
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    writer.close()

    return model


# Lancement des expériences
if __name__ == "__main__":

    for opt in ["SGD", "Momentum", "RMSprop", "Adam"]:
        train_model(
            opt,
            learning_rate=0.001
        )