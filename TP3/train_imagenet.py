"""
CSC 8607 – TP3 : Réseaux convolutifs (ImageNet, MLP vs AlexNet)
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageNet

data_dir = "/mnt/ssd/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 1. Pipeline d'entraînement (avec Data Augmentation)
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# 2. Pipeline de validation (déterministe)
val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def get_dataloaders(batch_size=256, n_workers=8, subset_size=None):
    print("Chargement d'ImageNet...")
    train_dataset = ImageNet(root=data_dir, split="train", transform=train_transform)
    val_dataset = ImageNet(root=data_dir, split="val", transform=val_transform)

    # Optionnel : sous-ensemble pour tester rapidement le code en séance
    if subset_size is not None:
        generator = torch.Generator().manual_seed(42)
        train_indices = torch.randperm(len(train_dataset), generator=generator)[:subset_size]
        val_indices = torch.randperm(len(val_dataset), generator=generator)[: subset_size // 4]
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


class MLP(nn.Module):
    def __init__(self, input_size=3 * 227 * 227, hidden_size=256, num_classes=1000):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetDeep(nn.Module):
    """AlexNet + 3 convolutions 256×3×3 (padding=1) + ReLU juste après Conv5."""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 3 couches supplémentaires après Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(loader, model, criterion):
    model.eval()  # Désactive le Dropout !
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_model(
    model_name="alexnet",
    batch_size=256,
    lr=1e-4,
    num_epochs=20,
    weight_decay=1e-5,
    patience=5,
    subset_size=20000,
):
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, subset_size=subset_size)

    if model_name == "alexnet":
        model = AlexNet(num_classes=1000).to(device)
    elif model_name == "mlp":
        model = MLP(num_classes=1000).to(device)
    elif model_name == "alexnet_deep":
        model = AlexNetDeep(num_classes=1000).to(device)
    else:
        raise ValueError("Modèle inconnu")

    print(f"Modèle {model_name} initialisé avec {count_parameters(model):,} paramètres.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    subset_tag = "full" if subset_size is None else str(subset_size)
    run_tag = f"{model_name}_sub{subset_tag}_lr{lr}"
    writer = SummaryWriter(log_dir=f"runs/{run_tag}")

    best_val_loss = float("inf")
    early_stop_counter = 0
    epoch_durations = []

    for epoch in range(num_epochs):
        model.train()  # Active le Dropout !
        running_loss, correct_train, total_train = 0.0, 0, 0
        t0 = time.perf_counter()

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        val_loss, val_acc = evaluate(val_loader, model, criterion)

        epoch_time = time.perf_counter() - t0
        epoch_durations.append(epoch_time)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("Time/EpochSeconds", epoch_time, epoch)
        writer.flush()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.4f}) | "
            f"Time: {epoch_time:.1f}s"
        )

        # --- EARLY STOPPING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            ckpt_path = f"checkpoints/{run_tag}_best.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping déclenché !")
                break

    writer.close()
    if epoch_durations:
        mean_epoch = sum(epoch_durations) / len(epoch_durations)
        print(f"Durée moyenne d'une époque : {mean_epoch:.1f}s ({mean_epoch / 60:.2f} min)")
        print(f"Estimation pour 100 époques : {100 * mean_epoch / 3600:.2f} h")
    return model


def print_parameter_counts():
    mlp = MLP()
    alex = AlexNet()
    alex_deep = AlexNetDeep()
    print(f"MLP paramètres          : {count_parameters(mlp):,}")
    print(f"AlexNet paramètres      : {count_parameters(alex):,}")
    print(f"AlexNetDeep paramètres  : {count_parameters(alex_deep):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP3 ImageNet – MLP vs AlexNet")
    parser.add_argument(
        "--model",
        type=str,
        default="alexnet",
        choices=["alexnet", "mlp", "alexnet_deep"],
        help="Architecture à entraîner",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Taille du sous-ensemble train (None = ImageNet complet)",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--count_only",
        action="store_true",
        help="Affiche uniquement le nombre de paramètres puis quitte",
    )
    args = parser.parse_args()

    print(f"Device : {device}")
    print_parameter_counts()

    if args.count_only:
        raise SystemExit(0)

    # Exercice 4 : entraînement long AlexNet sur ImageNet complet (défaut sbatch)
    # Mode rapide séance :
    #   python -u train_imagenet.py --model mlp --subset_size 20000 --num_epochs 10
    #   python -u train_imagenet.py --model alexnet --subset_size 20000 --num_epochs 10
    #   python -u train_imagenet.py --model alexnet_deep --subset_size 20000 --num_epochs 10
    train_model(
        model_name=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        patience=args.patience,
        subset_size=args.subset_size,
    )
