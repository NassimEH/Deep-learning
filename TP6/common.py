import csv
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RAW_ROOT = Path(os.environ.get("TP6_RAW", "/mnt/ssd/imagenet/2012"))
SMALL_ROOT = Path(os.environ.get("TP6_SMALL", "/mnt/hdd/homes/jromero/tp6/imagenet10_256"))

WNIDS = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
         "n03394916", "n03417042", "n03425413", "n03445777", "n03888605"]
CLASS_NAMES = ["tanche", "springer anglais", "lecteur de cassettes", "tronçonneuse",
               "église", "cor d'harmonie", "camion-poubelle", "pompe à essence",
               "balle de golf", "parachute"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ImageNet10(Dataset):
    def __init__(self, root, split, transform):
        self.samples = []
        for label, wnid in enumerate(WNIDS):
            folder = Path(root) / split / wnid
            files = sorted(p for p in folder.iterdir()
                           if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
            self.samples += [(p, label) for p in files]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), label


def get_transform(augment: bool):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_dataset(source="small", split="train", augment=None):
    root = RAW_ROOT if source == "raw" else SMALL_ROOT
    if augment is None:
        augment = split == "train"
    return ImageNet10(root, split, get_transform(augment))


def make_loader(dataset, batch_size=64, num_workers=3, pin_memory=True,
                train=True, seed=0, sampler=None):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )


def build_model(arch="resnet50", pretrained=False, num_classes=10):
    if arch == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
    elif arch == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
    else:
        raise ValueError(f"architecture inconnue : {arch}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_fp32_mode(tf32: bool) -> None:
    mode = "tf32" if tf32 else "ieee"
    torch.backends.cuda.matmul.fp32_precision = mode
    torch.backends.cudnn.conv.fp32_precision = mode


def make_train_step(model, opt, criterion, amp_dtype=None):
    def step(x, y):
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss
    return step


def synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def benchmark(step, device, n_warmup=10, n_iter=30):
    for _ in range(n_warmup):
        step()
    synchronize(device)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        step()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    return times


def device_name(device) -> str:
    if torch.device(device).type == "cuda":
        return torch.cuda.get_device_name()
    return f"CPU ({torch.get_num_threads()} threads)"


def append_result(row: dict, path: str) -> None:
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def median(values):
    return statistics.median(values)
