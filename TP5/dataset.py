import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = datasets.Flickr30k(
    root="/mnt/ssd/flickr30k/flickr30k-images",
    ann_file="/mnt/ssd/flickr30k/results_20130124.token",
    transform=transform,
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


if __name__ == "__main__":
    print(f"Dataset total : {len(dataset)}")
    print(f"Train : {len(train_dataset)} | Test : {len(test_dataset)}")
    images, captions = next(iter(train_loader))
    print(f"Batch images : {images.shape}")
    print(f"Exemple caption[0][0] : {captions[0][0]}")
