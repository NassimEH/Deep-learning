import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from dataset import test_loader, train_loader
from model import ImageCaptioningModel
from text_utils import idx2word, pretrained_embeddings, tokenize, untokenize, word2idx


def parse_args():
    parser = argparse.ArgumentParser(description="TP5 - Image Captioning")
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--log_dir", type=str, default="runs/captioning")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    embeddings = pretrained_embeddings.to(device)

    model = ImageCaptioningModel(
        cnn_model=resnet,
        embedding_model=embeddings,
        lstm_hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    writer = SummaryWriter(args.log_dir)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for i, (images, captions) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)

            caps = torch.LongTensor(
                [tokenize(x, word2idx) for x in captions[0]]
            ).to(device)

            outputs = model(images, caps[:, :-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                caps[:, 1:].reshape(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                writer.add_scalar(
                    "Loss/Train_Step",
                    total_loss / 10,
                    epoch * len(train_loader) + i,
                )
                print(
                    f"Epoch {epoch + 1}/{args.num_epochs} "
                    f"step {i + 1}/{len(train_loader)} "
                    f"loss={total_loss / 10:.4f}",
                    flush=True,
                )
                total_loss = 0.0

        model.eval()
        val_loss = 0.0
        pred_text = ""
        true_text = ""

        with torch.no_grad():
            for images, captions in test_loader:
                images = images.to(device)
                caps = torch.LongTensor(
                    [tokenize(x, word2idx) for x in captions[0]]
                ).to(device)

                outputs = model(images, caps[:, :-1])
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    caps[:, 1:].reshape(-1),
                )
                val_loss += loss.item()

                predictions = outputs[-1].argmax(dim=-1)
                pred_text = " ".join(untokenize(predictions.cpu().numpy(), idx2word))
                true_text = " ".join(untokenize(caps[-1][1:].cpu().numpy(), idx2word))

        avg_val = val_loss / max(len(test_loader), 1)
        print(f"Epoch {epoch + 1}")
        print(f"Prediction : {pred_text}")
        print(f"Target     : {true_text}")
        print(f"Val Loss   : {avg_val:.4f} | LR : {scheduler.get_last_lr()[0]}")

        writer.add_scalar("Loss/Validation", avg_val, epoch)
        scheduler.step()
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        writer.flush()

    writer.close()
    torch.save(model.state_dict(), "best_captioning.pth")
    print("Modèle sauvegardé dans best_captioning.pth")


if __name__ == "__main__":
    main()
