"""
CSC 8607 – TP4 : Réseaux récurrents et génération de séquences ABC
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Reproductibilité
torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/mnt/ssd/irishman"
SEQ_LENGTH = 500  # Longueur fixe (~2x la longueur moyenne)


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocabulary(train_data, val_data):
    # 1. Extraction déterministe (triée) des caractères uniques
    all_text = "".join(song["abc notation"] for song in train_data + val_data)
    unique_chars = sorted(list(set(all_text)))
    vocab_size = len(unique_chars)
    print(f"Taille du vocabulaire : {vocab_size} caractères uniques")

    # 2. Mappings caractère <-> index
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    index_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    return unique_chars, vocab_size, char_to_index, index_to_char


def make_vectorize_fn(char_to_index):
    def vectorize_string(s):
        return [char_to_index[char] for char in s]

    return vectorize_string


def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + " " * (max_length - len(sequence))


class MusicDataset(Dataset):
    def __init__(self, raw_data, vectorize_string, seq_length=SEQ_LENGTH, subset_size=None):
        if subset_size is not None:
            raw_data = raw_data[:subset_size]

        self.data = [
            vectorize_string(pad_or_truncate(song["abc notation"], seq_length))
            for song in raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Entrée : tous les caractères sauf le dernier
        x = torch.tensor(seq[:-1], dtype=torch.long)
        # Cible : tous les caractères sauf le premier (décalage de +1)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class MusicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type doit être 'rnn' ou 'lstm'")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        # x : (batch_size, seq_len)
        embeds = self.embedding(x)  # -> (batch_size, seq_len, embedding_dim)
        out, hidden_state = self.rnn(embeds, hidden_state)  # -> (B, T, H)
        logits = self.fc(out)  # -> (batch_size, seq_len, vocab_size)
        return logits, hidden_state


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_model(model, val_loader, loss_fn, vocab_size):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = model(x)

        # Aplatissement (batch_size * seq_len, vocab_size) pour CrossEntropyLoss
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

        total_loss += loss.item() * y.numel()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()

    return total_loss / total_tokens, total_correct / total_tokens


def train_model(
    train_data,
    val_data,
    vocab_size,
    vectorize_string,
    rnn_type="lstm",
    subset_size=10000,
    num_epochs=15,
    batch_size=128,
    lr=5e-3,
    embedding_dim=256,
    hidden_size=512,
    max_grad_norm=None,
    patience=5,
    save_path=None,
):
    train_dataset = MusicDataset(train_data, vectorize_string, subset_size=subset_size)
    val_subset = None if subset_size is None else min(len(val_data), max(1000, subset_size // 5))
    val_dataset = MusicDataset(val_data, vectorize_string, subset_size=val_subset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model = MusicModel(vocab_size, embedding_dim, hidden_size, rnn_type=rnn_type).to(device)
    print(f"Modèle {rnn_type} : {count_parameters(model):,} paramètres")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    clip_tag = f"clip{max_grad_norm}" if max_grad_norm is not None else "noclip"
    sub_tag = "full" if subset_size is None else str(subset_size)
    run_name = f"{rnn_type}_sub{sub_tag}_h{hidden_size}_lr{lr}_{clip_tag}_{time.strftime('%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs_rnn/{run_name}")
    if save_path is None:
        save_path = f"best_{rnn_type}.pth"

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss, total_train_tokens = 0.0, 0
        epoch_grad_norms = []

        for x, y in tqdm(
            train_loader, desc=f"[{run_name}] Epoch {epoch}/{num_epochs}", leave=False
        ):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()

            # Calcul de la norme des gradients (et clipping éventuel AVANT optimizer.step())
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
            else:
                # Calcul de la norme sans modifier les gradients (max_norm=inf)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                )

            epoch_grad_norms.append(grad_norm.item())
            if global_step % 20 == 0:
                writer.add_scalar("GradNorm/step", grad_norm.item(), global_step)

            optimizer.step()

            total_train_loss += loss.item() * y.numel()
            total_train_tokens += y.numel()
            global_step += 1

        avg_train_loss = total_train_loss / total_train_tokens
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, vocab_size)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("GradNorm/Epoch_Avg", avg_grad_norm, epoch)
        writer.flush()

        print(
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Avg GradNorm: {avg_grad_norm:.3f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "embedding_dim": embedding_dim,
                    "hidden_size": hidden_size,
                    "rnn_type": rnn_type,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"  -> Meilleur modèle sauvegardé dans {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping déclenché à l'époque {epoch}.")
                break

    writer.close()

    # Recharge le meilleur checkpoint
    try:
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


@torch.no_grad()
def generate_music(
    model, start_sequence, char_to_index, index_to_char, output_length=300, temperature=1.0
):
    model.eval()
    generated_seq = start_sequence

    # 1. Phase de pré-remplissage (lecture de l'amorce)
    input_indices = [char_to_index[c] for c in start_sequence]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)

    # On récupère les logits et l'état caché à la fin de l'amorce
    logits, hidden_state = model(input_tensor, hidden_state=None)

    # 2. Boucle de génération auto-régressive caractère par caractère
    for _ in range(output_length):
        # Logits du dernier pas de temps divisés par la température
        last_logits = logits[:, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1).squeeze(0)

        # Tirage aléatoire selon la distribution de probabilité
        next_index = torch.multinomial(probs, num_samples=1).item()
        next_char = index_to_char[next_index]
        generated_seq += next_char

        # Le caractère généré devient la seule entrée (1, 1) du pas suivant
        input_tensor = torch.tensor([[next_index]], dtype=torch.long, device=device)
        logits, hidden_state = model(input_tensor, hidden_state=hidden_state)

    return generated_seq


def print_dataset_stats(train_data, val_data):
    print(f"Nombre de chansons (Train) : {len(train_data)}")
    print(f"Nombre de chansons (Val)   : {len(val_data)}")
    print("\n--- Exemple de partition (Train[0]) ---")
    print(train_data[0]["abc notation"])

    lengths = [len(song["abc notation"]) for song in train_data]
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    print(f"Longueur moyenne : {avg_len:.1f} | Longueur max : {max_len}")
    return avg_len, max_len


def print_param_breakdown(vocab_size=95, embedding_dim=256, hidden_size=512):
    emb = vocab_size * embedding_dim
    rnn = embedding_dim * hidden_size + hidden_size * hidden_size + 2 * hidden_size
    lstm = 4 * rnn
    fc = hidden_size * vocab_size + vocab_size
    print("--- Calcul théorique des paramètres ---")
    print(f"Embedding : {emb:,}")
    print(f"RNN block : {rnn:,}")
    print(f"LSTM block: {lstm:,} (= 4 × RNN)")
    print(f"FC        : {fc:,}")
    print(f"Total RNN : {emb + rnn + fc:,}")
    print(f"Total LSTM: {emb + lstm + fc:,}")

    for rnn_type in ("rnn", "lstm"):
        model = MusicModel(vocab_size, embedding_dim, hidden_size, rnn_type=rnn_type)
        print(f"count_parameters({rnn_type}) = {count_parameters(model):,}")


def generate_temperature_examples(model, char_to_index, index_to_char, out_dir="generated"):
    os.makedirs(out_dir, exist_ok=True)
    prompt = "X:1\nT:Irish Reel\nM:4/4\nL:1/8\nK:D\n"
    temperatures = [0.2, 0.8, 2.0]
    results = {}

    for t in temperatures:
        abc = generate_music(
            model,
            prompt,
            char_to_index,
            index_to_char,
            output_length=300,
            temperature=t,
        )
        results[t] = abc
        path = os.path.join(out_dir, f"irish_reel_T{t}.abc")
        with open(path, "w", encoding="utf-8") as f:
            f.write(abc)
        print(f"\n===== Génération T={t} =====\n{abc}\n(sauvegardé dans {path})")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="TP4 – Génération musicale ABC (RNN/LSTM)")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["stats", "count", "quick", "full", "generate"],
        help=(
            "stats: stats dataset | count: paramètres | "
            "quick: 3 runs séance | full: LSTM long | generate: génération seule"
        ),
    )
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["rnn", "lstm"])
    parser.add_argument("--subset_size", type=int, default=50000)
    parser.add_argument("--use_full_data", action="store_true", help="Ignore subset_size (None)")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default="best_lstm.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device : {device}")

    if args.mode == "count":
        print_param_breakdown()
        raise SystemExit(0)

    print("Chargement des données irishman...")
    train_data = load_data(os.path.join(DATA_DIR, "train.json"))
    val_data = load_data(os.path.join(DATA_DIR, "validation.json"))
    avg_len, max_len = print_dataset_stats(train_data, val_data)

    unique_chars, vocab_size, char_to_index, index_to_char = build_vocabulary(
        train_data, val_data
    )
    vectorize_string = make_vectorize_fn(char_to_index)

    # Sauvegarde du vocabulaire pour la génération ultérieure
    vocab_path = "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"chars": unique_chars}, f, ensure_ascii=False, indent=2)
    print(f"Vocabulaire sauvegardé dans {vocab_path}")

    if args.mode == "stats":
        print_param_breakdown(vocab_size=vocab_size)
        raise SystemExit(0)

    max_grad_norm = None if args.no_clip else args.max_grad_norm
    subset_size = None if args.use_full_data else args.subset_size

    if args.mode == "quick":
        # Exercice 3 : 3 runs rapides sur 10 000 partitions
        configs = [
            {"rnn_type": "rnn", "max_grad_norm": None, "save_path": "best_rnn_noclip.pth"},
            {"rnn_type": "rnn", "max_grad_norm": 1.0, "save_path": "best_rnn_clip.pth"},
            {"rnn_type": "lstm", "max_grad_norm": 1.0, "save_path": "best_lstm_clip.pth"},
        ]
        for cfg in configs:
            print(f"\n========== Run : {cfg} ==========")
            train_model(
                train_data,
                val_data,
                vocab_size,
                vectorize_string,
                rnn_type=cfg["rnn_type"],
                subset_size=10000,
                num_epochs=15,
                batch_size=128,
                lr=5e-3,
                embedding_dim=256,
                hidden_size=512,
                max_grad_norm=cfg["max_grad_norm"],
                patience=5,
                save_path=cfg["save_path"],
            )

        # Génération illustrative avec le LSTM clip
        model = MusicModel(vocab_size, 256, 512, rnn_type="lstm").to(device)
        try:
            ckpt = torch.load("best_lstm_clip.pth", map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load("best_lstm_clip.pth", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        generate_temperature_examples(model, char_to_index, index_to_char)

    elif args.mode == "full":
        # Exercice 4 : entraînement long LSTM + génération T in {0.2, 0.8, 2.0}
        model = train_model(
            train_data,
            val_data,
            vocab_size,
            vectorize_string,
            rnn_type="lstm",
            subset_size=subset_size,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            max_grad_norm=max_grad_norm,
            patience=args.patience,
            save_path=args.checkpoint,
        )
        generate_temperature_examples(model, char_to_index, index_to_char)

    elif args.mode == "generate":
        with open(vocab_path, "r", encoding="utf-8") as f:
            unique_chars = json.load(f)["chars"]
        char_to_index = {c: i for i, c in enumerate(unique_chars)}
        index_to_char = {i: c for i, c in enumerate(unique_chars)}
        vocab_size = len(unique_chars)

        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location=device)
        model = MusicModel(
            ckpt.get("vocab_size", vocab_size),
            ckpt.get("embedding_dim", args.embedding_dim),
            ckpt.get("hidden_size", args.hidden_size),
            rnn_type=ckpt.get("rnn_type", "lstm"),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        generate_temperature_examples(model, char_to_index, index_to_char)
