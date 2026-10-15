# CSC 8607 – TP4 : Réseaux récurrents et génération de séquences ABC

**Auteur :** *(à compléter)*  
**Données :** `/mnt/ssd/irishman` (`train.json`, `validation.json`)

---

## Exercice 1 — Exploration et prétraitement

### Statistiques du dataset (à remplir après exécution sur le cluster)

```bash
python -u train_music.py --mode stats
```

| Grandeur | Valeur |
|----------|--------|
| # partitions **train** | *(ex. ~200 000 — à reporter depuis le log)* |
| # partitions **validation** | *(à reporter)* |
| **vocab_size** (caractères uniques) | *(typiquement ~90–100 ; l'énoncé utilise 95)* |
| Longueur **moyenne** (train) | *(≈ 250 caractères)* |
| Longueur **max** (train) | *(à reporter)* |

### Première partition et métadonnées ABC

Coller ici `train_data[0]["abc notation"]` puis l'écouter sur [ABC Player](https://abcjs.net/abcjs-editor.html).

Rôle des **en-têtes** (lignes avant les notes) :

| Champ | Rôle |
|-------|------|
| `X:` | numéro / identifiant de la partition |
| `T:` | titre |
| `M:` | métrique (ex. `6/8`, `4/4`) — découpage rythmique des mesures |
| `L:` | durée de note par défaut (ex. `1/8`) |
| `K:` | tonalité / armure (ex. `G`, `Dmaj`) |
| `R:` / autres | éventuellement type de danse, tempo, etc. |

Les notes (`A`–`G`, `a`–`g`), altérations et barres (`|`, `:|`, `||`) viennent **après** ces métadonnées.

### Pourquoi `sorted(...)` sur le vocabulaire ?

`set(...)` en Python a un ordre d'itération **non déterministe** (hash aléatoire selon les runs / versions). `sorted(...)` fixe un ordre lexicographique **reproductible**.

Sans `sorted` : si vous sauvegardez les poids aujourd'hui puis reconstruisez `char_to_index` demain, l'indice du caractère `'A'` peut changer → les embeddings et la couche de sortie ne correspondent plus aux bons symboles → génération **incompréhensible** (mauvais mapping index ↔ caractère).

### Pourquoi `SEQ_LENGTH = 500` plutôt que `max_len` ?

- La moyenne est ~250 ; 500 couvre la grande majorité des partitions utiles (~2× la moyenne) sans trop de padding.
- Aligner sur `max_len` (souvent très grand, outliers) :
  - **temps** : BPTT sur des séquences beaucoup plus longues (coût ∝ longueur) ;
  - **mémoire GPU** : activations stockées sur toute la longueur × batch ;
  - **apprentissage** : énormément de tokens de padding (`' '`) → le modèle apprend surtout à prédire des espaces, gradients dilués sur le contenu musical réel.

---

## Exercice 2 — Dataset auto-régressif et modèles

### Shapes du DataLoader

Avec `SEQ_LENGTH = 500` et `BATCH_SIZE = 128` :

- chaque séquence paddée a longueur 500 ;
- `x = seq[:-1]` et `y = seq[1:]` → longueur **499** ;
- donc **`x.shape = y.shape = (128, 499)`**.

**Pourquoi `torch.long` ?**  
`nn.Embedding` et `CrossEntropyLoss` attendent des **indices entiers** de classe / de token, pas des flottants. `float32` serait invalide (et sémantiquement absurde pour un index discret).

### Calcul théorique des paramètres (`V=95`, `E=256`, `H=512`)

#### Embedding
\[
V \times E = 95 \times 256 = 24\,320
\]

#### Bloc RNN (Elman) — matrices `ih`, `hh` + **2 biais**
\[
\begin{align*}
W_{ih} &: H \times E = 512 \times 256 = 131\,072 \\
W_{hh} &: H \times H = 512 \times 512 = 262\,144 \\
b_{ih} + b_{hh} &: 2H = 1\,024 \\
\textbf{Total RNN} &= 394\,240
\end{align*}
\]

#### Bloc LSTM — **4 portes** (i, f, g, o) ⇒ ×4
\[
\begin{align*}
W_{ih} &: 4HE = 524\,288 \\
W_{hh} &: 4H^2 = 1\,048\,576 \\
b_{ih}+b_{hh} &: 8H = 4\,096 \\
\textbf{Total LSTM} &= 1\,576\,960 = 4 \times 394\,240
\end{align*}
\]

Le LSTM a **exactement 4×** plus de paramètres dans le bloc récurrent car chaque porte a sa propre copie des matrices $W_{ih}$, $W_{hh}$ et des deux biais.

#### Couche linéaire `fc`
\[
H \times V + V = 512 \times 95 + 95 = 48\,735
\]

| Architecture | Embedding | RNN/LSTM | FC | **Total** |
|--------------|-----------|----------|-----|-----------|
| `rnn` | 24 320 | 394 240 | 48 735 | **467 295** |
| `lstm` | 24 320 | 1 576 960 | 48 735 | **1 650 015** |

Vérification : `python -u train_music.py --mode count` (ou `count_parameters`).

### Pourquoi `nn.Embedding` plutôt qu'un scalaire → Linear ?

Un indice $i \in \{0,\ldots,V-1\}$ traité comme un **réel** encoderait une fausse métrique (ex. `'B'=2` « plus proche » de `'A'=1` que de `'Z'`).  
`nn.Embedding` est une **table de vecteurs** (équivalent one-hot × matrice) : chaque caractère a un embedding indépendant, sans ordre artificiel sur les indices.

### `hidden_state` : RNN vs LSTM

- **`nn.RNN`** : retourne un seul tenseur $h_t$ (état caché), shape `(num_layers, batch, H)` (ici 1 couche).
- **`nn.LSTM`** : retourne un **tuple** `(h_t, c_t)` :
  - $h_t$ : état caché (sortie exposée / lue par la couche suivante) ;
  - $c_t$ : **cell state** (mémoire longue durée, mise à jour additive via portes).

Rôle : $c_t$ stocke l'information sur le long terme ; les portes contrôlent écriture/oubli/lecture, ce qui atténue le vanishing gradient par rapport au RNN simple.

---

## Exercice 3 — Entraînement, gradients, RNN vs LSTM

### Commandes (mode rapide séance)

```bash
python -u train_music.py --mode quick
# équivalent aux 3 runs : RNN noclip, RNN clip=1.0, LSTM clip=1.0
# (15 époques, subset_size=10000, hidden=512)
tensorboard --logdir runs_rnn
```

### Pourquoi `logits.reshape(-1, vocab_size)` et `y.reshape(-1)` ?

`CrossEntropyLoss` attend :

- prédictions `(N, C)` — ici $C=$ `vocab_size` ;
- cibles `(N,)` — indices de classe.

Or `logits` est `(B, T, V)` et `y` est `(B, T)`. Le `-1` fusionne batch et temps :  
$N = B \times T = 128 \times 499 = 63\,872$ tokens traités comme exemples indépendants pour la loss.

### Ordre : `backward` → `clip_grad_norm_` → `step`

1. `loss.backward()` **calcule et stocke** les `.grad` ;
2. `clip_grad_norm_` **modifie** ces `.grad` (rescale si $\|g\|_2 > max\_norm$) ;
3. `optimizer.step()` **applique** les gradients (éventuellement déjà clipés).

Si on clipait après `step()`, la mise à jour aurait déjà eu lieu avec des gradients explosifs. Si on clipait avant `backward()`, il n'y aurait pas encore de gradients.

### Analyse expérimentale (TensorBoard)

> Insérer les captures :
>
> `![Loss/Val](figures/tb_loss_val_3runs.png)`  
> `![Acc/Val](figures/tb_acc_val_3runs.png)`  
> `![GradNorm](figures/tb_gradnorm_3runs.png)`

**Run 1 (RNN, no clip) vs Run 2 (RNN, clip=1.0) :**  
Sans clipping, la `GradNorm` peut **exploser** par moments (produit de 500 jacobiennes en BPTT) → spikes de loss, instabilité. Le clipping borne $\|g\|_2$ et corrige principalement le problème d'**exploding gradient** du RNN simple (il ne résout pas à lui seul le *vanishing*).

**Run 2 (RNN+clip) vs Run 3 (LSTM+clip) :**  
Le LSTM converge en général **plus vite** et atteint une meilleure Val Loss / Accuracy.  
Mathématiquement, la cellule :
\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\]
est une **voie additive** contrôlée par la forget gate $f_t$. Si $f_t \approx 1$, le gradient $\partial c_t / \partial c_{t-1}$ reste proche de 1 sur de nombreux pas → les dépendances longues (structure de mesures, motifs répétés dans une jig/reel) sont mieux préservées que dans le RNN Elman $h_t = \tanh(W_h h_{t-1} + \cdots)$ où le facteur $|\lambda(W_h)|^{T}$ s'atténue ou explose.

### Accuracy artificiellement optimiste à cause du padding

Une partition de 250 caractères paddée à 500 contient ~250 espaces. Prédire `' '` après le début du padding est **trivial** → l'accuracy globale est gonflée.

**Correction simple :**  
`nn.CrossEntropyLoss(ignore_index=char_to_index[' '])`  
(ou un index de padding dédié) pour **exclure** les positions de padding de la loss (et idéalement de l'accuracy).

---

## Exercice 4 — Génération, température, Slurm

### Complexité algorithmique

- **Sans réutiliser `hidden_state`** : à l'étape $k$, on repasse la séquence de longueur $\approx L_0 + k$ → coût total  
  $\sum_{k=1}^{N} O(L_0 + k) = O(N^2)$ (quadratique en $N$ caractères générés).
- **Avec propagation de `hidden_state`** : chaque pas coûte $O(1)$ (un seul token) → **$O(N)$** au total.

### Rôle de la température $T$

\[
P(y_t = c) = \frac{\exp(z_{t,c}/T)}{\sum_{c'}\exp(z_{t,c'}/T)}
\]

- **$T \to 0^+$** : la distribution se concentre sur $\arg\max_c z_{t,c}$ → équivalent au **greedy decoding** (déterministe, peu diversifié, risque de boucles).
- **$T \to +\infty$** : $z/T \to 0$ → distribution **uniforme** sur le vocabulaire → génération chaotique, syntaxe ABC souvent invalide.

### Générations à $T \in \{0.2, 0.8, 2.0\}$

Amorce :
```text
X:1
T:Irish Reel
M:4/4
L:1/8
K:D
```

Après le job long, coller ici le contenu de `generated/irish_reel_T*.abc` (aussi dans les logs Slurm) :

### T = 0.2 (attendu : cohérent, répétitif)

```abc
(à coller depuis generated/irish_reel_T0.2.abc)
```

Commentaire typique : syntaxe souvent correcte, motifs qui se répètent, peu de surprise.

### T = 0.8 (attendu : bon compromis)

```abc
(à coller depuis generated/irish_reel_T0.8.abc)
```

Commentaire typique : barres `|` et notes plus variées, encore globalement « musicale ».

### T = 2.0 (attendu : divers / chaotique)

```abc
(à coller depuis generated/irish_reel_T2.0.abc)
```

Commentaire typique : caractères rares, en-têtes cassés, mesures irrégulières — richesse au détriment de la validité.

### Job Slurm

```bash
cd TP4
sbatch run_music.sh
squeue -u $USER
tail -f logs/music-lstm-XXXX.out
```

Hyperparamètres du script : LSTM, `hidden_size=1024`, `subset_size=50000`, `max_grad_norm=1.0`, `num_epochs=30`, puis génération automatique des 3 températures.

### Bonus — augmentation par transposition

En ABC, on peut :

1. parser l'en-tête `K:` et les notes (`A`–`G` / `a`–`g`, `#`, `b`) ;
2. appliquer un décalage de $k$ demi-tons (table chromatique) ;
3. réécrire `K:` (ex. `G` → `A`) et les notes transposées.

Chaque partition engendre plusieurs variantes **structurellement identiques** mais avec d'autres caractères → élargit le support empirique, réduit le sur-apprentissage sur des tonalités / doigtés spécifiques du corpus irlandais.

---

## Fichiers rendus

```
TP4/
├── train_music.py      # chargement, vocab, dataset, RNN/LSTM, train, generate
├── run_music.sh        # job Slurm LSTM long + génération
├── rapport.md
├── vocab.json          # généré à l'exécution
├── best_lstm.pth       # meilleur checkpoint (après train)
├── generated/          # irish_reel_T0.2/0.8/2.0.abc
├── runs_rnn/           # TensorBoard
├── figures/            # captures
└── logs/               # sorties sbatch
```

### Commandes rappel

```bash
# Stats dataset + vocab
python -u train_music.py --mode stats

# Comptage paramètres (sans données cluster si V=95 forcé en count)
python -u train_music.py --mode count

# 3 runs séance
python -u train_music.py --mode quick

# Job long
sbatch run_music.sh

# Génération seule depuis un checkpoint
python -u train_music.py --mode generate --checkpoint best_lstm.pth
```

### Checklist Git

- [x] `train_music.py`
- [x] `run_music.sh`
- [x] `rapport.md` (réponses théoriques)
- [ ] Captures TensorBoard des 3 runs
- [ ] Stats exactes train/val/vocab depuis le cluster
- [ ] 3 fichiers ABC générés + commentaires d'écoute
- [ ] `git add TP4 && git commit && git push`
