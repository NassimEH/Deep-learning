# CSC 8607 – TP3 : Réseaux convolutifs (CI3)

**Auteur :** *(à compléter)*  
**Environnement :** PyTorch, ImageNet ILSVRC 2012 (`/mnt/ssd/imagenet/2012`), cluster GPU Slurm

---

## Exercice 1 — Préparation d'ImageNet et augmentation de données

### Pipelines implémentés

- **Train :** `RandomResizedCrop(227)` → `RandomHorizontalFlip()` → `ToTensor()` → `Normalize(ImageNet mean/std)`
- **Val :** `Resize(256)` → `CenterCrop(227)` → `ToTensor()` → `Normalize(ImageNet mean/std)`
- DataLoaders : `shuffle=True` (train), `shuffle=False` (val), `num_workers=8`, `pin_memory=True`

### Q1. Que se passe-t-il si l'on oublie `transforms.Normalize` ?

**Mathématiquement / numériquement :**

Les pixels JPEG convertis par `ToTensor()` sont dans $[0, 1]$. Sans normalisation, les activations des premières couches (surtout les convolutions avec grands kernels) reçoivent des entrées dont l'échelle et le biais ne correspondent pas à ceux pour lesquels l'initialisation (ex. Kaiming/Xavier) et le learning rate ont été conçus.

Lors de la descente de gradient :

1. Les **gradients** peuvent être mal calibrés (trop grands ou trop petits) car la magnitude des activations dépend de l'échelle des entrées.
2. Les **différentes voies RGB** n'ont pas la même dynamique (canaux plus/moins lumineux) : l'optimiseur doit compenser ces biais au lieu d'apprendre des features.
3. En pratique : convergence plus lente, instabilité éventuelle, performances dégradées.

**Pourquoi mean/std uniquement sur le train ?**

La validation et le test doivent simuler des données *futures* non vues. Utiliser leurs statistiques pour normaliser (ou pour choisir mean/std) constituerait une **fuite de données** (*data leakage*) : on ferait entrer de l'information du jeu d'évaluation dans le prétraitement. On fixe donc mean/std sur le train (ici les statistiques historiques ImageNet) et on les réapplique telles quelles en val/test.

### Q2. Pourquoi RandomResizedCrop + Flip en train, mais Resize + CenterCrop (sans flip) en val ?

- **Train :** l'augmentation aléatoire augmente la diversité vue par le réseau (échelles, cadrages, symétries horizontales). Cela agit comme une régularisation et améliore la généralisation.
- **Val :** on veut une mesure **reproductible et comparable** d'époque en époque. Un crop/flip aléatoire ferait varier la métrique sans lien avec l'apprentissage. Le protocole standard (resize 256 + center crop 227) est déterministe.

### Q3. Exemple où `RandomHorizontalFlip` serait néfaste

Toute tâche où le sens gauche/droite **change la classe** :

- lecture de **chiffres / lettres / panneaux** (un « 6 » miroir n'est plus un « 6 » ; un « b » devient un « d ») ;
- **signalisation routière** asymétrique ;
- classification de **radiographies** ou d'écritures manuscrites où l'orientation porte du sens ;
- détection de **latéralité** (main gauche vs main droite).

Dans ces cas, le flip crée des exemples dont l'étiquette d'origine est **incorrecte**.

---

## Exercice 2 — MLP vs AlexNet

### Étape 1 — MLP : nombre de paramètres

Entrée aplatie : $3 \times 227 \times 227 = 154\,587$.

| Couche | Formule (avec biais) | # paramètres |
|--------|----------------------|--------------|
| Linear1 | $154587 \times 256 + 256$ | **39 574 528** |
| Linear2 | $256 \times 256 + 256$ | 65 792 |
| Linear3 | $256 \times 1000 + 1000$ | 257 000 |
| **Total avec biais** | | **39 897 320** |
| **Total sans biais** | $154587\cdot256 + 256\cdot256 + 256\cdot1000$ | **39 895 808** |

La **première couche dense** concentre plus de 99 % des paramètres ($\approx 39{,}57$ M / $39{,}90$ M $\approx 99{,}2\%$).

Vérification PyTorch : `count_parameters(MLP())` → `39,897,320`.

### Défaut structurel majeur du MLP sur image aplatie

Le flatten **détruit la structure spatiale** (voisinage des pixels). Le MLP :

- n'a **aucune invariance** (translation, petite déformation) ;
- doit réapprendre séparément les mêmes motifs à chaque position ;
- n'exploite pas le **partage de poids** local des convolutions.

### Étape 2 — Tableau AlexNet (entrée $3\times227\times227$)

Formule spatiale (stride $S$, padding $P$, kernel $K$) :

\[
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - K}{S} \right\rfloor + 1
\]

Paramètres conv : $C_{\text{out}} \times (C_{\text{in}} \cdot K \cdot K + 1)$ (biais inclus).  
Mémoire sortie (1 image, float32) : $C \cdot H \cdot W \cdot 4 / 1024$ KB.

| Couche | Taille entrée (C, H×W) | Hyperparamètres | Taille sortie (C, H×W) | # Paramètres (avec biais) | Mémoire sortie (KB) |
|--------|------------------------|-----------------|------------------------|---------------------------|---------------------|
| Conv1 + ReLU | 3, 227×227 | 64 filtres, 11×11, S=4, P=2 | 64, 56×56 | 23 296 | 784.00 |
| MaxPool1 | 64, 56×56 | 3×3, S=2, P=0 | 64, 27×27 | 0 | 182.25 |
| Conv2 + ReLU | 64, 27×27 | 192 filtres, 5×5, S=1, P=2 | 192, 27×27 | 307 392 | 546.75 |
| MaxPool2 | 192, 27×27 | 3×3, S=2, P=0 | 192, 13×13 | 0 | 126.75 |
| Conv3 + ReLU | 192, 13×13 | 384 filtres, 3×3, S=1, P=1 | 384, 13×13 | 663 936 | 253.50 |
| Conv4 + ReLU | 384, 13×13 | 256 filtres, 3×3, S=1, P=1 | 256, 13×13 | 884 992 | 169.00 |
| Conv5 + ReLU | 256, 13×13 | 256 filtres, 3×3, S=1, P=1 | 256, 13×13 | 590 080 | 169.00 |
| MaxPool3 | 256, 13×13 | 3×3, S=2, P=0 | 256, 6×6 | 0 | 36.00 |
| Flatten + Dropout | 256, 6×6 | p=0.5 | 9216 (1D) | 0 | 36.00 |
| Linear1 + ReLU + Dropout | 9216 | 4096 neurones, p=0.5 | 4096 | 37 752 832 | 16.00 |
| Linear2 + ReLU | 4096 | 4096 neurones | 4096 | 16 781 312 | 16.00 |
| Linear3 (sortie) | 4096 | 1000 classes | 1000 | 4 097 000 | 3.91 |

**Total paramètres AlexNet :** $23\,296 + 307\,392 + 663\,936 + 884\,992 + 590\,080 + 37\,752\,832 + 16\,781\,312 + 4\,097\,000 = \mathbf{61\,100\,840}$

**Total mémoire activations ≈ 2 339 KB ≈ 2 340 KB** (cohérent avec l'énoncé).

Vérification PyTorch : `count_parameters(AlexNet())` → `61,100,840`.

### Où se concentrent paramètres vs mémoire d'activations ?

- **Paramètres (poids) :** l'essentiel est dans le **classifieur dense** (`Linear1` ≈ 37,8 M, puis `Linear2` ≈ 16,8 M). Les convolutions ne totalisent qu'environ 2,5 M de paramètres grâce au partage de poids.
- **Mémoire des activations :** l'essentiel est en **début de réseau**, là où les cartes sont encore grandes spatialement (`Conv1` 784 KB, `Conv2` 547 KB, `MaxPool1` 182 KB…). Les couches denses n'occupent que quelques dizaines de KB en activations.

### Paradoxe mémoire MLP vs AlexNet

Les deux réseaux ont un ordre de grandeur **similaire** en paramètres (~40 M vs ~61 M), mais :

- **MLP :** après flatten, une seule activation d'entrée = $154\,587 \times 4 / 1024 \approx 604$ KB, puis couches cachées 1 KB chacune → **mémoire d'activations très faible** (~610 KB).
- **AlexNet :** on stocke **toutes les cartes de caractéristiques intermédiaires** (nécessaires au backward) → ~**2340 KB**/image.

Le paradoxe : AlexNet utilise plus de VRAM pour les activations malgré un partage de poids efficace, car le **forward convolutif produit beaucoup de feature maps spatiales** à conserver pour la backprop. Le MLP « paie » en **poids de la première couche**, pas en activations intermédiaires.

---

## Exercice 3 — Boucle d'entraînement, Early Stopping, comparaison

### Rôle de `model.eval()` / `model.train()` et Dropout

- `model.train()` : active le mode entraînement → **Dropout actif** (neurons masqués avec proba $p=0.5$).
- `model.eval()` : mode évaluation → **Dropout désactivé** (tous les neurones actifs, scaling déterministe).

Si l'on oublie `model.eval()` pendant l'évaluation : le Dropout reste stochastique → métriques **bruitées, sous-estimées et non reproductibles**, et la loss de validation n'est plus comparable d'une époque à l'autre.

### Valeurs théoriques à l'initialisation (1000 classes équilibrées)

Choix purement aléatoire uniforme sur les classes :

- **Accuracy** $\approx 1/1000 = 0{,}001$ (0,1 %)
- **CrossEntropy** $\approx \ln(1000) \approx 6{,}907$

Les toutes premières mesures (avant apprentissage significatif) doivent être proches de ces ordres de grandeur. Dès que le modèle commence à apprendre, la loss descend sous $\ln(1000)$ et l'accuracy monte au-dessus de 0,1 %.

### Comparaison MLP vs AlexNet (subset 20 000, 10 époques)

**Commandes :**

```bash
python -u train_imagenet.py --model mlp --subset_size 20000 --num_epochs 10
python -u train_imagenet.py --model alexnet --subset_size 20000 --num_epochs 10
tensorboard --logdir runs
```

**Attentes / observations typiques :**

| | MLP | AlexNet |
|---|-----|---------|
| Apprentissage | Lent / limité | **Bien meilleur** (features spatiales) |
| Overfitting | Fort et rapide (trop de params sur peu de data, pas d'inductive bias) | Présent mais atténué par Dropout + convolutions |
| Val accuracy | Reste faible | Monte nettement plus haut |

**Lequel apprend le mieux ?** AlexNet.  
**Lequel overfit le plus vite ?** Le MLP : ~40 M de paramètres presque tous dans la 1ʳᵉ couche, aucune structure spatiale, et seulement 20 k images → mémorisation rapide du train, val qui stagne/diminue.

> **Capture TensorBoard :** placer la capture dans `figures/tb_subset_mlp_vs_alexnet.png` après la séance et l'insérer ici :
>
> `![TensorBoard subset](figures/tb_subset_mlp_vs_alexnet.png)`

### Variante plus profonde (`AlexNetDeep`)

Trois convolutions $256\times3\times3$ (padding 1) + ReLU ajoutées **juste après Conv5**.

```bash
python -u train_imagenet.py --model alexnet_deep --subset_size 20000 --num_epochs 10
```

**Observation attendue :** démarrage plus difficile (loss qui descend moins vite, accuracy plus basse au début) par rapport à AlexNet standard.

Ce n'est **pas de l'overfitting** (qui se manifesterait par train ≪ val en loss, train accuracy ≫ val). C'est du **sous-apprentissage (underfitting)** / difficulté d'optimisation : le réseau plus profond souffre d'une **atténuation du gradient** (*vanishing gradient*) à travers davantage de non-linéarités ReLU empilées sans résidus (pas de skip connections à la ResNet). Les premières couches reçoivent des gradients trop faibles → apprentissage lent.

---

## Exercice 4 — Entraînement long via sbatch

### Lancement

```bash
cd TP3
mkdir -p logs
sbatch run_imagenet.sh
squeue -u $USER
# suivi :
tail -f logs/alexnet-full-XXXX.out
```

Le script lance `AlexNet` avec `subset_size=None` et `num_epochs=50`.

### Profilage temporel (à compléter après le job)

D'après les logs (`Time: Xs` affiché chaque époque, ou moyenne en fin de run) :

| Métrique | Valeur (à remplir depuis les logs) |
|----------|-------------------------------------|
| Durée d'**une** époque complète (~1,28 M images) | **___ min ___ s** |
| Estimation pour **100** époques | $100 \times T_{\text{époque}} / 3600$ = **___ h** |

Exemple de calcul (remplacer $T$) : si $T = 25$ min/époque → $100 \times 25 / 60 \approx 41{,}7$ h.

> Insérer ici le tableau des valeurs par époque (Loss/Acc train & val) et/ou :
>
> `![TensorBoard full](figures/tb_full_alexnet.png)`

### Pourquoi moins de sur-apprentissage sur ImageNet complet ?

Sur 20 k images, le modèle peut **mémoriser** le sous-ensemble. Sur **1,28 M d'images + Data Augmentation** :

1. chaque époque présente des crops/flips **différents** → quasi-impossible de mémoriser ;
2. la diversité et le volume de données forcent l'apprentissage de **features générales** ;
3. l'écart Train/Val loss reste donc **beaucoup plus faible**.

### Pourquoi Loss/Val < Loss/Train en début d'entraînement ? (2 raisons liées au code)

1. **Dropout** : actif uniquement en `model.train()`. Pendant le train, ~50 % des neurones du classifieur sont masqués → prédiction plus difficile → loss train **pénalisée**. En `model.eval()`, Dropout désactivé → loss val plus basse.
2. **Data augmentation** : `RandomResizedCrop` + `RandomHorizontalFlip` rendent les exemples d'entraînement **plus difficiles** que le protocole déterministe de validation (`Resize` + `CenterCrop`). La val mesure donc une tâche plus « facile ».

*(Bonus fréquent : la loss train est une moyenne sur l'époque pendant laquelle les poids évoluent, alors que la val est mesurée avec les poids finaux de l'époque.)*

---

## Fichiers rendus

```
TP3/
├── train_imagenet.py   # transforms, MLP, AlexNet, AlexNetDeep, train + Early Stopping
├── run_imagenet.sh     # script Slurm
├── rapport.md          # ce rapport
├── logs/               # sorties sbatch (après job)
├── runs/               # logs TensorBoard
└── figures/            # captures d'écran TensorBoard
```

### Commandes utiles (rappel)

```bash
# Comptage des paramètres (sans charger ImageNet)
python -u train_imagenet.py --count_only

# Mode rapide séance
python -u train_imagenet.py --model mlp --subset_size 20000 --num_epochs 10
python -u train_imagenet.py --model alexnet --subset_size 20000 --num_epochs 10
python -u train_imagenet.py --model alexnet_deep --subset_size 20000 --num_epochs 10

# Job long
sbatch run_imagenet.sh
```

---

## Checklist avant push Git

- [x] `train_imagenet.py` complété
- [x] `run_imagenet.sh` prêt
- [x] `rapport.md` avec réponses théoriques et tableau AlexNet
- [ ] Captures TensorBoard (subset + full) dans `figures/`
- [ ] Logs d'au moins 3–5 époques ImageNet complet
- [ ] Remplir le tableau de profilage temporel (durée/époque)
- [ ] `git add TP3 && git commit && git push`
