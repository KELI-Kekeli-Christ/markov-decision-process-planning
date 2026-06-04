# TP4 — Algorithmes itératifs pour les MDP (Processus de Décision Markovien)

> **Objectif** : Implémenter deux algorithmes pour résoudre un MDP à horizon infini sur un plateau de jeu avec obstacles.
> **Algorithmes** : (Q1) Value Iteration | (Q2) Policy Iteration

---

## 1. Cadre théorique — Qu'est-ce qu'un MDP ?

Un MDP est défini par le tuple **(S, A, P, R, γ)** :

| Symbole | Signification |
|---|---|
| **S** | Ensemble fini d'états |
| **A** | Ensemble fini d'actions |
| **P(s'\|s, a)** | Probabilité de transiter vers s' depuis s en faisant l'action a |
| **R(s)** | Récompense reçue en étant dans l'état s |
| **γ ∈ [0,1]** | Facteur d'escompte |

**Propriété de Markov** : la transition ne dépend que de l'état courant et de l'action, pas du passé :

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

---

## 2. Modélisation du plateau de jeu

### 2.1 Les états

Le plateau est une grille **n × m** (n lignes, m colonnes).
Chaque case = un état `s = (x, y)` avec `x ∈ {1,...,n}` et `y ∈ {1,...,m}`.

**Configuration par défaut du TP** (grille 3×4) :

```
y=1   y=2   y=3   y=4
+-----+-----+-----+-----+   x=3 (ligne du haut)
|     |     |     | +1  |
+-----+-----+-----+-----+   x=2 (ligne du milieu)
|     | MUR |     | -1  |
+-----+-----+-----+-----+   x=1 (ligne du bas)
|     |     |     |     |
+-----+-----+-----+-----+
```

- **État (2,2)** : obstacle (mur, inaccessible)
- **État (2,4)** : récompense R = **-1**
- **État (3,4)** : récompense R = **+1**
- Tous les autres états : R = **0**
- Position de départ : `(x0, y0) = (1, 1)`

### 2.2 Les actions

4 actions possibles : **Nord (N), Sud (S), Est (E), Ouest (O)**

### 2.3 Les probabilités de transition P(s'|s, a)

Le robot n'est pas parfait : **80% de chance d'aller dans la direction demandée**, et **20% réparti uniformément entre les directions perpendiculaires** (10% chacune).

**Exemple pour l'action Nord :**

```
          P = 0.8 (Nord)
               ↑
P = 0.1 (O) ←[s]→ P = 0.1 (E)
```

**Tableau complet des probabilités de transition :**

| Action demandée | Direction effective | Probabilité |
|---|---|---|
| Nord | Nord | 0.8 |
| Nord | Est | 0.1 |
| Nord | Ouest | 0.1 |
| Sud | Sud | 0.8 |
| Sud | Est | 0.1 |
| Sud | Ouest | 0.1 |
| Est | Est | 0.8 |
| Est | Nord | 0.1 |
| Est | Sud | 0.1 |
| Ouest | Ouest | 0.8 |
| Ouest | Nord | 0.1 |
| Ouest | Sud | 0.1 |

**Règles de bord (mur et obstacle)** :
- Si le déplacement amène hors de la grille ou dans une case obstacle → **le robot reste dans son état actuel** `s' = s`.
- Conséquence : la probabilité de rester sur place augmente selon le nombre de mouvements bloqués.

**Formule générale de transition :**

Pour chaque direction `d` avec probabilité `p_d` associée à l'action `a` :
```
Si next(s, d) est valide et non-obstacle :
    P(next(s,d) | s, a) += p_d
Sinon :
    P(s | s, a) += p_d   ← le robot reste sur place
```

---

## 3. Politique et fonction de valeur

### 3.1 Politique

Une **politique** π est une fonction qui associe à chaque état une action :

```
π : S → A
```

### 3.2 Fonction de valeur d'une politique

La **fonction de valeur** `V^π(s)` représente l'espérance de la somme des récompenses escomptées en partant de l'état `s` et en suivant la politique π :

```
V^π(s) = E[ Σ_{t=0}^{∞} γ^t · R(s_t) | s_0 = s, a_t = π(s_t), s_{t+1}|s_t,a_t ~ P ]
```

### 3.3 Équation de Bellman (pour une politique fixée)

```
V^π(s) = R(s) + γ · Σ_{s' ∈ S} P(s' | s, π(s)) · V^π(s')
```

**Forme matricielle** (utile pour Policy Iteration) :

Soit :
- `v^π ∈ R^|S|` : vecteur des valeurs de tous les états
- `r ∈ R^|S|` : vecteur des récompenses
- `P^π ∈ R^{|S|×|S|}` : matrice de transition sous politique π, avec `P^π_{ij} = P(s_{t+1}=i | s_t=j, a_t=π(s_t))`

Alors l'équation de Bellman devient :

```
v^π = r + γ · P^π · v^π
⟹ (I - γ P^π) · v^π = r
⟹ v^π = (I - γ P^π)^{-1} · r
```

> Résoudre ce système linéaire donne exactement V^π.

---

## 4. Politique et valeur optimales

### 4.1 Politique optimale

```
π* = argmax_π  V^π(s),  pour tout s ∈ S
```

### 4.2 Équation d'optimalité de Bellman

La **valeur optimale** V* satisfait :

```
V*(s) = R(s) + γ · max_{a ∈ A}  Σ_{s' ∈ S}  P(s' | s, a) · V*(s')
```

Une fois V* connue, la politique optimale est :

```
π*(s) = argmax_{a ∈ A}  Σ_{s' ∈ S}  P(s' | s, a) · V*(s')
```

---

## 5. Question 1 — Value Iteration (Itération sur la valeur)

### 5.1 Idée

Appliquer l'opérateur de Bellman de façon répétée jusqu'à convergence. On part d'une estimation arbitraire de V et on la raffine à chaque itération.

### 5.2 Algorithme

```
ALGORITHME : Value Iteration
---------------------------------
ENTRÉES : grille n×m, R, P, γ, ε (seuil de convergence)

1. Initialisation :
   V(s) ← 0   pour tout s ∈ S

2. Répéter :
   Pour tout s ∈ S :
       V_new(s) ← R(s) + γ · max_{a ∈ A} [ Σ_{s'} P(s'|s,a) · V(s') ]

   δ ← max_{s ∈ S} |V_new(s) - V(s)|
   V ← V_new

   Tant que δ > ε

3. Extraire la politique optimale :
   Pour tout s ∈ S :
       π*(s) ← argmax_{a ∈ A} [ Σ_{s'} P(s'|s,a) · V(s') ]

SORTIE : V* (approximée), π*
```

### 5.3 Formule de mise à jour (le coeur de l'algorithme)

```
V_{k+1}(s) = R(s) + γ · max_{a ∈ A} Σ_{s' ∈ S} P(s'|s,a) · V_k(s')
```

On note `Q(s, a)` la **valeur action-état** pour factoriser le calcul :

```
Q(s, a) = Σ_{s' ∈ S} P(s'|s,a) · V_k(s')

V_{k+1}(s) = R(s) + γ · max_{a ∈ A} Q(s, a)
```

### 5.4 Critère de convergence

On arrête quand la variation maximale entre deux itérations est inférieure à un seuil ε :

```
δ = max_{s ∈ S} | V_{k+1}(s) - V_k(s) |  <  ε
```

Avec `ε = (γ · ε_policy) / (2 · (1-γ))` si on veut une garantie sur la politique.
En pratique, utiliser `ε = 1e-6` suffit.

### 5.5 Illustration numérique (du cours, γ = 0.9)

```
Itération 0 (init) :    Itération 1 :         Itération 5 :         Itération 10 :
+---+---+---+----+      +----+---+------+----+ +------+-------+-------+-------+
| 0 | 0 | 0 | +1 |      |  0 | 0 | 0.72 |1.81| |0.809 | 1.598 | 2.475 | 3.745 |
+---+---+---+----+      +----+---+------+----+ +------+-------+-------+-------+
| 0 |MUR| 0 | -1 |  →   |  0 |MUR|  0   |-99.91| |0.268 |  MUR  | 0.302 |-99.59 |
+---+---+---+----+      +----+---+------+----+ +------+-------+-------+-------+
| 0 | 0 | 0 |  0 |      |  0 | 0 |  0   |  0 | |  0   | 0.034 | 0.122 | 0.004 |
+---+---+---+----+      +----+---+------+----+ +------+-------+-------+-------+
```

### 5.6 Ce qu'il faut afficher / tester

- La grille des valeurs V à chaque itération (ou toutes les k itérations)
- Le nombre d'itérations jusqu'à convergence
- La politique optimale finale (flèches sur la grille)
- Tester avec différentes valeurs de γ (ex : 0.5, 0.9, 0.99)

---

## 6. Question 2 — Policy Iteration (Itération sur la politique)

### 6.1 Idée

Au lieu d'itérer sur la valeur, on itère sur la politique elle-même. Chaque itération comporte deux phases : **évaluation exacte** de la politique courante, puis **amélioration** de la politique.

### 6.2 Algorithme

```
ALGORITHME : Policy Iteration
---------------------------------
ENTRÉES : grille n×m, R, P, γ

1. Initialisation :
   π(s) ← action arbitraire   pour tout s ∈ S
   (ex : toujours Nord, ou aléatoire)

2. Répéter :

   --- ÉTAPE A : Évaluation de la politique (Policy Evaluation) ---
   Résoudre le système linéaire :
       (I - γ P^π) · v^π = r
   Soit :   v^π = (I - γ P^π)^{-1} · r

   --- ÉTAPE B : Amélioration de la politique (Policy Improvement) ---
   Pour tout s ∈ S :
       π_new(s) ← argmax_{a ∈ A} [ Σ_{s'} P(s'|s,a) · v^π(s') ]

   --- ÉTAPE C : Test de convergence ---
   Si π_new == π :
       STOP  (politique optimale trouvée)
   Sinon :
       π ← π_new

SORTIE : π* (politique optimale), V^π* (valeur optimale)
```

### 6.3 Détail de l'étape A — Construction de P^π

La matrice `P^π` de taille `|S| × |S|` est construite comme suit :

Pour chaque état `s` (indexé par `j`) et chaque état `s'` (indexé par `i`) :

```
P^π_{i,j} = P(s' = i | s = j, a = π(j))
```

C'est-à-dire : **on fixe l'action à π(s)** et on lit les probabilités de transition.

### 6.4 Détail de l'étape A — Résolution du système linéaire

Le système à résoudre est :

```
(I - γ P^π) · v = r
```

Où :
- `I` est la matrice identité de taille `|S| × |S|`
- `γ` est le facteur d'escompte
- `P^π` est la matrice de transition sous la politique actuelle
- `r` est le vecteur des récompenses

On résout ce système linéaire avec `numpy.linalg.solve(A, b)` :

```
A = I - γ · P^π
b = r
v^π = solve(A, b)
```

> Attention : il faut indexer correctement les états (aplatir la grille 2D en vecteur 1D).

### 6.5 Détail de l'étape B — Amélioration de la politique

Une fois `v^π` calculé, on améliore la politique :

```
π_new(s) = argmax_{a ∈ A} Σ_{s' ∈ S} P(s'|s,a) · v^π(s')
```

On recalcule pour chaque état et chaque action la somme pondérée des valeurs des états suivants.

### 6.6 Propriétés importantes

- **Convergence garantie** : Policy Iteration converge toujours en un nombre fini d'itérations (au plus `|A|^|S|` itérations, mais en pratique très peu).
- **Convergence plus rapide** que Value Iteration en nombre d'itérations.
- **Plus coûteux par itération** car il faut résoudre un système linéaire à chaque étape.

---

## 7. Construction détaillée des probabilités de transition

C'est la partie la plus technique à implémenter. Voici comment procéder.

### 7.1 Indexation des états

Avec une grille n×m, on crée un index plat. Exemple pour 3×4 :

```
État (x, y) → index = (x-1) * m + (y-1)
Exemple : état (2,3) → index = (2-1)*4 + (3-1) = 6
```

### 7.2 Fonction de transition élémentaire

Pour calculer `next_state(s, direction)` :

```
Directions : N=(+1,0), S=(-1,0), E=(0,+1), O=(0,-1)
   (en coordonnées (x,y))

next_state(s=(x,y), direction=(dx,dy)):
    x' = x + dx
    y' = y + dy
    Si (x', y') hors grille OU (x', y') est obstacle :
        retourner s   (reste sur place)
    Sinon :
        retourner (x', y')
```

### 7.3 Construction du tenseur P

```
P[s', s, a] = probabilité d'aller en s' depuis s avec action a
```

Pour chaque état `s`, chaque action `a`, et chaque direction `d` de `a` avec sa probabilité `p_d` :

```
s_suiv = next_state(s, d)
P[s_suiv, s, a] += p_d
```

### 7.4 Correspondance actions/directions

```
Action Nord  : {Nord: 0.8, Est: 0.1, Ouest: 0.1}
Action Sud   : {Sud: 0.8, Est: 0.1, Ouest: 0.1}
Action Est   : {Est: 0.8, Nord: 0.1, Sud: 0.1}
Action Ouest : {Ouest: 0.8, Nord: 0.1, Sud: 0.1}
```

---

## 8. Structure suggérée du programme

### Structures de données recommandées

```
- grille          : tableau 2D n×m (True/False pour les obstacles)
- R               : tableau 2D n×m (récompenses)
- P               : tableau 3D [|S|, |S|, |A|] (probabilités de transition)
- V               : tableau 1D de taille |S| (valeurs)
- policy          : tableau 1D de taille |S| (action pour chaque état)
```

### Paramètres à définir

```
n = 3           # nombre de lignes
m = 4           # nombre de colonnes
gamma = 0.9     # facteur d'escompte
epsilon = 1e-6  # seuil de convergence (pour Value Iteration)

# Récompenses
R[2,4] = -1     # état (2,4)
R[3,4] = +1     # état (3,4)

# Obstacle
obstacle = {(2,2)}

# Position de départ
start = (1, 1)
```

---

## 9. Comparaison Value Iteration vs Policy Iteration

| Critère | Value Iteration | Policy Iteration |
|---|---|---|
| Initialisation | V(s) = 0 | π arbitraire |
| Par itération | Mise à jour de V | Évaluation exacte + amélioration |
| Coût par itération | O(\|S\|² × \|A\|) | O(\|S\|³) pour la résolution + O(\|S\|² × \|A\|) |
| Nb d'itérations | Plus grand | Très faible (souvent < 20) |
| Convergence | Asymptotique (ε) | Exacte (en nb fini de politiques) |
| Résultat final | V* approché, puis π* | π* exact |

---

## 10. Ce qu'on attend dans le rapport

1. **Description du MDP** : états, actions, transitions, récompenses
2. **Implémentation Value Iteration** :
   - Code commenté
   - Courbe de convergence (δ en fonction des itérations)
   - Affichage de la grille des valeurs et de la politique optimale
   - Effet du facteur γ sur la politique
3. **Implémentation Policy Iteration** :
   - Code commenté
   - Nombre d'itérations jusqu'à convergence
   - Comparaison avec Value Iteration (temps, nb d'itérations)
4. **Analyse** : discussion des résultats, influence de γ, comportement du robot

---

## 11. Points de vigilance

- **L'état obstacle (2,2)** : ne jamais lui assigner de valeur ou de politique, le traiter comme un mur.
- **Normalisation de P** : vérifier que `Σ_{s'} P(s'|s,a) = 1` pour tout `(s,a)`.
- **Indexation** : être cohérent dans la conversion état `(x,y)` ↔ index entier.
- **Matrice `(I - γP^π)` singulière** : ne peut pas l'être si `γ < 1`, mais vérifier quand même.
- **Affichage** : orienter la grille correctement (ligne 1 en bas, ligne n en haut).
- **États terminaux** : dans ce TP, les états (2,4) et (3,4) ne sont PAS nécessairement terminaux à moins que le sujet le précise — ils ont juste des récompenses non nulles.