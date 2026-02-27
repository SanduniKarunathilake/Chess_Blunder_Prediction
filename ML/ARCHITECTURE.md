# Chess Blunder Detection — Project Architecture

> **Audience:** Written for beginners, reviewed by seniors.  
> If you're new to ML, every section explains the *why* before the *how*.  
> If you're experienced, jump straight to the section headers.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Folder Structure](#2-folder-structure)
3. [Dataset Reference](#3-dataset-reference)
4. [Data Pipeline (How Raw Data Becomes Features)](#4-data-pipeline)
5. [Feature Engineering — What We Feed the Model](#5-feature-engineering)
6. [Model — Why XGBoost](#6-model--why-xgboost)
7. [Training Pipeline](#7-training-pipeline)
8. [Prediction — Two Ways to Use the Model](#8-prediction--two-input-options)
9. [Notebooks Guide](#9-notebooks-guide)
10. [System Requirements](#10-system-requirements)
11. [Quick Start](#11-quick-start)
12. [Common Questions (FAQ)](#12-faq)

---

## 1. What This Project Does

A **blunder** in chess is a move that causes a large drop in position evaluation — typically **200+ centipawns** (2 pawns of value).

This project trains a Machine Learning classifier that answers one question:

> *Given a chess position and some context, how likely is the next move to be a blunder?*

**Target accuracy: 70–80%** — intentionally realistic.  
A 99% accurate blunder detector would require a full chess engine (Stockfish).  
Our goal is a lightweight ML model that can be served instantly, with no engine needed at inference time.

---

## 2. Folder Structure

```
CW/
├── data/
│   └── raw/
│       ├── Chess games stats.csv   ← game-level aggregated stats (18 638 games)
│       ├── datasetECO.csv          ← ECO opening code reference table
│       ├── positions.csv           ← move-by-move FEN positions (main training data)
│       └── sample_game.pgn         ← PGN used for demo predictions
│
├── models/
│   └── blunder_model.pkl           ← saved XGBoost model (created after training)
│
├── notebooks/                      ← scratch / exploration space
│
├── src/
│   ├── __init__.py
│   ├── features.py        ← FEN string → numeric features
│   ├── preprocessing.py   ← load → clean → label → engineer
│   ├── train.py           ← split → fit → evaluate → save
│   └── predict.py         ← load model → predict blunders per move
│
├── blunder.ipynb    ← MAIN notebook  (full pipeline end-to-end)
├── chess.ipynb      ← exploration notebook (regression + ECO classification)
├── chess2.ipynb     ← extended exploration
├── chess3.ipynb     ← extended exploration
└── ARCHITECTURE.md  ← this file
```

> **Rule of thumb:** `src/` is production code (importable, tested).  
> Notebooks are for exploration and presentation — they call `src/`.

---

## 3. Dataset Reference

### 3a. `positions.csv` — Primary Training Source

Each row = one board position after a move was played.

| Column | Type | Example | What It Means |
|--------|------|---------|---------------|
| `fen` | string | `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1` | Full board state in FEN notation |
| `playing` | string | `e2e4` | The move that was just played (UCI format) |
| `score` | int | `-35` | Engine evaluation in centipawns (negative = black is better) |
| `mate` | int/null | *(empty)* | Moves to mate, if applicable |
| `depth` | int | `20` | Engine search depth when evaluating |
| `game_id` | string | `j1dkb5dw` | Unique game identifier |
| `date` | string | `2012.12.31` | Game date |
| `time` | string | `23:01:03` | Game start time |
| `white` | string | `BFG9k` | White player username |
| `black` | string | `mamalak` | Black player username |
| `white_result` | int | `1` | White outcome: 1=win, 0=loss, 0.5=draw |
| `black_result` | int | `0` | Black outcome (mirror of white_result) |
| `white_elo` | int | `1639` | White player ELO rating |
| `black_elo` | int | `1403` | Black player ELO rating |
| `opening` | string | `French Defense: Normal Variation` | Opening name |
| `time_control` | string | `600+8` | Format: `base_seconds+increment_seconds` |
| `termination` | string | `Normal` | How the game ended |

### 3b. `Chess games stats.csv` — Game-Level Aggregates

Used in `chess.ipynb` for exploration. Each row = one full game summarised.

| Column | What It Means |
|--------|---------------|
| `Game ID` | Unique game identifier |
| `White Rating` | White ELO |
| `Black Rating` | Black ELO |
| `Opening ECO` | ECO code (e.g. `C20`) |
| `White Centi-pawn Loss` | Total centipawn loss for white |
| `White's Number of Inaccuracies` | Count of inaccuracies |
| `White's Number of Mistakes` | Count of mistakes |
| `White's Number of Blunders` | Count of blunders |
| *(same 4 columns for Black)* | — |

---

## 4. Data Pipeline

The pipeline lives in `src/preprocessing.py`.  
Think of it as an **assembly line** — raw CSV goes in, clean feature matrix comes out.

```
positions.csv
     │
     ▼
load_and_clean()       → drop rows missing score / elo / depth / fen
     │
     ▼
label_blunders()       → compare consecutive scores per game
     │                   if |score_diff| > 200 centipawns → blunder = 1
     ▼
add_fen_features()     → parse FEN string → material_diff, total_pieces, in_check
     │
     ▼
add_player_features()  → rating_diff, avg_rating, base_time (from time_control)
     │
     ▼
(X, y)                 → feature matrix + binary label
```

### What is a "centipawn"?

> 1 centipawn = 1/100th of a pawn's value.  
> A score drop of 200 centipawns means you effectively lost 2 pawns of advantage in one move.  
> That is the `BLUNDER_THRESHOLD` constant in `preprocessing.py`.

### Blunder Labelling — Step by Step

```python
# Sort all rows by game, then compute the evaluation difference between moves
df["eval_diff"] = df.groupby("game_id")["score"].diff()

# If the absolute drop is more than 200 centipawns → blunder
df["blunder"] = (df["eval_diff"].abs() > 200).astype(int)
```

---

## 5. Feature Engineering

### What We Feed the Model

```python
FEATURE_COLUMNS = [
    "white_elo",       # How strong is white?
    "black_elo",       # How strong is black?
    "depth",           # Engine analysis depth (data quality signal)
    "material_diff",   # White material minus black material (centipawns)
    "total_pieces",    # Total pieces on board → game phase proxy
    "in_check",        # Is side-to-move in check? (1 or 0)
    "rating_diff",     # white_elo - black_elo
    "avg_rating",      # (white_elo + black_elo) / 2
    "base_time",       # Base seconds from time_control string
]

X = df[FEATURE_COLUMNS]
y = df["blunder"]       # 0 = normal move, 1 = blunder
```

> **Extended feature set (from user requirements):**  
> The features below are the full target set. Some require FEN-to-move extraction
> or opening table lookups — currently the FEN-based subset is implemented.

```python
# Full feature roadmap
FEATURES = [
    'score',          # Raw engine eval (very powerful — but not available at inference without engine)
    'depth',          # ✅ implemented
    'score_diff',     # eval drop — this IS the label signal during training
    'turn',           # 0=White, 1=Black
    'piece_count',    # ✅ implemented as total_pieces
    'move_number',    # Which move in the game
    'is_opening',     # move_number < 15 (rough heuristic)
    'is_endgame',     # total_pieces < 12 (rough heuristic)
    'base_time',      # ✅ implemented
    'increment',      # +increment from time_control string
    'white_elo',      # ✅ implemented
    'black_elo',      # ✅ implemented
    'elo_diff',       # ✅ implemented as rating_diff
]
```

### Why These Features?

| Feature | Intuition |
|---------|-----------|
| `white_elo / black_elo` | Lower-rated players blunder more frequently |
| `material_diff` | Losing position → more desperate moves → more blunders |
| `total_pieces` | Endgames with fewer pieces are more precise; mistakes are more costly |
| `in_check` | Check forces narrow set of replies → higher blunder risk |
| `base_time` | Faster time controls → less thinking time → more blunders |
| `rating_diff` | Large skill gap → weaker side more likely to blunder under pressure |

### FEN Feature Extraction (`src/features.py`)

```python
def extract_fen_features(fen: str) -> tuple[int, int, int]:
    board = chess.Board(fen)
    # Material values: pawn=1, knight/bishop=3, rook=5, queen=9
    material_diff = white_material - black_material
    total_pieces   = len(board.piece_map())
    in_check       = int(board.is_check())
    return material_diff, total_pieces, in_check
```

---

## 6. Model — Why XGBoost

**XGBoost** (Extreme Gradient Boosting) is an ensemble of decision trees trained sequentially, where each tree corrects the errors of the previous one.

### Why Not Logistic Regression or Random Forest?

| Model | Problem |
|-------|---------|
| Logistic Regression | Linear — can't capture "low ELO + endgame + in check" interactions |
| Random Forest | Good, but slower and no native class-imbalance handling |
| **XGBoost** | **Handles imbalance via `scale_pos_weight`, fast, robust to noisy features** |

### Class Imbalance

Most moves are *not* blunders — maybe 1 in 10–15 moves is a blunder.  
XGBoost handles this automatically:

```python
ratio = count(non_blunders) / count(blunders)
# e.g. ratio = 12.0

model = XGBClassifier(
    scale_pos_weight=ratio,   # tells XGBoost blunders are 12x rarer
    n_estimators=600,
    max_depth=8,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

### Target Accuracy

- **Expected range: 70–80%**
- Below 70%: model is underfit — consider adding `score`, `move_number`, `is_endgame`
- Above 85%: check for data leakage (e.g. `score_diff` is the label signal — never include it as a feature)

---

## 7. Training Pipeline

All training logic is in `src/train.py`.

```
build_dataset(path)
        │
        ▼
train_test_split (80/20, stratified on label)
        │
        ▼
compute_scale_pos_weight
        │
        ▼
XGBClassifier.fit(X_train, y_train)
        │
        ▼
evaluate → accuracy_score + classification_report
        │
        ▼
save_model → models/blunder_model.pkl
```

### Running Training (minimal code)

```python
from src.preprocessing import build_dataset
from src.train import train, evaluate, save_model

X, y = build_dataset("data/raw/positions.csv")
model, X_test, y_test = train(X, y)
evaluate(model, X_test, y_test)
save_model(model)
```

---

## 8. Prediction — Two Input Options

Once the model is trained, there are **two ways** a player can interact with it.

---

### Option 1 — Enter a Full PGN Game (Best UX)

The player pastes or uploads a complete game in PGN format.  
The system analyses **every move** and returns a blunder probability for each one.

**How it works:**

```
PGN file
    │
    ▼
chess.pgn.read_game()          → parse headers (ELO, time control)
    │
    ▼
iterate mainline_moves()       → replay the game move by move
    │
    ▼
extract_fen_features(board.fen()) → material_diff, total_pieces, in_check
    │
    ▼
model.predict_proba(sample)    → [prob_normal, prob_blunder]
    │
    ▼
DataFrame: Move | SAN | Side | Blunder_Probability
```

**Minimal code:**

```python
from src.predict import predict_game

results = predict_game("data/raw/sample_game.pgn")
print(results.head(30))
# Output:
#    Move_Number Move_SAN   Side  Blunder_Probability
# 0            1      e4  White                0.081
# 1            1     e5   Black                0.094
# ...
```

**Sample PGN format:**

```pgn
[Event "Blitz"]
[White "BFG9k"]
[Black "mamalak"]
[WhiteElo "1639"]
[BlackElo "1403"]
[TimeControl "600+8"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 ...
```

---

### Option 2 — Enter One Position Manually (Simple Demo)

The player doesn't have a full game — they just want to test one specific position.  
They enter the raw numbers for a single board state.

**Minimal code:**

```python
import joblib
import pandas as pd

model = joblib.load("models/blunder_model.pkl")

# Player fills in these values
sample = pd.DataFrame([{
    "white_elo":     1639,    # your rating
    "black_elo":     1403,    # opponent rating
    "depth":         20,      # engine depth (use 20 as default)
    "material_diff": 0,       # 0 = even material
    "total_pieces":  28,      # pieces still on board
    "in_check":      0,       # 1 if you're in check
    "rating_diff":   236,     # white_elo - black_elo
    "avg_rating":    1521,    # (white_elo + black_elo) / 2
    "base_time":     600,     # base seconds (600 = 10 min game)
}])

prob = model.predict_proba(sample)[0][1]
print(f"Blunder probability: {prob:.1%}")
# → Blunder probability: 12.3%
```

**Interactive version (terminal UI):**

```python
def get_manual_prediction():
    print("=== Chess Blunder Predictor ===")
    white_elo  = int(input("Your rating (e.g. 1500): "))
    black_elo  = int(input("Opponent rating (e.g. 1400): "))
    pieces     = int(input("Total pieces on board (start=32): "))
    in_check   = int(input("Are you in check? (1=yes, 0=no): "))
    base_time  = int(input("Game time in seconds (600=10min): "))

    sample = [[
        white_elo, black_elo, 20,
        0,                           # material_diff (assume even)
        pieces, in_check,
        white_elo - black_elo,
        (white_elo + black_elo) / 2,
        base_time,
    ]]

    model  = joblib.load("models/blunder_model.pkl")
    prob   = model.predict_proba(sample)[0][1]
    label  = "⚠ LIKELY BLUNDER" if prob > 0.5 else "✓ Probably OK"
    print(f"\n{label}  (probability: {prob:.1%})")
```

---

### Comparison: Option 1 vs Option 2

| | Option 1 (PGN) | Option 2 (Manual) |
|---|---|---|
| Input | Full game PGN text | Single position values |
| Output | Per-move blunder table | Single probability |
| Best for | Post-game analysis | Quick position check |
| Requires | Valid PGN file | Just a few numbers |
| FEN parsing | Automatic | Not needed |

---

## 9. Notebooks Guide

| Notebook | Purpose | Status |
|----------|---------|--------|
| `blunder.ipynb` | **Main pipeline** — runs full end-to-end training and prediction | Ready to run |
| `chess.ipynb` | Exploratory — Multi-output regression + ECO opening classification | Exploration |
| `chess2.ipynb` | Extended exploration with multiple models | Exploration |
| `chess3.ipynb` | Additional experiments | Exploration |

> **Start here:** `blunder.ipynb` — it's the cleanest, most structured notebook.  
> The `chess*.ipynb` notebooks are scratchpads showing the experimentation process — messy but educational.

---

## 10. System Requirements

### Python Version

```
Python >= 3.10   (type hints like tuple[int, int, int] require 3.10+)
```

### Required Packages

```txt
# Core ML
xgboost>=1.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# Chess utilities
chess>=1.10.0       # python-chess — FEN parsing, PGN reading, board logic

# Model persistence
joblib>=1.3.0

# Jupyter (for notebooks)
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.0.0
```

### Install All Dependencies

```bash
pip install xgboost scikit-learn pandas numpy chess joblib jupyter
```

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| CPU | Any dual-core | Quad-core (XGBoost uses multi-threading) |
| Disk | 500 MB | 1 GB |
| GPU | Not required | Not required |

> XGBoost is CPU-optimised — no GPU needed for this dataset size.

---

## 11. Quick Start

### Step 1 — Clone and set up environment

```bash
cd CW
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install xgboost scikit-learn pandas numpy chess joblib jupyter
```

### Step 2 — Verify data files exist

```
data/raw/positions.csv          ← required for training
data/raw/Chess games stats.csv  ← required for chess.ipynb exploration
data/raw/sample_game.pgn        ← required for prediction demo
```

### Step 3 — Train the model

Open `blunder.ipynb` and run all cells in order, **or** run programmatically:

```python
# train_and_save.py
import sys, os
sys.path.insert(0, os.getcwd())

from src.preprocessing import build_dataset
from src.train import train, evaluate, save_model

X, y = build_dataset("data/raw/positions.csv")
model, X_test, y_test = train(X, y)
evaluate(model, X_test, y_test)
save_model(model)
```

```bash
python train_and_save.py
```

Expected output:
```
Original shape: (XXXXX, 17)
After cleaning: (XXXXX, 17)
After blunder labelling: (XXXXX, 17)
...
==============================
XGBOOST RESULTS
==============================
Accuracy: 74.XX %
              precision    recall  f1-score   support
           0       0.XX      0.XX      0.XX      XXXX
           1       0.XX      0.XX      0.XX      XXXX
```

### Step 4 — Predict on a game

```python
from src.predict import predict_game

df = predict_game("data/raw/sample_game.pgn")
print(df[df["Blunder_Probability"] > 0.5])   # show suspected blunders
```

---

## 12. FAQ

**Q: Why is the accuracy "only" 70–80%?**  
A: A blunder depends on the *next* move the player makes — which is inherently unpredictable. Without a live engine evaluation, this range is the realistic maximum for ML-only prediction.

**Q: What is FEN?**  
A: Forsyth-Edwards Notation — a single string that completely describes a chess position. Example: `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1`. The `chess` library parses these for us.

**Q: Why use `score` during training but not during prediction?**  
A: During training, `positions.csv` already has engine scores pre-computed. At prediction time on a live PGN, we don't run an engine — so features that *don't* require an engine (ELO, FEN geometry, time control) are used instead.

**Q: What does `scale_pos_weight` do?**  
A: Most moves are not blunders (~1 in 12). Without correction, the model would just predict "not blunder" for everything and get ~92% accuracy without learning anything useful. `scale_pos_weight` forces the model to treat each blunder as if it appeared 12x more often.

**Q: Can I add more features?**  
A: Yes. The easiest additions are `move_number` (already in FEN's half-move clock), `is_opening` (`move_number < 15`), and `is_endgame` (`total_pieces < 12`). Add them in `src/preprocessing.py → add_player_features()` and update `FEATURE_COLUMNS`.

**Q: How do I use this with my own games from Chess.com or Lichess?**  
A: Export your game as PGN from either platform and pass the file path to `predict_game("your_game.pgn")`.

---

*Architecture document — last updated February 2026*
