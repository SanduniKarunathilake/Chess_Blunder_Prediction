"""
src/preprocessing.py
--------------------
Full data pipeline: raw CSV → clean feature matrix (X, y).

Pipeline stages
---------------
1. load_and_clean()      — load positions.csv, drop unusable rows
2. label_blunders()      — normalise evals to White's perspective,
                           compute cp_loss per move → binary label
3. add_fen_features()    — parse FEN → material_diff, total_pieces, in_check,
                           move_number, game_phase
4. add_player_features() — rating_diff, avg_rating, base_time
5. build_dataset()       — orchestrates all stages, returns (X, y)

Key fixes (v2)
--------------
- Engine scores normalised to White's perspective before computing cp_loss.
  (Raw scores alternate perspective each half-move.)
- Blunder = centipawn LOSS **by the mover** ≥ 300 cp.
- New features: eval_before, move_number, game_phase.
- Removed 'depth' (engine analysis depth is not a game feature).

Constants
---------
BLUNDER_THRESHOLD : int
    Centipawn loss required to label a move as a blunder (300 cp).
FEATURE_COLUMNS : list[str]
    The exact set of columns fed to the model.
"""

import re

import numpy as np
import pandas as pd

from src.features import extract_fen_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLUNDER_THRESHOLD = 300   # centipawns — loss ≥ 300 cp by the mover

FEATURE_COLUMNS = [
    "white_elo",
    "black_elo",
    "material_diff",
    "total_pieces",
    "in_check",
    "rating_diff",
    "avg_rating",
    "base_time",
    "eval_before",      # engine eval (White perspective) before the move
    "move_number",      # full-move number in the game
    "game_phase",       # 0 = opening, 1 = middlegame, 2 = endgame
]


# ---------------------------------------------------------------------------
# Stage 1 — Load & clean
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load positions.csv and drop rows that cannot be used for training.

    Rows are dropped when any of the following columns are missing:
    score, white_elo, black_elo, fen, game_id.

    Parameters
    ----------
    path : str
        Path to positions.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no missing values in critical columns.
    """
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    required = ["score", "white_elo", "black_elo", "fen", "game_id"]
    df = df.dropna(subset=required)

    # Ensure numeric types
    df["score"]     = pd.to_numeric(df["score"],     errors="coerce")
    df["white_elo"] = pd.to_numeric(df["white_elo"], errors="coerce")
    df["black_elo"] = pd.to_numeric(df["black_elo"], errors="coerce")

    df = df.dropna(subset=required)
    df = df.reset_index(drop=True)

    print(f"After cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — Blunder labelling  (FIXED — v2)
# ---------------------------------------------------------------------------

def _side_to_move(fen: str) -> str:
    """Return 'w' or 'b' from a FEN string."""
    parts = fen.split()
    return parts[1] if len(parts) >= 2 else "w"


def label_blunders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary ``blunder`` column by computing centipawn loss per move.

    Steps
    -----
    1. Determine whose turn it is from the FEN (``side_to_move``).
       The engine score in ``positions.csv`` is from the perspective of the
       side to move.  We normalise every score to **White's perspective**::

           score_white =  score   if side_to_move == 'w'
           score_white = -score   if side_to_move == 'b'

    2. Compute ``cp_loss`` for the mover.
       Each row is the position *after* a move.  The mover is the
       side that just played (opposite of ``side_to_move`` in the FEN).

       - If White just moved (``side_to_move == 'b'``)::
             cp_loss = eval_w_before − eval_w_after

       - If Black just moved (``side_to_move == 'w'``)::
             cp_loss = eval_w_after − eval_w_before

       A positive ``cp_loss`` means the mover worsened their position.

    3. ``blunder = (cp_loss >= BLUNDER_THRESHOLD)``

    Also stores ``eval_before`` (White-perspective eval of the previous
    position) as a training feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from ``load_and_clean()``.

    Returns
    -------
    pd.DataFrame
        Original + columns: ``side_to_move``, ``score_white``,
        ``eval_before``, ``cp_loss``, ``blunder``.
    """
    df = df.copy()

    # Sort within each game by the natural row order
    df = df.sort_values(["game_id"]).reset_index(drop=True)

    # 1. Normalise score to White's perspective
    df["side_to_move"] = df["fen"].apply(_side_to_move)
    df["score_white"] = np.where(
        df["side_to_move"] == "w",
        df["score"],
        -df["score"],
    )

    # 2. eval_before = previous position's normalised score (within game)
    df["eval_before"] = df.groupby("game_id")["score_white"].shift(1)

    # Who moved?  The mover is the *opposite* of side_to_move in the FEN.
    #   side_to_move == 'b' → White just moved
    #   side_to_move == 'w' → Black just moved
    white_moved = df["side_to_move"] == "b"

    df["cp_loss"] = np.where(
        white_moved,
        df["eval_before"] - df["score_white"],      # White's loss
        df["score_white"] - df["eval_before"],       # Black's loss
    )

    # 3. Blunder when mover lost ≥ threshold
    df["blunder"] = (df["cp_loss"] >= BLUNDER_THRESHOLD).astype(int)

    # Drop first move of each game (no eval_before → NaN)
    df = df.dropna(subset=["eval_before"]).reset_index(drop=True)

    blunder_rate = df["blunder"].mean() * 100
    print(f"After blunder labelling: {df.shape}  |  blunder rate: {blunder_rate:.1f}%")
    return df


# ---------------------------------------------------------------------------
# Stage 3 — FEN features
# ---------------------------------------------------------------------------

def add_fen_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse every FEN string and add five columns:
        material_diff, total_pieces, in_check, move_number, game_phase.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'fen' column.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with five new columns appended.
    """
    df = df.copy()

    results = df["fen"].apply(extract_fen_features)
    df["material_diff"] = results.apply(lambda t: t[0])
    df["total_pieces"]  = results.apply(lambda t: t[1])
    df["in_check"]      = results.apply(lambda t: t[2])
    df["move_number"]   = results.apply(lambda t: t[3])
    df["game_phase"]    = results.apply(lambda t: t[4])

    print("FEN features added: material_diff, total_pieces, in_check, "
          "move_number, game_phase")
    return df


# ---------------------------------------------------------------------------
# Stage 4 — Player / game-level features
# ---------------------------------------------------------------------------

def _parse_base_time(tc: str) -> float:
    """
    Extract the base seconds from a time-control string like '600+8'.

    Returns 300.0 (5 min) as a fallback for missing / unparseable values.
    """
    if not isinstance(tc, str):
        return 300.0
    match = re.match(r"^(\d+)", tc)
    return float(match.group(1)) if match else 300.0


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive rating_diff, avg_rating, and base_time from existing columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with white_elo, black_elo, and time_control columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with three new columns appended.
    """
    df = df.copy()

    df["rating_diff"] = df["white_elo"] - df["black_elo"]
    df["avg_rating"]  = (df["white_elo"] + df["black_elo"]) / 2

    if "time_control" in df.columns:
        df["base_time"] = df["time_control"].apply(_parse_base_time)
    else:
        df["base_time"] = 300.0

    print(f"Player features added: rating_diff, avg_rating, base_time")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the full pipeline and return (X, y) ready for model training.

    Parameters
    ----------
    path : str
        Path to positions.csv.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns = FEATURE_COLUMNS.
    y : pd.Series
        Binary blunder labels (0 = normal, 1 = blunder).
    """
    df = load_and_clean(path)
    df = label_blunders(df)
    df = add_fen_features(df)
    df = add_player_features(df)

    # Drop any rows where feature engineering produced NaNs
    df = df.dropna(subset=FEATURE_COLUMNS + ["blunder"]).reset_index(drop=True)

    X = df[FEATURE_COLUMNS]
    y = df["blunder"]

    print(f"\nFinal dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Blunders: {y.sum():,}  ({y.mean()*100:.1f}%)")
    return X, y
