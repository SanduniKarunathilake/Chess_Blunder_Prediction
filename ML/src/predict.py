"""
src/predict.py
--------------
Load a trained model **and a Stockfish engine**, then predict blunder
probability for moves in a PGN game.

v2 changes
----------
- Uses Stockfish to evaluate every position (before and after each move).
- Centipawn loss (cp_loss) is computed from the engine evaluations.
- Blunder is determined by cp_loss ≥ 300, **not** by the ML model alone.
- The ML model provides a supplementary "risk score" based on features.

Functions
---------
predict_game(pgn_path, model_path, engine_path, depth)
    → DataFrame with per-move blunder analysis.
"""

import os
import re

import chess
import chess.pgn
import chess.engine
import joblib
import pandas as pd

from src.features import extract_fen_features

DEFAULT_MODEL_PATH  = os.path.join("models", "blunder_model.pkl")
DEFAULT_ENGINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "backend", "engine", "stockfish",
    "stockfish-windows-x86-64-avx2.exe",
)

BLUNDER_THRESHOLD = 300   # centipawns

FEATURE_COLUMNS = [
    "white_elo",
    "black_elo",
    "material_diff",
    "total_pieces",
    "in_check",
    "rating_diff",
    "avg_rating",
    "base_time",
    "eval_before",
    "move_number",
    "game_phase",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_elo(value, fallback: int = 1500) -> int:
    """Safely convert an ELO header value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _parse_base_time(tc: str) -> float:
    """Extract base seconds from '600+8' style time-control strings."""
    if not isinstance(tc, str):
        return 300.0
    match = re.match(r"^(\d+)", tc)
    return float(match.group(1)) if match else 300.0


def _eval_cp(info) -> int:
    """Extract centipawn score from White's perspective, capping mates."""
    score = info["score"].white()
    return score.score(mate_score=10_000)


# ---------------------------------------------------------------------------
# Full PGN game analysis (engine + model)
# ---------------------------------------------------------------------------

def predict_game(
    pgn_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    engine_path: str = DEFAULT_ENGINE_PATH,
    depth: int = 16,
) -> pd.DataFrame:
    """
    Replay every move in a PGN game, evaluate each position with Stockfish,
    and flag blunders based on centipawn loss.

    Parameters
    ----------
    pgn_path : str
        Path to a .pgn file (single game).
    model_path : str
        Path to the saved model pickle.
    engine_path : str
        Path to the Stockfish binary.
    depth : int
        Stockfish analysis depth (default 16).

    Returns
    -------
    pd.DataFrame
        Columns: Move_Number, Move_SAN, Side, Eval_Before, Eval_After,
                 CP_Loss, Is_Blunder, ML_Risk
    """
    model = joblib.load(model_path)

    with open(pgn_path, encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)

    if game is None:
        raise ValueError(f"No games found in {pgn_path}")

    # --- Read game-level headers ---
    white_elo = _parse_elo(game.headers.get("WhiteElo"))
    black_elo = _parse_elo(game.headers.get("BlackElo"))
    tc_str    = game.headers.get("TimeControl", "600+0")
    base_time = _parse_base_time(tc_str)

    rating_diff = white_elo - black_elo
    avg_rating  = (white_elo + black_elo) / 2

    limit = chess.engine.Limit(depth=depth)

    records = []
    engine  = chess.engine.SimpleEngine.popen_uci(engine_path)

    try:
        board = game.board()

        # Evaluate starting position
        eval_before_w = _eval_cp(engine.analyse(board, limit))

        for move_idx, move in enumerate(game.mainline_moves(), start=1):
            san  = board.san(move)
            side = "White" if board.turn == chess.WHITE else "Black"
            white_moved = board.turn == chess.WHITE

            # Board features BEFORE the move
            material_diff, total_pieces, in_check, move_number, game_phase = (
                extract_fen_features(board.fen())
            )

            # Push the move
            board.push(move)

            # Evaluate position AFTER the move
            eval_after_w = _eval_cp(engine.analyse(board, limit))

            # Centipawn loss for the mover (from their perspective)
            if white_moved:
                cp_loss = eval_before_w - eval_after_w
            else:
                cp_loss = eval_after_w - eval_before_w

            is_blunder = cp_loss >= BLUNDER_THRESHOLD

            # ML model risk score
            sample = pd.DataFrame([{
                "white_elo":     white_elo,
                "black_elo":     black_elo,
                "material_diff": material_diff,
                "total_pieces":  total_pieces,
                "in_check":      in_check,
                "rating_diff":   rating_diff,
                "avg_rating":    avg_rating,
                "base_time":     base_time,
                "eval_before":   eval_before_w,
                "move_number":   move_number,
                "game_phase":    game_phase,
            }])
            ml_risk = float(model.predict_proba(sample[FEATURE_COLUMNS])[0][1])

            records.append({
                "Move_Number":  (move_idx + 1) // 2,
                "Move_SAN":     san,
                "Side":         side,
                "Eval_Before":  eval_before_w,
                "Eval_After":   eval_after_w,
                "CP_Loss":      max(cp_loss, 0),
                "Is_Blunder":   is_blunder,
                "ML_Risk":      round(ml_risk, 4),
            })

            # Next iteration's eval_before = this iteration's eval_after
            eval_before_w = eval_after_w

    finally:
        engine.quit()

    return pd.DataFrame(records)
