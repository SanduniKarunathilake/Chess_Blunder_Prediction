"""
backend/app.py
--------------
Flask REST API:

  POST /api/predict-pgn      → per-move blunder analysis (Stockfish + ML)
  POST /api/opening          → opening recommendation
  POST /api/elo              → rating change after one game
  POST /api/elo/session      → rating changes across multiple games

PGN analysis uses Stockfish engine to compute centipawn loss per move.
Blunder = cp_loss ≥ 300. ML model provides a supplementary risk score.
"""

import io
import os
import re
import sys
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

import chess
import chess.pgn
import chess.engine
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # allow frontend to call the API

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "blunder_model.pkl")
OPENING_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "chess_opening_model.pkl")
ENGINE_PATH = os.path.join(
    os.path.dirname(__file__), "engine", "stockfish",
    "stockfish-windows-x86-64-avx2.exe",
)

BLUNDER_THRESHOLD = 300   # centipawn loss threshold for blunder classification
ENGINE_DEPTH      = 16    # Stockfish analysis depth (balance speed vs accuracy)

# Load blunder model once at startup
try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Blunder model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[WARN] Could not load blunder model: {e}")

# Verify Stockfish engine is available
if os.path.exists(ENGINE_PATH):
    print(f"[OK] Stockfish found at {ENGINE_PATH}")
else:
    print(f"[WARN] Stockfish NOT found at {ENGINE_PATH}")
    print("       PGN analysis will not work without the engine.")

# Load opening model once at startup
try:
    _opening_data    = joblib.load(OPENING_MODEL_PATH)
    opening_model    = _opening_data["model"]
    opening_le_eco   = _opening_data["le_eco"]
    opening_features = _opening_data["features"]
    print(f"[OK] Opening model loaded from {OPENING_MODEL_PATH}")
    print(f"     {len(opening_le_eco.classes_)} ECO codes available")
except Exception as e:
    opening_model = None
    opening_le_eco = None
    opening_features = None
    print(f"[WARN] Could not load opening model: {e}")

# ECO code → human-readable opening name lookup
ECO_NAMES = {
    "A00": "Irregular Openings", "A01": "Nimzowitsch-Larsen Attack", "A02": "Bird's Opening",
    "A03": "Bird's Opening", "A04": "Réti Opening", "A05": "Réti Opening",
    "A06": "Réti Opening", "A07": "King's Indian Attack", "A08": "King's Indian Attack",
    "A09": "Réti Opening", "A10": "English Opening", "A11": "English Opening: Caro-Kann",
    "A13": "English Opening", "A15": "English Opening: Anglo-Indian",
    "A16": "English Opening", "A17": "English Opening: Hedgehog",
    "A20": "English Opening", "A21": "English Opening", "A22": "English Opening: Bremen",
    "A25": "English Opening: Sicilian Reversed", "A28": "English Opening: Four Knights",
    "A30": "English Opening: Symmetrical", "A34": "English Opening: Symmetrical",
    "A35": "English Opening: Symmetrical", "A36": "English Opening: Symmetrical",
    "A40": "Queen's Pawn Game", "A41": "Queen's Pawn: Robatsch",
    "A42": "Modern Defense", "A43": "Old Benoni", "A44": "Old Benoni",
    "A45": "Queen's Pawn Game", "A46": "Queen's Pawn Game",
    "A48": "King's Indian: London System", "A49": "King's Indian: Fianchetto",
    "A50": "Queen's Pawn", "A51": "Budapest Gambit", "A52": "Budapest Gambit",
    "A53": "Old Indian Defense", "A56": "Benoni Defense",
    "A57": "Benko Gambit", "A58": "Benko Gambit Accepted",
    "A60": "Benoni Defense", "A65": "Benoni: Classical",
    "A80": "Dutch Defense", "A84": "Dutch Defense", "A85": "Dutch Defense",
    "B00": "Uncommon King's Pawn", "B01": "Scandinavian Defense",
    "B02": "Alekhine's Defense", "B03": "Alekhine's Defense", "B04": "Alekhine's Defense",
    "B06": "Modern Defense", "B07": "Pirc Defense", "B08": "Pirc: Classical",
    "B09": "Pirc: Austrian Attack", "B10": "Caro-Kann Defense",
    "B11": "Caro-Kann: Two Knights", "B12": "Caro-Kann Defense",
    "B13": "Caro-Kann: Exchange", "B15": "Caro-Kann", "B17": "Caro-Kann: Steinitz",
    "B18": "Caro-Kann: Classical", "B20": "Sicilian Defense",
    "B21": "Sicilian: Smith-Morra Gambit", "B22": "Sicilian: Alapin",
    "B23": "Sicilian: Closed", "B24": "Sicilian: Closed",
    "B25": "Sicilian: Closed", "B27": "Sicilian Defense",
    "B28": "Sicilian: O'Kelly", "B29": "Sicilian: Nimzowitsch",
    "B30": "Sicilian Defense", "B31": "Sicilian: Rossolimo",
    "B32": "Sicilian: Löwenthal", "B33": "Sicilian: Sveshnikov",
    "B34": "Sicilian: Accelerated Dragon", "B35": "Sicilian: Accelerated Dragon",
    "B36": "Sicilian: Maroczy Bind", "B40": "Sicilian Defense",
    "B41": "Sicilian: Kan", "B43": "Sicilian: Kan",
    "B44": "Sicilian: Taimanov", "B45": "Sicilian: Taimanov",
    "B46": "Sicilian: Taimanov", "B50": "Sicilian Defense",
    "B51": "Sicilian: Moscow", "B52": "Sicilian: Canal-Sokolsky",
    "B53": "Sicilian", "B54": "Sicilian: Richter-Rauzer",
    "B56": "Sicilian", "B57": "Sicilian: Sozin",
    "B70": "Sicilian: Dragon", "B72": "Sicilian: Dragon",
    "B76": "Sicilian: Dragon: Yugoslav Attack",
    "B80": "Sicilian: Scheveningen", "B90": "Sicilian: Najdorf",
    "B92": "Sicilian: Najdorf", "B94": "Sicilian: Najdorf",
    "B95": "Sicilian: Najdorf",
    "C00": "French Defense", "C01": "French: Exchange", "C02": "French: Advance",
    "C03": "French: Tarrasch", "C05": "French: Tarrasch",
    "C07": "French: Tarrasch", "C10": "French Defense",
    "C11": "French: Classical", "C13": "French: Classical",
    "C15": "French: Winawer", "C18": "French: Winawer",
    "C20": "King's Pawn Game", "C21": "Center Game",
    "C22": "Center Game", "C23": "Bishop's Opening",
    "C24": "Bishop's Opening", "C25": "Vienna Game",
    "C26": "Vienna Game", "C27": "Vienna Game",
    "C28": "Vienna Game", "C30": "King's Gambit",
    "C31": "King's Gambit Declined", "C33": "King's Gambit Accepted",
    "C34": "King's Gambit Accepted", "C35": "King's Gambit Accepted",
    "C36": "King's Gambit Accepted", "C37": "King's Gambit Accepted",
    "C40": "King's Knight Opening", "C41": "Philidor Defense",
    "C42": "Petrov's Defense", "C43": "Petrov: Stafford Gambit",
    "C44": "King's Pawn Game", "C45": "Scotch Game",
    "C46": "Three Knights Game", "C47": "Four Knights Game",
    "C48": "Four Knights: Spanish", "C49": "Four Knights: Double Spanish",
    "C50": "Italian Game", "C51": "Evans Gambit",
    "C53": "Italian Game: Giuoco Piano", "C54": "Italian Game: Giuoco Piano",
    "C55": "Italian Game: Two Knights", "C57": "Two Knights: Traxler",
    "C58": "Two Knights Defense", "C60": "Ruy López",
    "C61": "Ruy López: Bird's", "C62": "Ruy López: Old Steinitz",
    "C63": "Ruy López: Schliemann", "C64": "Ruy López: Classical",
    "C65": "Ruy López: Berlin", "C66": "Ruy López: Berlin",
    "C67": "Ruy López: Berlin", "C68": "Ruy López: Exchange",
    "C69": "Ruy López: Exchange", "C70": "Ruy López",
    "C77": "Ruy López: Morphy", "C78": "Ruy López: Arkhangelsk",
    "C80": "Ruy López: Open", "C88": "Ruy López: Closed",
    "C89": "Ruy López: Marshall Attack",
    "D00": "Queen's Pawn Game", "D01": "Richter-Veresov Attack",
    "D02": "Queen's Pawn Game", "D03": "Torre Attack",
    "D04": "Queen's Pawn Game", "D05": "Colle System",
    "D06": "Queen's Gambit", "D07": "Queen's Gambit: Chigorin",
    "D08": "Queen's Gambit: Albin Counter-Gambit",
    "D10": "Queen's Gambit Declined: Slav", "D11": "Slav Defense",
    "D12": "Slav Defense", "D15": "Slav Defense",
    "D20": "Queen's Gambit Accepted", "D21": "Queen's Gambit Accepted",
    "D25": "Queen's Gambit Accepted",
    "D30": "Queen's Gambit Declined", "D31": "Queen's Gambit Declined",
    "D32": "Queen's Gambit Declined: Tarrasch",
    "D35": "Queen's Gambit Declined: Exchange",
    "D37": "Queen's Gambit Declined",
    "D40": "Queen's Gambit Declined: Semi-Tarrasch",
    "D43": "Queen's Gambit Declined: Semi-Slav",
    "D44": "Queen's Gambit Declined: Semi-Slav",
    "D45": "Queen's Gambit Declined: Semi-Slav",
    "D50": "Queen's Gambit Declined", "D52": "Queen's Gambit Declined",
    "D53": "Queen's Gambit Declined", "D55": "Queen's Gambit Declined",
    "D80": "Grünfeld Defense", "D85": "Grünfeld Defense", "D94": "Grünfeld Defense",
    "E00": "Queen's Pawn: Catalan", "E10": "Queen's Pawn Game",
    "E11": "Bogo-Indian Defense", "E20": "Nimzo-Indian Defense",
    "E21": "Nimzo-Indian Defense", "E30": "Nimzo-Indian Defense",
    "E32": "Nimzo-Indian: Classical", "E60": "King's Indian Defense",
    "E61": "King's Indian Defense", "E70": "King's Indian Defense",
    "E76": "King's Indian: Four Pawns Attack",
    "E77": "King's Indian: Four Pawns Attack",
    "E81": "King's Indian: Sämisch",
    "E90": "King's Indian: Classical", "E91": "King's Indian: Classical",
}

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

# Piece material values in centipawns (same as ML/src/features.py)
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_int(value, name, lo=None, hi=None):
    """Return (parsed_int, error_string_or_None)."""
    if value is None:
        return None, f"'{name}' is required."
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None, f"'{name}' must be an integer."
    if lo is not None and v < lo:
        return None, f"'{name}' must be >= {lo}."
    if hi is not None and v > hi:
        return None, f"'{name}' must be <= {hi}."
    return v, None


def validate_float(value, name, lo=None, hi=None):
    if value is None:
        return None, f"'{name}' is required."
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, f"'{name}' must be a number."
    if lo is not None and v < lo:
        return None, f"'{name}' must be >= {lo}."
    if hi is not None and v > hi:
        return None, f"'{name}' must be <= {hi}."
    return v, None


# ---------------------------------------------------------------------------
# FEN helpers (inlined — no dependency on ML/src)
# ---------------------------------------------------------------------------

def extract_fen_features(fen: str):
    """Parse FEN → (material_diff, total_pieces, in_check, move_number, game_phase)."""
    board = chess.Board(fen)
    white = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_diff = white - black
    total_pieces  = len(board.piece_map())
    in_check      = int(board.is_check())
    move_number   = board.fullmove_number
    game_phase    = 0 if total_pieces >= 28 else (1 if total_pieces >= 14 else 2)
    return material_diff, total_pieces, in_check, move_number, game_phase


def parse_base_time(tc: str) -> float:
    if not isinstance(tc, str):
        return 300.0
    m = re.match(r"^(\d+)", tc)
    return float(m.group(1)) if m else 300.0


# ---------------------------------------------------------------------------
# ELO helpers (inlined — no dependency on ML/src)
# ---------------------------------------------------------------------------

def get_k_factor(rating: float, games_played: int, age: int | None = None) -> int:
    if games_played < 30 or (age is not None and age < 18):
        return 40
    if rating < 2400:
        return 20
    return 10


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def outcome_score(outcome: str) -> float:
    o = outcome.strip().lower()
    if o in ("win", "w", "1"):
        return 1.0
    if o in ("draw", "d", "0.5"):
        return 0.5
    if o in ("loss", "lose", "l", "0"):
        return 0.0
    raise ValueError(f"Unknown outcome '{outcome}'")


def compute_rating_change(rating, opp_rating, outcome, games_played=30, age=None):
    k = get_k_factor(rating, games_played, age)
    e = elo_expected(rating, opp_rating)
    s = outcome_score(outcome)
    delta = round(k * (s - e), 1)
    new_r = round(rating + delta, 1)

    if k == 40:
        if age is not None and age < 18:
            k_reason = f"junior (age {age} < 18)"
        else:
            k_reason = f"provisional (games {games_played} < 30)"
    elif k == 20:
        k_reason = f"standard (rating {rating} < 2400)"
    else:
        k_reason = f"elite (rating {rating} >= 2400)"

    return {
        "old_rating":      rating,
        "new_rating":      new_r,
        "change":          delta,
        "expected_score":  round(e, 4),
        "actual_score":    s,
        "k_factor":        k,
        "k_reason":        k_reason,
        "win_probability": round(e * 100, 1),
        "outcome":         outcome.lower(),
        "opponent_rating": opp_rating,
        "age":             age,
        "games_played":    games_played,
    }


# =========================================================================
#  ROUTES
# =========================================================================

# ---------------------------------------------------------------------------
# POST /api/opening  — opening recommendation based on ELO + color + time
# ---------------------------------------------------------------------------

@app.route("/api/opening", methods=["POST"])
def api_opening():
    """
    Expects JSON:
    {
        "your_elo": 1500,
        "opponent_elo": 1400,
        "color": "white",        // "white" or "black"
        "base_time": 600,        // seconds
        "increment": 0           // optional, default 0
    }

    Returns the top-N openings ranked by predicted win probability.
    """
    if opening_model is None:
        return jsonify({"error": "Opening model not loaded."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    errors = []

    your_elo, err = validate_int(data.get("your_elo"), "your_elo", 100, 3500)
    if err: errors.append(err)

    opponent_elo, err = validate_int(data.get("opponent_elo"), "opponent_elo", 100, 3500)
    if err: errors.append(err)

    color = data.get("color", "").strip().lower()
    if color not in ("white", "black"):
        errors.append("'color' must be 'white' or 'black'.")

    base_time, err = validate_float(data.get("base_time"), "base_time", 0, 10800)
    if err: errors.append(err)

    increment = data.get("increment", 0)
    try:
        increment = float(increment)
    except (TypeError, ValueError):
        errors.append("'increment' must be a number.")

    if errors:
        return jsonify({"errors": errors}), 422

    # Assign ELOs based on color
    if color == "white":
        white_elo, black_elo = your_elo, opponent_elo
    else:
        white_elo, black_elo = opponent_elo, your_elo

    elo_diff = white_elo - black_elo

    # Run prediction for every known ECO code
    eco_classes = list(opening_le_eco.classes_)
    eco_encoded = opening_le_eco.transform(eco_classes)

    rows = []
    for eco_code, enc in zip(eco_classes, eco_encoded):
        rows.append({
            "WhiteElo":     white_elo,
            "BlackElo":     black_elo,
            "EloDiff":      elo_diff,
            "BaseTime":     base_time,
            "Increment":    increment,
            "ECO_encoded":  enc,
        })

    df = pd.DataFrame(rows)
    probas = opening_model.predict_proba(df[opening_features])
    win_probs = probas[:, 1]  # probability of class 1 (win)

    # Build results sorted by win probability descending
    results = []
    for i, eco_code in enumerate(eco_classes):
        wp = float(win_probs[i])
        results.append({
            "eco_code":       eco_code,
            "opening_name":   ECO_NAMES.get(eco_code, eco_code),
            "win_probability": round(wp, 4),
            "win_percent":     f"{wp * 100:.1f}%",
        })

    results.sort(key=lambda x: x["win_probability"], reverse=True)

    return jsonify({
        "color":         color,
        "your_elo":      your_elo,
        "opponent_elo":  opponent_elo,
        "base_time":     base_time,
        "increment":     increment,
        "total_openings": len(results),
        "openings":      results,
    })


# ---------------------------------------------------------------------------
# POST /api/predict-pgn  — full PGN game analysis (Stockfish + ML)
# ---------------------------------------------------------------------------

@app.route("/api/predict-pgn", methods=["POST"])
def api_predict_pgn():
    """
    Expects JSON:
    {
        "pgn": "[Event \"...\"]\n1. e4 e5 2. ...",
        "white_elo": 1500,    // optional override
        "black_elo": 1400,    // optional override
        "time_control": "600+8"  // optional override
    }

    Uses Stockfish to evaluate every position.
    Blunder = centipawn loss ≥ 300 by the mover.
    Also provides the ML model's risk score for context.
    """
    if not os.path.exists(ENGINE_PATH):
        return jsonify({"error": "Stockfish engine not found. Cannot analyse PGN."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    pgn_text = data.get("pgn", "").strip()
    if not pgn_text:
        return jsonify({"error": "'pgn' field is required and must not be empty."}), 422

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception as e:
        return jsonify({"error": f"Failed to parse PGN: {str(e)}"}), 422

    if game is None:
        return jsonify({"error": "No valid game found in the PGN text."}), 422

    # ELO / time control — use overrides or PGN headers
    white_elo = data.get("white_elo")
    if white_elo is None:
        try:
            white_elo = int(game.headers.get("WhiteElo", 1500))
        except ValueError:
            white_elo = 1500
    else:
        white_elo, err = validate_int(white_elo, "white_elo", 100, 3500)
        if err:
            return jsonify({"error": err}), 422

    black_elo = data.get("black_elo")
    if black_elo is None:
        try:
            black_elo = int(game.headers.get("BlackElo", 1500))
        except ValueError:
            black_elo = 1500
    else:
        black_elo, err = validate_int(black_elo, "black_elo", 100, 3500)
        if err:
            return jsonify({"error": err}), 422

    tc_str    = data.get("time_control", game.headers.get("TimeControl", "600+0"))
    base_time = parse_base_time(tc_str)

    rating_diff = white_elo - black_elo
    avg_rating  = (white_elo + black_elo) / 2

    limit = chess.engine.Limit(depth=ENGINE_DEPTH)

    records = []
    engine  = None

    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        board  = game.board()

        # Evaluate starting position
        eval_before_w = engine.analyse(board, limit)["score"].white().score(mate_score=10_000)

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
            eval_after_w = engine.analyse(board, limit)["score"].white().score(mate_score=10_000)

            # Centipawn loss for the mover
            if white_moved:
                cp_loss = eval_before_w - eval_after_w
            else:
                cp_loss = eval_after_w - eval_before_w

            is_blunder = cp_loss >= BLUNDER_THRESHOLD

            # ML model risk score (if model is loaded)
            ml_risk = None
            if model is not None:
                try:
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
                    ml_risk = round(float(
                        model.predict_proba(sample[FEATURE_COLUMNS])[0][1]
                    ), 4)
                except Exception:
                    ml_risk = None

            records.append({
                "move_number":    (move_idx + 1) // 2,
                "half_move":      move_idx,
                "move_san":       san,
                "side":           side,
                "eval_before":    eval_before_w,
                "eval_after":     eval_after_w,
                "cp_loss":        max(cp_loss, 0),
                "is_blunder":     is_blunder,
                "ml_risk":        ml_risk,
            })

            # Next iteration
            eval_before_w = eval_after_w

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Engine analysis failed: {str(e)}"}), 500
    finally:
        if engine is not None:
            engine.quit()

    blunder_count = sum(1 for r in records if r["is_blunder"])

    return jsonify({
        "total_moves":      len(records),
        "blunders_found":   blunder_count,
        "white_elo":        white_elo,
        "black_elo":        black_elo,
        "engine_depth":     ENGINE_DEPTH,
        "blunder_threshold": BLUNDER_THRESHOLD,
        "moves":            records,
    })


# ---------------------------------------------------------------------------
# POST /api/elo  — single game rating calculation
# ---------------------------------------------------------------------------

@app.route("/api/elo", methods=["POST"])
def api_elo():
    """
    Expects JSON:
    {
        "your_rating": 1500,
        "opponent_rating": 1600,
        "outcome": "win",           // "win" | "draw" | "loss"
        "games_played": 30,         // optional, default 30
        "age": 17                   // optional; under-18 forces K=40
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    errors = []

    your_rating, err = validate_float(data.get("your_rating"), "your_rating", 100, 3500)
    if err: errors.append(err)

    opp_rating, err = validate_float(data.get("opponent_rating"), "opponent_rating", 100, 3500)
    if err: errors.append(err)

    outcome = data.get("outcome", "").strip().lower()
    if outcome not in ("win", "draw", "loss"):
        errors.append("'outcome' must be 'win', 'draw', or 'loss'.")

    games_played = data.get("games_played", 30)
    games_played, err = validate_int(games_played, "games_played", 0, 100000)
    if err: errors.append(err)

    age = data.get("age", None)
    if age is not None:
        age, err = validate_int(age, "age", 5, 120)
        if err: errors.append(err)

    if errors:
        return jsonify({"errors": errors}), 422

    result = compute_rating_change(your_rating, opp_rating, outcome, games_played, age)
    return jsonify(result)


# ---------------------------------------------------------------------------
# POST /api/elo/session  — multi-game rating session
# ---------------------------------------------------------------------------

@app.route("/api/elo/session", methods=["POST"])
def api_elo_session():
    """
    Expects JSON:
    {
        "your_rating": 1500,
        "games_played": 30,
        "age": 17,                  // optional; under-18 forces K=40
        "games": [
            {"opponent_rating": 1600, "outcome": "win",  "opponent_name": "Alice"},
            {"opponent_rating": 1450, "outcome": "loss", "opponent_name": "Bob"},
            {"opponent_rating": 1550, "outcome": "draw"}
        ]
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    errors = []

    your_rating, err = validate_float(data.get("your_rating"), "your_rating", 100, 3500)
    if err: errors.append(err)

    games_played = data.get("games_played", 30)
    games_played, err = validate_int(games_played, "games_played", 0, 100000)
    if err: errors.append(err)

    age = data.get("age", None)
    if age is not None:
        age, err = validate_int(age, "age", 5, 120)
        if err: errors.append(err)

    games = data.get("games", [])
    if not isinstance(games, list) or len(games) == 0:
        errors.append("'games' must be a non-empty array.")

    if len(games) > 100:
        errors.append("Maximum 100 games per session.")

    if errors:
        return jsonify({"errors": errors}), 422

    current = your_rating
    results = []
    history = [your_rating]
    wins = draws = losses = 0

    for idx, g in enumerate(games, start=1):
        opp_r = g.get("opponent_rating")
        opp_r, err = validate_float(opp_r, f"games[{idx-1}].opponent_rating", 100, 3500)
        if err:
            return jsonify({"error": err}), 422

        oc = g.get("outcome", "").strip().lower()
        if oc not in ("win", "draw", "loss"):
            return jsonify({"error": f"games[{idx-1}].outcome must be 'win', 'draw', or 'loss'."}), 422

        res = compute_rating_change(current, opp_r, oc, games_played + idx - 1, age)
        res["game_number"]   = idx
        res["opponent_name"] = g.get("opponent_name", f"Opponent {idx}")

        s = res["actual_score"]
        if s == 1.0:   wins   += 1
        elif s == 0.5: draws  += 1
        else:          losses += 1

        current = res["new_rating"]
        history.append(current)
        results.append(res)

    return jsonify({
        "starting_rating": your_rating,
        "final_rating":    current,
        "total_change":    round(current - your_rating, 1),
        "age":             age,
        "summary":         {"wins": wins, "draws": draws, "losses": losses},
        "games":           results,
        "rating_history":  history,
    })


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":              "ok",
        "blunder_model_loaded": model is not None,
        "opening_model_loaded": opening_model is not None,
        "stockfish_available":  os.path.exists(ENGINE_PATH),
        "engine_depth":         ENGINE_DEPTH,
        "blunder_threshold":    BLUNDER_THRESHOLD,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
