"""
src/elo.py
----------
Standard FIDE-style ELO rating system.

How ELO works
-------------
Before a game, the expected score (probability of winning) for each player
is calculated from the rating difference.

    E_A = 1 / (1 + 10 ^ ((R_B - R_A) / 400))

After the game, both ratings are updated:

    R_A_new = R_A + K * (S_A - E_A)

Where:
    R_A  = current rating of player A
    R_B  = current rating of player B
    E_A  = expected score of A  (0.0 – 1.0)
    S_A  = actual score: 1 = win, 0.5 = draw, 0 = loss
    K    = K-factor (how much a single game can change your rating)

K-factor rules (FIDE standard)
-------------------------------
    K = 40  → age < 18  OR  games_played < 30
    K = 20  → rating < 2400
    K = 10  → rating >= 2400 (elite)

Outcomes accepted
-----------------
    "win"   → S = 1.0
    "draw"  → S = 0.5
    "loss"  → S = 0.0

Functions
---------
get_k_factor(rating, games_played, age)  → int
expected_score(rating_a, rating_b)       → float
rating_change(rating, opponent_rating, outcome, games_played, age) → dict
simulate_session(player_rating, opponents, games_played, age)      → dict
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# K-factor
# ---------------------------------------------------------------------------

def get_k_factor(rating: float, games_played: int, age: int | None = None) -> int:
    """
    Return the FIDE K-factor for a player.

    Parameters
    ----------
    rating : float
        Current ELO rating.
    games_played : int
        Total number of rated games played so far.
    age : int or None
        Player's age in years. If None, the age condition is not applied.

    Returns
    -------
    int
        K-factor:
            40  — age < 18 OR games_played < 30 (junior / provisional)
            20  — rating < 2400  (standard)
            10  — rating >= 2400 (elite)
    """
    if games_played < 30 or (age is not None and age < 18):
        return 40   # junior or provisional player
    if rating < 2400:
        return 20   # regular player
    return 10       # elite (2400+)


# ---------------------------------------------------------------------------
# Core ELO formulas
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate the expected score (win probability) for player A against player B.

    Parameters
    ----------
    rating_a : float
        ELO rating of player A.
    rating_b : float
        ELO rating of player B.

    Returns
    -------
    float
        Expected score in [0, 1].
        > 0.5 means A is the favourite.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _outcome_to_score(outcome: str) -> float:
    """Convert 'win' / 'draw' / 'loss' to 1.0 / 0.5 / 0.0."""
    outcome = outcome.strip().lower()
    if outcome in ("win", "w", "1"):
        return 1.0
    if outcome in ("draw", "d", "0.5"):
        return 0.5
    if outcome in ("loss", "lose", "l", "0"):
        return 0.0
    raise ValueError(f"Unknown outcome '{outcome}'. Use 'win', 'draw', or 'loss'.")


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def rating_change(
    rating: float,
    opponent_rating: float,
    outcome: str,
    games_played: int = 30,
    age: int | None = None,
) -> dict:
    """
    Calculate the new ELO rating after one game.

    Parameters
    ----------
    rating : float
        Your current ELO rating.
    opponent_rating : float
        Opponent's current ELO rating.
    outcome : str
        Result from your perspective: 'win', 'draw', or 'loss'.
    games_played : int
        Number of rated games you have played so far (affects K-factor).
        Default 30 (standard K=20).
    age : int or None
        Player's age in years. If None, age condition is not applied.
        Players under 18 always receive K=40 regardless of games played.

    Returns
    -------
    dict with keys:
        old_rating        : float       — rating before the game
        new_rating        : float       — rating after the game
        change            : float       — points gained (+) or lost (−)
        expected_score    : float       — probability of winning (0–1)
        actual_score      : float       — 1.0 / 0.5 / 0.0
        k_factor          : int         — K used (10 / 20 / 40)
        k_reason          : str         — why that K was chosen
        win_probability   : str         — human-readable win chance (e.g. "63.2%")
        outcome           : str         — 'win' / 'draw' / 'loss'
        opponent_rating   : float
        rating_diff       : float       — opponent_rating − your_rating (+ = underdog)
        age               : int | None  — player age supplied
        games_played      : int         — games played before this game
    """
    k      = get_k_factor(rating, games_played, age)
    e      = expected_score(rating, opponent_rating)
    s      = _outcome_to_score(outcome)
    delta  = round(k * (s - e), 1)
    new_r  = round(rating + delta, 1)

    # Build a human-readable reason for the K chosen
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
        "win_probability": f"{e * 100:.1f}%",
        "outcome":         outcome.lower(),
        "opponent_rating": opponent_rating,
        "rating_diff":     round(opponent_rating - rating, 1),
        "age":             age,
        "games_played":    games_played,
    }


# ---------------------------------------------------------------------------
# Multi-game session
# ---------------------------------------------------------------------------

def simulate_session(
    player_rating: float,
    opponents: list[dict],
    games_played: int = 30,
    age: int | None = None,
) -> dict:
    """
    Simulate a sequence of games and track how the rating evolves.

    Parameters
    ----------
    player_rating : float
        Starting ELO rating.
    opponents : list of dict
        Each dict must have:
            'opponent_rating' : float
            'outcome'         : str  ('win' / 'draw' / 'loss')
        Optionally:
            'opponent_name'   : str  (label for display)
    games_played : int
        Rated games played *before* this session (affects K-factor).
    age : int or None
        Player's age in years.  Under-18 players always receive K=40.

    Returns
    -------
    dict with keys:
        starting_rating  : float
        final_rating     : float
        total_change     : float
        age              : int | None
        games            : list[dict]   — per-game breakdown (includes k_reason)
        summary          : dict         — wins / draws / losses / net
        rating_history   : list[float]  — rating after each game (includes start)

    Example
    -------
    >>> result = simulate_session(
    ...     player_rating=1500,
    ...     age=17,
    ...     opponents=[
    ...         {"opponent_rating": 1600, "outcome": "win",  "opponent_name": "Alice"},
    ...         {"opponent_rating": 1450, "outcome": "loss", "opponent_name": "Bob"},
    ...         {"opponent_rating": 1550, "outcome": "draw", "opponent_name": "Carol"},
    ...     ]
    ... )
    >>> print(result["final_rating"])
    """
    current_rating = player_rating
    game_records   = []
    rating_history = [player_rating]
    wins = draws = losses = 0

    for idx, opp in enumerate(opponents, start=1):
        result = rating_change(
            rating=current_rating,
            opponent_rating=opp["opponent_rating"],
            outcome=opp["outcome"],
            games_played=games_played + idx - 1,
            age=age,
        )
        result["game_number"]    = idx
        result["opponent_name"]  = opp.get("opponent_name", f"Opponent {idx}")

        if result["actual_score"] == 1.0:
            wins   += 1
        elif result["actual_score"] == 0.5:
            draws  += 1
        else:
            losses += 1

        current_rating = result["new_rating"]
        rating_history.append(current_rating)
        game_records.append(result)

    return {
        "starting_rating": player_rating,
        "final_rating":    current_rating,
        "total_change":    round(current_rating - player_rating, 1),
        "age":             age,
        "games":           game_records,
        "summary": {
            "wins":   wins,
            "draws":  draws,
            "losses": losses,
            "net":    round(current_rating - player_rating, 1),
        },
        "rating_history": rating_history,
    }


# ---------------------------------------------------------------------------
# Convenience: what do I need to reach a target rating?
# ---------------------------------------------------------------------------

def games_to_target(
    current_rating: float,
    target_rating: float,
    opponent_rating: float,
    games_played: int = 30,
    age: int | None = None,
) -> dict:
    """
    Estimate how many consecutive wins are needed to reach a target rating,
    assuming all opponents have the same rating.

    Parameters
    ----------
    current_rating  : float
    target_rating   : float
    opponent_rating : float
    games_played    : int
    age             : int or None
        Player's age. Under-18 players always use K=40.

    Returns
    -------
    dict with keys:
        games_needed   : int
        final_rating   : float
        rating_per_win : float  — average gain per win
        age            : int | None
    """
    r     = current_rating
    count = 0
    while r < target_rating and count < 10_000:
        res = rating_change(r, opponent_rating, "win", games_played + count, age)
        r   = res["new_rating"]
        count += 1

    return {
        "games_needed":    count,
        "final_rating":    round(r, 1),
        "rating_per_win":  round((r - current_rating) / count, 1) if count else 0,
        "age":             age,
    }
