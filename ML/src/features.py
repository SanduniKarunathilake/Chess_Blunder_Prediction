"""
src/features.py
---------------
FEN string → numeric features

Functions
---------
extract_fen_features(fen) -> tuple[int, int, int, int, int]
    Returns (material_diff, total_pieces, in_check, move_number, game_phase)
    for a given FEN string.
"""

import chess


# Piece material values in centipawns
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,    # King is not counted in material
}


def extract_fen_features(fen: str) -> tuple[int, int, int, int, int]:
    """
    Parse a FEN string and return five numeric features.

    Parameters
    ----------
    fen : str
        A valid FEN string describing the board position.

    Returns
    -------
    material_diff : int
        White material minus black material, in centipawns.
        Positive = white is ahead, negative = black is ahead.
    total_pieces : int
        Total number of pieces on the board (both sides, including kings).
    in_check : int
        1 if the side to move is currently in check, else 0.
    move_number : int
        Full move number extracted from the FEN.
    game_phase : int
        0 = opening (≥28 pieces), 1 = middlegame (14-27), 2 = endgame (<14).
    """
    board = chess.Board(fen)

    white_material = sum(
        PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.color == chess.WHITE
    )
    black_material = sum(
        PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.color == chess.BLACK
    )

    material_diff = white_material - black_material
    total_pieces  = len(board.piece_map())
    in_check      = int(board.is_check())
    move_number   = board.fullmove_number

    # Game phase heuristic based on piece count
    if total_pieces >= 28:
        game_phase = 0   # opening
    elif total_pieces >= 14:
        game_phase = 1   # middlegame
    else:
        game_phase = 2   # endgame

    return material_diff, total_pieces, in_check, move_number, game_phase
