"""Board and move encoding shared by training and inference.

The board is always encoded from the side-to-move perspective. For Black,
rank order is flipped; files are deliberately not mirrored, so kingside and
queenside retain their meaning. This keeps the representation symmetric
between colors and preserves castling semantics.
"""
from __future__ import annotations

import numpy as np

INPUT_CHANNELS = 18
BOARD_SIZE = 8
PROMOTION_TYPES = 5
ACTION_SIZE = 64 * 64 * PROMOTION_TYPES

_PROMOTION_TO_INDEX = {None: 0, "q": 1, "r": 2, "b": 3, "n": 4}
_INDEX_TO_PROMOTION = {value: key for key, value in _PROMOTION_TO_INDEX.items()}
_PIECE_TO_TYPE = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}


def _square_from_coord(coord: str) -> int:
    if len(coord) != 2 or coord[0] not in "abcdefgh" or coord[1] not in "12345678":
        raise ValueError(f"Invalid UCI square: {coord!r}")
    return (ord(coord[0]) - ord("a")) + 8 * (int(coord[1]) - 1)


def _canonical_square(square: int, turn: str) -> int:
    return square ^ 56 if turn == "b" else square


def encode_uci(move_uci: str, turn: str) -> int:
    """Map UCI move text to the fixed 20,480-action policy space."""
    if turn not in ("w", "b", "white", "black"):
        raise ValueError(f"Invalid side to move: {turn!r}")
    turn = "w" if turn in ("w", "white") else "b"
    if len(move_uci) not in (4, 5):
        raise ValueError(f"Invalid UCI move: {move_uci!r}")
    from_square = _canonical_square(_square_from_coord(move_uci[:2]), turn)
    to_square = _canonical_square(_square_from_coord(move_uci[2:4]), turn)
    promotion = move_uci[4].lower() if len(move_uci) == 5 else None
    if promotion not in _PROMOTION_TO_INDEX:
        raise ValueError(f"Unsupported promotion in {move_uci!r}")
    return ((from_square * 64 + to_square) * PROMOTION_TYPES) + _PROMOTION_TO_INDEX[promotion]


def encode_fen(fen: str) -> np.ndarray:
    """Encode a standard six-field FEN into an (18, 8, 8) float32 tensor."""
    fields = fen.strip().split()
    if len(fields) != 6:
        raise ValueError(f"FEN must have 6 fields, got {len(fields)}: {fen!r}")
    placement, turn, castling, ep_square, halfmove, _fullmove = fields
    if turn not in ("w", "b"):
        raise ValueError(f"Invalid FEN turn: {turn!r}")
    if castling != "-" and (any(token not in "KQkq" for token in castling) or len(set(castling)) != len(castling)):
        raise ValueError(f"Invalid FEN castling field: {castling!r}")
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError("FEN piece placement must contain 8 ranks")

    planes = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for source_row, rank_text in enumerate(ranks):
        col = 0
        for symbol in rank_text:
            if symbol.isdigit():
                col += int(symbol)
                continue
            piece_type = _PIECE_TO_TYPE.get(symbol.lower())
            if piece_type is None or col >= 8:
                raise ValueError(f"Invalid FEN rank: {rank_text!r}")
            row = 7 - source_row if turn == "b" else source_row
            channel = (piece_type - 1) if (symbol.isupper() == (turn == "w")) else (6 + piece_type - 1)
            planes[channel, row, col] = 1.0
            col += 1
        if col != 8:
            raise ValueError(f"FEN rank does not describe 8 squares: {rank_text!r}")

    if float(planes[5].sum()) != 1.0 or float(planes[11].sum()) != 1.0:
        raise ValueError("FEN must contain exactly one king for each side")
    if float(planes[0, 0, :].sum() + planes[0, 7, :].sum() + planes[6, 0, :].sum() + planes[6, 7, :].sum()) > 0:
        raise ValueError("FEN cannot place pawns on the first or eighth rank")

    # Castling planes: own K-side, own Q-side, opponent K-side, opponent Q-side.
    rights = set() if castling == "-" else set(castling)
    own_short, own_long, opp_short, opp_long = (
        ("K", "Q", "k", "q") if turn == "w" else ("k", "q", "K", "Q")
    )
    for channel, token in enumerate((own_short, own_long, opp_short, opp_long), start=12):
        if token in rights:
            planes[channel, :, :] = 1.0

    if ep_square != "-":
        square = _square_from_coord(ep_square)
        file_index = square % 8
        rank_index = square // 8
        source_row = 7 - rank_index  # FEN rank 8 is screen row 0.
        row = 7 - source_row if turn == "b" else source_row
        planes[16, row, file_index] = 1.0

    try:
        halfmove_count = int(halfmove)
        fullmove_count = int(_fullmove)
        if halfmove_count < 0 or fullmove_count < 1:
            raise ValueError
        halfmove_value = min(halfmove_count, 100) / 100.0
    except ValueError as exc:
        raise ValueError("Invalid FEN move counters") from exc
    planes[17, :, :] = halfmove_value
    return planes


def encode_board(board) -> np.ndarray:
    """Encode a python-chess Board without making python-chess a train-time dependency."""
    return encode_fen(board.fen())


def encode_move(move, turn) -> int:
    """Encode a python-chess Move."""
    turn_text = "w" if turn in (True, "w", "white") else "b"
    return encode_uci(move.uci(), turn_text)


def decode_action(action_index: int, turn):
    """Decode an action into python-chess Move; python-chess is needed only here."""
    if not 0 <= int(action_index) < ACTION_SIZE:
        raise ValueError("Action index outside policy space")
    try:
        import chess
    except ImportError as exc:  # pragma: no cover - installation guidance
        raise RuntimeError("Install python-chess to decode moves: pip install python-chess") from exc
    promotion_index = int(action_index) % PROMOTION_TYPES
    pair_index = int(action_index) // PROMOTION_TYPES
    from_square = pair_index // 64
    to_square = pair_index % 64
    is_black = turn in (False, "b", "black")
    if is_black:
        from_square ^= 56
        to_square ^= 56
    promotion_name = _INDEX_TO_PROMOTION[promotion_index]
    promotion_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
    return chess.Move(from_square, to_square, promotion=promotion_map.get(promotion_name))


def legal_action_indices(board) -> list[int]:
    return [encode_move(move, board.turn) for move in board.legal_moves]


def create_legal_action_mask(board, device=None):
    import torch
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool, device=device)
    indices = legal_action_indices(board)
    if indices:
        mask[indices] = True
    return mask
