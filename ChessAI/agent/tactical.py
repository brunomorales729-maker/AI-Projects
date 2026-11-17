"""Tactical analysis for the chess engine.

Provides:
- Material evaluation.
- Static Exchange Evaluation (SEE).
- Best immediate capture evaluation.
- Post-move tactical safety evaluation.
- Attacked-piece reporting for both colors.
- Bounded quiescence search.

All scores are in pawn units and use the side-to-move perspective
when evaluating a position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 20.0,
}

MATE_SCORE = 100_000.0
TACTICAL_MATE_BONUS = 20.0


@dataclass(frozen=True)
class PieceThreat:
    square: str
    piece: str
    color: str
    attacked: bool
    defended: bool
    attacker_count: int
    defender_count: int
    hanging: bool
    least_attacker_value: Optional[float]


@dataclass(frozen=True)
class TacticalReport:
    side_to_move: str
    legal_capture_count: int
    captures: tuple[str, ...]
    threatened_pieces: tuple[PieceThreat, ...]


class TacticalAnalyzer:
    """Tactical evaluator with bounded exchange and quiescence searches."""

    def __init__(self, quiescence_depth: int = 4):
        if quiescence_depth < 0:
            raise ValueError("quiescence_depth must be >= 0")

        self.quiescence_depth = int(quiescence_depth)
        self.nodes_searched = 0

    @staticmethod
    def material_balance(
        board: chess.Board,
        perspective: Optional[chess.Color] = None,
    ) -> float:
        """Material balance in pawn units."""

        if perspective is None:
            perspective = board.turn

        score = 0.0

        for piece in board.piece_map().values():
            value = PIECE_VALUES[piece.piece_type]

            if piece.color == perspective:
                score += value
            else:
                score -= value

        return score

    @staticmethod
    def _piece_name(piece: chess.Piece) -> str:
        return chess.piece_name(piece.piece_type).capitalize()

    @staticmethod
    def _color_name(color: chess.Color) -> str:
        return "white" if color == chess.WHITE else "black"

    @staticmethod
    def _capture_value(board: chess.Board, move: chess.Move) -> float:
        """Material value gained immediately by a capture or promotion."""

        victim = board.piece_at(move.to_square)

        if board.is_en_passant(move):
            victim_value = PIECE_VALUES[chess.PAWN]
        else:
            victim_value = (
                PIECE_VALUES[victim.piece_type] if victim else 0.0
            )

        if move.promotion:
            victim_value += (
                PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]
            )

        return victim_value

    def _see_reply(
        self,
        board: chess.Board,
        target: chess.Square,
        depth: int,
    ) -> float:
        """Best net gain from legal recaptures on a target square."""

        if depth <= 0 or board.is_game_over(claim_draw=True):
            return 0.0

        # The opponent can always decline to recapture.
        best_gain = 0.0

        candidates = [
            move
            for move in board.legal_moves
            if move.to_square == target and board.is_capture(move)
        ]

        candidates.sort(
            key=lambda move: self._capture_value(board, move),
            reverse=True,
        )

        for move in candidates:
            gain = self._capture_value(board, move)

            board.push(move)
            try:
                net_gain = gain - self._see_reply(
                    board,
                    target,
                    depth - 1,
                )
            finally:
                board.pop()

            best_gain = max(best_gain, net_gain)

        return best_gain

    def static_exchange(
        self,
        board: chess.Board,
        move: chess.Move,
        max_plies: int = 8,
    ) -> float:
        """Estimate the net material result of a capture or promotion.

        Positive: favorable for the mover.
        Negative: unfavorable for the mover.
        """

        if move not in board.legal_moves:
            raise ValueError(f"Move is not legal: {move}")

        if max_plies < 1:
            raise ValueError("max_plies must be >= 1")

        if not board.is_capture(move) and move.promotion is None:
            return 0.0

        immediate_gain = self._capture_value(board, move)
        target = move.to_square

        board.push(move)
        try:
            reply_gain = self._see_reply(
                board,
                target,
                max_plies - 1,
            )
        finally:
            board.pop()

        return immediate_gain - reply_gain

    def best_capture_gain(
        self,
        board: chess.Board,
        max_candidates: Optional[int] = 8,
        exclude_target: Optional[chess.Square] = None,
        max_plies: int = 6,
    ) -> float:
        """Best profitable immediate capture available to side to move.

        exclude_target is useful after a capture: the recapture on that
        square has already been included in the original move's SEE.
        """

        if board.is_game_over(claim_draw=True):
            return 0.0

        if max_candidates is not None and max_candidates < 1:
            raise ValueError("max_candidates must be >= 1 or None")

        candidates = [
            move
            for move in board.legal_moves
            if (
                board.is_capture(move)
                or move.promotion is not None
            )
            and move.to_square != exclude_target
        ]

        candidates.sort(
            key=lambda move: self._capture_value(board, move),
            reverse=True,
        )

        if max_candidates is not None:
            candidates = candidates[:max_candidates]

        best_gain = 0.0

        for move in candidates:
            gain = self.static_exchange(
                board,
                move,
                max_plies=max_plies,
            )
            best_gain = max(best_gain, gain)

        return best_gain

    def move_score(
        self,
        board: chess.Board,
        move: chess.Move,
        risk_weight: float = 1.0,
        max_reply_candidates: int = 8,
    ) -> float:
        """Evaluate a candidate move from its mover's perspective.

        Rewards:
        - Profitable captures.
        - Profitable promotions.
        - Checkmate.

        Penalizes:
        - Profitable immediate captures left available to the opponent.

        The opponent's recapture on the destination square is excluded
        from the second calculation because SEE already accounts for it.
        """

        if move not in board.legal_moves:
            raise ValueError(f"Move is not legal: {move}")

        if risk_weight < 0:
            raise ValueError("risk_weight must be non-negative")

        if max_reply_candidates < 1:
            raise ValueError("max_reply_candidates must be >= 1")

        is_capture = board.is_capture(move)
        destination = move.to_square

        immediate_gain = 0.0

        if is_capture or move.promotion is not None:
            immediate_gain = self.static_exchange(
                board,
                move,
                max_plies=6,
            )

        board.push(move)

        try:
            if board.is_checkmate():
                return TACTICAL_MATE_BONUS + max(0.0, immediate_gain)

            excluded_square = destination if is_capture else None

            opponent_best_gain = self.best_capture_gain(
                board,
                max_candidates=max_reply_candidates,
                exclude_target=excluded_square,
                max_plies=4,
            )
        finally:
            board.pop()

        return immediate_gain - risk_weight * opponent_best_gain

    def analyze(self, board: chess.Board) -> TacticalReport:
        """Report captures and attacked pieces belonging to both sides."""

        us = board.turn

        legal_captures = [
            move
            for move in board.legal_moves
            if board.is_capture(move) or move.promotion is not None
        ]

        capture_scores = [
            (move, self.static_exchange(board, move))
            for move in legal_captures
        ]

        capture_scores.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        captures = tuple(
            f"{board.uci(move)} (SEE {score:+.2f})"
            for move, score in capture_scores
        )

        threats = []

        for square, piece in board.piece_map().items():
            enemy = not piece.color

            attacked_by = list(board.attackers(enemy, square))

            if not attacked_by:
                continue

            defenders = list(board.attackers(piece.color, square))

            attacker_values = [
                PIECE_VALUES[attacker.piece_type]
                for attacker_sq in attacked_by
                if (attacker := board.piece_at(attacker_sq)) is not None
            ]

            threats.append(
                PieceThreat(
                    square=chess.square_name(square),
                    piece=self._piece_name(piece),
                    color=self._color_name(piece.color),
                    attacked=True,
                    defended=bool(defenders),
                    attacker_count=len(attacked_by),
                    defender_count=len(defenders),
                    # This is a simple diagnostic indicator, not proof
                    # that the piece will be lost in a legal exchange.
                    hanging=(
                        piece.piece_type != chess.KING
                        and not defenders
                    ),
                    least_attacker_value=(
                        min(attacker_values) if attacker_values else None
                    ),
                )
            )

        piece_values_by_name = {
            "Pawn": PIECE_VALUES[chess.PAWN],
            "Knight": PIECE_VALUES[chess.KNIGHT],
            "Bishop": PIECE_VALUES[chess.BISHOP],
            "Rook": PIECE_VALUES[chess.ROOK],
            "Queen": PIECE_VALUES[chess.QUEEN],
            "King": PIECE_VALUES[chess.KING],
        }

        threats.sort(
            key=lambda item: (
                item.hanging,
                piece_values_by_name.get(item.piece, 0.0),
            ),
            reverse=True,
        )

        return TacticalReport(
            side_to_move=self._color_name(us),
            legal_capture_count=len(legal_captures),
            captures=captures,
            threatened_pieces=tuple(threats),
        )

    def _quiescence(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        depth: int,
        ply: int,
    ) -> float:
        self.nodes_searched += 1

        if board.is_checkmate():
            return -MATE_SCORE + ply

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ):
            return 0.0

        in_check = board.is_check()

        if not in_check:
            stand_pat = self.material_balance(board, board.turn)

            if stand_pat >= beta:
                return beta

            alpha = max(alpha, stand_pat)

            if depth <= 0:
                return stand_pat

            moves = [
                move
                for move in board.legal_moves
                if board.is_capture(move) or move.promotion is not None
            ]

        else:
            # Stand-pat is invalid while in check.
            if depth <= 0:
                evasions = list(board.legal_moves)

                if not evasions:
                    return -MATE_SCORE + ply

                best = -MATE_SCORE

                for move in evasions:
                    board.push(move)
                    try:
                        if board.is_checkmate():
                            score = -MATE_SCORE + ply + 1
                        else:
                            score = -self.material_balance(
                                board,
                                board.turn,
                            )
                    finally:
                        board.pop()

                    best = max(best, score)

                return best

            moves = list(board.legal_moves)

        moves.sort(
            key=lambda move: self._capture_value(board, move),
            reverse=True,
        )

        if not moves:
            return self.material_balance(board, board.turn)

        for move in moves:
            board.push(move)
            try:
                score = -self._quiescence(
                    board,
                    -beta,
                    -alpha,
                    depth - 1,
                    ply + 1,
                )
            finally:
                board.pop()

            if score >= beta:
                return beta

            alpha = max(alpha, score)

        return alpha

    def quiescence(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
    ) -> float:
        """Return a tactical score in pawn units from side-to-move POV."""

        search_depth = (
            self.quiescence_depth if depth is None else int(depth)
        )

        if search_depth < 0:
            raise ValueError("depth must be >= 0")

        self.nodes_searched = 0

        return self._quiescence(
            board,
            -MATE_SCORE,
            MATE_SCORE,
            search_depth,
            0,
        )

    def evaluate(self, board: chess.Board) -> dict:
        """Return a serializable tactical report."""

        report = self.analyze(board)
        score = self.quiescence(board)

        return {
            "side_to_move": report.side_to_move,
            "material_balance": self.material_balance(board),
            "quiescence_score": score,
            "quiescence_depth": self.quiescence_depth,
            "nodes_searched": self.nodes_searched,
            "best_capture_gain": self.best_capture_gain(board),
            "legal_capture_count": report.legal_capture_count,
            "captures": list(report.captures),
            "threatened_pieces": [
                {
                    "square": item.square,
                    "piece": item.piece,
                    "color": item.color,
                    "attacked": item.attacked,
                    "defended": item.defended,
                    "attacker_count": item.attacker_count,
                    "defender_count": item.defender_count,
                    "hanging": item.hanging,
                    "least_attacker_value": item.least_attacker_value,
                }
                for item in report.threatened_pieces
            ],
        }
