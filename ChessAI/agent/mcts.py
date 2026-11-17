"""PUCT search guided by a neural policy/value network and tactical analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from agent.encoding import encode_move
from agent.policy import ChessPolicy
from agent.tactical import TacticalAnalyzer


@dataclass
class Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    tactical_score: float = 0.0
    children: dict = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(
        self,
        policy=None,
        simulations=50,
        c_puct=1.5,
        tactical=True,
        tactical_prior_weight=0.55,
        tactical_value_weight=0.05,
        tactical_value_scale=8.0,
        tactical_quiescence_depth=2,
        tactical_reply_candidates=8,
        tactical_max_prior_score=6.0,
        tactical_risk_weight=1.0,
    ):
        if simulations < 1:
            raise ValueError("MCTS simulations must be >= 1")

        if c_puct <= 0:
            raise ValueError("c_puct must be positive")

        if tactical_prior_weight < 0:
            raise ValueError("tactical_prior_weight must be >= 0")

        if not 0.0 <= tactical_value_weight <= 1.0:
            raise ValueError("tactical_value_weight must be between 0 and 1")

        if tactical_value_scale <= 0:
            raise ValueError("tactical_value_scale must be positive")

        if tactical_quiescence_depth < 0:
            raise ValueError("tactical_quiescence_depth must be >= 0")

        if tactical_reply_candidates < 1:
            raise ValueError("tactical_reply_candidates must be >= 1")

        if tactical_max_prior_score <= 0:
            raise ValueError("tactical_max_prior_score must be positive")

        if tactical_risk_weight < 0:
            raise ValueError("tactical_risk_weight must be >= 0")

        self.policy = policy or ChessPolicy()
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)

        self.tactical_enabled = bool(tactical)

        self.tactical_prior_weight = float(tactical_prior_weight)
        self.tactical_value_weight = float(tactical_value_weight)
        self.tactical_value_scale = float(tactical_value_scale)
        self.tactical_reply_candidates = int(tactical_reply_candidates)
        self.tactical_max_prior_score = float(tactical_max_prior_score)
        self.tactical_risk_weight = float(tactical_risk_weight)

        self.tactical_analyzer = (
            TacticalAnalyzer(
                quiescence_depth=tactical_quiescence_depth
            )
            if self.tactical_enabled
            else None
        )

        self.last_root_stats = []
        self.last_chosen_move = None

    @staticmethod
    def _terminal_value(board) -> float:
        outcome = board.outcome(claim_draw=True)

        if outcome is None or outcome.winner is None:
            return 0.0

        # The side to move has lost in a checkmate position.
        return -1.0 if board.is_checkmate() else 0.0

    def _expand(self, node, board) -> float:
        """Expand a node using neural priors adjusted by tactical scores."""

        _move, neural_value, probabilities = self.policy.predict(board)

        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return self._terminal_value(board)

        log_weights = []
        tactical_scores = {}

        for move in legal_moves:
            action_index = encode_move(move, board.turn)

            base_prior = max(
                float(probabilities[action_index].item()),
                1e-12,
            )

            tactical_score = 0.0

            if self.tactical_enabled:
                tactical_score = self.tactical_analyzer.move_score(
                    board,
                    move,
                    risk_weight=self.tactical_risk_weight,
                    max_reply_candidates=self.tactical_reply_candidates,
                )

            tactical_scores[move] = tactical_score

            # Clip extreme values so tactical analysis cannot completely
            # overwhelm the neural policy.
            clipped_score = max(
                -self.tactical_max_prior_score,
                min(self.tactical_max_prior_score, tactical_score),
            )

            adjusted_log_prior = (
                math.log(base_prior)
                + self.tactical_prior_weight * clipped_score
            )

            log_weights.append(adjusted_log_prior)

        # Numerically stable softmax over legal moves.
        max_log_weight = max(log_weights)

        weights = [
            math.exp(weight - max_log_weight)
            for weight in log_weights
        ]

        total_weight = sum(weights)

        if total_weight <= 0:
            priors = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            priors = [
                weight / total_weight
                for weight in weights
            ]

        node.children.clear()

        for move, prior in zip(legal_moves, priors):
            node.children[move] = Node(
                prior=float(prior),
                tactical_score=float(tactical_scores[move]),
            )

        value = float(neural_value)

        # Blend a small tactical material signal into the neural value.
        # Both values use the side-to-move perspective.
        if (
            self.tactical_enabled
            and self.tactical_value_weight > 0
        ):
            tactical_score = self.tactical_analyzer.quiescence(
                board,
                depth=self.tactical_analyzer.quiescence_depth,
            )

            tactical_value = math.tanh(
                tactical_score / self.tactical_value_scale
            )

            weight = self.tactical_value_weight

            value = (
                (1.0 - weight) * value
                + weight * tactical_value
            )

        return max(-1.0, min(1.0, value))

    def _select_child(self, node):
        best_score = -float("inf")
        best_move = None
        best_child = None

        root_visits = math.sqrt(max(1, node.visits))

        for move, child in node.children.items():
            # Child values use the child's side-to-move perspective.
            q = -child.q_value

            u = (
                self.c_puct
                * child.prior
                * root_visits
                / (1 + child.visits)
            )

            score = q + u

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _simulate(self, root, board):
        node = root
        path = [root]

        while node.children and not board.is_game_over(claim_draw=True):
            move, child = self._select_child(node)

            board.push(move)

            node = child
            path.append(node)

        if board.is_game_over(claim_draw=True):
            value = self._terminal_value(board)
        else:
            value = self._expand(node, board)

        # Back up values while alternating player perspective.
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value

    def search(self, board, simulations=None, temperature=0.0):
        if board.is_game_over(claim_draw=True):
            raise ValueError("Cannot search a terminal position")

        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        count = (
            self.simulations
            if simulations is None
            else int(simulations)
        )

        if count < 1:
            raise ValueError("simulations must be >= 1")

        root = Node(prior=1.0)

        self._expand(root, board)

        for _ in range(count):
            self._simulate(
                root,
                board.copy(stack=True),
            )

        moves = list(root.children)

        visits = np.asarray(
            [root.children[move].visits for move in moves],
            dtype=np.float64,
        )

        if temperature == 0:
            selected = int(np.argmax(visits))
        else:
            weights = visits ** (1.0 / temperature)

            if weights.sum() <= 0:
                weights = np.ones_like(weights)

            selected = int(
                np.random.choice(
                    len(moves),
                    p=weights / weights.sum(),
                )
            )

        chosen = moves[selected]

        if chosen not in board.legal_moves:
            raise RuntimeError("PUCT returned an illegal move")

        stats = [
            {
                "move": move.uci(),
                "visits": child.visits,
                "q_value": -child.q_value,
                "prior": child.prior,
                "tactical_score": child.tactical_score,
            }
            for move, child in root.children.items()
        ]

        stats.sort(
            key=lambda item: (
                item["visits"],
                item["prior"],
            ),
            reverse=True,
        )

        self.last_root_stats = stats
        self.last_chosen_move = chosen

        return chosen, stats
