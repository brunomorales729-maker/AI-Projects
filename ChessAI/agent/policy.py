"""Neural inference. Stockfish is intentionally not imported or called here."""
from __future__ import annotations

from pathlib import Path

import torch

from agent.encoding import encode_board, decode_action, create_legal_action_mask
from agent.network import ChessResNet
from config import CHECKPOINT_PATH, DEVICE


class ChessPolicy:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH, device=None):
        self.device = torch.device(device or DEVICE)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        self.model = ChessResNet().to(self.device)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:  # compatibility with older PyTorch versions
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.checkpoint_path = checkpoint_path

    @torch.inference_mode()
    def predict(self, board, temperature: float = 0.0):
        if board.is_game_over(claim_draw=True):
            raise ValueError("The game is over; there are no legal moves.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        state = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        legal_mask = create_legal_action_mask(board, device=self.device)
        logits = logits.squeeze(0).masked_fill(~legal_mask, float("-inf"))
        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=0)
            action = torch.multinomial(probabilities, 1).item()
        else:
            probabilities = torch.softmax(logits, dim=0)
            action = int(torch.argmax(logits).item())
        move = decode_action(action, board.turn)
        if move not in board.legal_moves:
            raise RuntimeError(f"Model returned illegal move {move.uci()}")
        return move, float(value.item()), probabilities

    def select_move(self, board, temperature: float = 0.0):
        move, value, _ = self.predict(board, temperature)
        return move, value
