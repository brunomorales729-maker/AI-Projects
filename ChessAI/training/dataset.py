"""Compact JSONL dataset reader; only the current definitive dataset is loaded."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from agent.encoding import encode_fen, encode_uci
from config import DATASET_PATH, POLICY_TEMPERATURE_CP, VALUE_SCALE_CP


class ChessDataset(Dataset):
    def __init__(self, path=DATASET_PATH, policy_temperature_cp=POLICY_TEMPERATURE_CP, value_scale_cp=VALUE_SCALE_CP):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        self.sample_count = 0
        self.states = []
        self.action_indices = []
        self.policy_targets = []
        self.values = []
        self.stages = []
        self.position_keys = []
        self.group_ids = []
        self.turns = []

        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    fen_fields = record["fen"].split()
                    if len(fen_fields) != 6:
                        raise ValueError("FEN must have six fields")
                    fen_turn = "white" if fen_fields[1] == "w" else "black"
                    if record.get("turn", fen_turn) != fen_turn:
                        raise ValueError("turn field disagrees with FEN")
                    candidates = record["candidates"]
                    if not isinstance(candidates, list) or not candidates:
                        raise ValueError("position has no Stockfish candidates")
                    # Stable score ordering: all scores are side-to-move POV.
                    candidates = sorted(candidates, key=lambda item: (int(item["score_cp"]), int(item.get("depth", 0))), reverse=True)
                    state = encode_fen(record["fen"])
                    action_ids, scores = [], []
                    seen_moves = set()
                    for candidate in candidates:
                        move_uci = candidate["move"]
                        if move_uci in seen_moves:
                            continue
                        seen_moves.add(move_uci)
                        action_ids.append(encode_uci(move_uci, fen_fields[1]))
                        scores.append(int(candidate["score_cp"]))
                    if not action_ids:
                        raise ValueError("no valid candidate actions")
                    scores_array = np.asarray(scores, dtype=np.float32)
                    weights = np.exp(np.clip((scores_array - scores_array.max()) / policy_temperature_cp, -30.0, 0.0))
                    weights /= weights.sum()
                    value = float(np.tanh(scores_array[0] / value_scale_cp))
                    self.sample_count += 1
                    self.states.append(state)
                    self.action_indices.append(np.asarray(action_ids, dtype=np.int64))
                    self.policy_targets.append(weights.astype(np.float32))
                    self.values.append(value)
                    self.stages.append(record.get("stage", "middlegame"))
                    # Fullmove number is irrelevant, but halfmove clock affects the 50-move rule
                    # and is encoded by the network, so it remains part of the rule-state key.
                    position_key = " ".join(fen_fields[:5])
                    self.position_keys.append(position_key)
                    game_id = record.get("game_id")
                    if game_id is None:
                        # Legacy fallback: prevent exact position/transposition leakage.
                        group_id = f"position:{position_key}"
                    else:
                        # game_id values are globally unique within the definitive dataset;
                        # do not prefix with source, because one transposed game component
                        # can contain records originally emitted by different source files.
                        group_id = str(game_id)
                    self.group_ids.append(group_id)
                    self.turns.append(fen_turn)
                except Exception as exc:
                    raise ValueError(f"Invalid dataset line {line_number}: {exc}") from exc

        if self.sample_count == 0:
            raise ValueError(f"Dataset is empty: {self.path}")
        self.states = np.stack(self.states).astype(np.float32, copy=False)
        self.values = np.asarray(self.values, dtype=np.float32)

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.states[index]),
            torch.from_numpy(self.action_indices[index]),
            torch.from_numpy(self.policy_targets[index]),
            torch.tensor(self.values[index], dtype=torch.float32),
        )


def collate_chess_batch(samples):
    states, actions, weights, values = zip(*samples)
    max_candidates = max(len(item) for item in actions)
    padded_actions = torch.zeros((len(samples), max_candidates), dtype=torch.long)
    padded_weights = torch.zeros((len(samples), max_candidates), dtype=torch.float32)
    for row, (action_row, weight_row) in enumerate(zip(actions, weights)):
        count = len(action_row)
        padded_actions[row, :count] = action_row
        padded_weights[row, :count] = weight_row
    return torch.stack(states), padded_actions, padded_weights, torch.stack(values)
