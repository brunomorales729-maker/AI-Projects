"""Validate schema, FENs, candidate legality, duplicates and Stockfish score order."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import DATASET_PATH


def main():
    parser = argparse.ArgumentParser(description="Validate the definitive chess dataset")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    try:
        import chess
        full_chess_validation = True
    except ImportError:
        chess = None
        full_chess_validation = False

    stages, turns, candidate_counts, depths = Counter(), Counter(), Counter(), []
    duplicate_keys = set()
    game_groups = set()
    duplicates = malformed = illegal_positions = illegal_candidates = turn_mismatch = order_errors = 0
    missing_game_ids = ply_mismatches = 0
    score_min, score_max = float("inf"), -float("inf")
    samples = 0

    with args.dataset.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                fields = record["fen"].split()
                if len(fields) != 6 or fields[1] not in ("w", "b"):
                    raise ValueError("invalid six-field FEN")
                turn = "white" if fields[1] == "w" else "black"
                if record.get("turn") != turn:
                    turn_mismatch += 1
                game_id = record.get("game_id")
                if not isinstance(game_id, str) or not game_id.strip():
                    missing_game_ids += 1
                else:
                    game_groups.add(game_id)
                if "ply" in record:
                    expected_ply = 2 * (int(fields[5]) - 1) + (0 if fields[1] == "w" else 1)
                    if int(record["ply"]) != expected_ply:
                        ply_mismatches += 1
                else:
                    ply_mismatches += 1
                # Keep distinct 50-move clocks; ignore only the non-rule-relevant fullmove counter.
                key = " ".join(fields[:5])
                if key in duplicate_keys:
                    duplicates += 1
                duplicate_keys.add(key)
                stage = record.get("stage")
                if stage not in ("opening", "middlegame", "endgame"):
                    raise ValueError(f"invalid stage {stage!r}")
                candidates = record.get("candidates")
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError("missing candidate moves")
                board = None
                if full_chess_validation:
                    board = chess.Board(record["fen"])
                    if not board.is_valid():
                        illegal_positions += 1
                scores = []
                seen_moves = set()
                for candidate in candidates:
                    move_text = candidate["move"]
                    if len(move_text) not in (4, 5):
                        raise ValueError(f"invalid UCI move {move_text!r}")
                    score = int(candidate["score_cp"])
                    depth = int(candidate.get("depth", 0))
                    if depth < 1:
                        raise ValueError(f"invalid Stockfish depth {depth}")
                    scores.append(score)
                    score_min = min(score_min, score)
                    score_max = max(score_max, score)
                    depths.append(depth)
                    if move_text in seen_moves:
                        malformed += 1
                    seen_moves.add(move_text)
                    if full_chess_validation:
                        try:
                            move = chess.Move.from_uci(move_text)
                            if move not in board.legal_moves:
                                illegal_candidates += 1
                        except ValueError:
                            illegal_candidates += 1
                if scores != sorted(scores, reverse=True):
                    order_errors += 1
                stages[stage] += 1
                turns[turn] += 1
                candidate_counts[len(candidates)] += 1
                samples += 1
            except Exception as exc:
                malformed += 1
                print(f"Line {line_number}: {exc}")

    print("\n=== DATASET AUDIT ===")
    print(f"File: {args.dataset}")
    print(f"Records: {samples:,}")
    print(f"Full python-chess validation: {'yes' if full_chess_validation else 'NO (schema-only fallback)'}")
    print(f"Stages: {dict(stages)}")
    print(f"Turns: {dict(turns)}")
    print(f"Candidate counts: {dict(candidate_counts)}")
    print(f"Unique game groups: {len(game_groups)}")
    print(f"Missing game IDs: {missing_game_ids}")
    print(f"Ply metadata mismatches/missing: {ply_mismatches}")
    print(f"Duplicate position keys: {duplicates}")
    print(f"Malformed records / duplicate moves: {malformed}")
    print(f"Invalid chess positions: {illegal_positions if full_chess_validation else 'not checked'}")
    print(f"Illegal candidate moves: {illegal_candidates if full_chess_validation else 'not checked'}")
    print(f"Turn mismatches: {turn_mismatch}")
    print(f"Candidate ordering errors: {order_errors}")
    if depths:
        print(f"Depth range: {min(depths)}–{max(depths)} | median {sorted(depths)[len(depths)//2]}")
        print(f"Score range (side-to-move POV): {score_min}–{score_max} cp")

    fatal = malformed or duplicates or turn_mismatch or order_errors or missing_game_ids or ply_mismatches
    if full_chess_validation:
        fatal = fatal or illegal_positions or illegal_candidates
    if not full_chess_validation:
        print("Install requirements.txt and rerun for legal-position and legal-move checks.")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
