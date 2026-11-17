"""Generate one balanced Stockfish MultiPV dataset (one sample per side to move)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine

from config import DATASET_PATH, STOCKFISH_PATH

STAGE_RATIOS = {"opening": 0.20, "middlegame": 0.60, "endgame": 0.20}


def position_key(board: chess.Board) -> str:
    fields = board.fen().split()
    # Halfmove clock changes 50-move draw status; fullmove counter does not affect legal moves.
    return " ".join(fields[:5])


def classify_stage(board: chess.Board) -> str:
    pieces = list(board.piece_map().values())
    non_king = [piece for piece in pieces if piece.piece_type != chess.KING]
    material = sum(
        {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}[p.piece_type]
        for p in non_king
    )
    if board.fullmove_number <= 12 and len(pieces) >= 24:
        return "opening"
    if len(non_king) <= 8 or material <= 13:
        return "endgame"
    return "middlegame"


def randomize_opening(board: chess.Board, rng: random.Random, min_plies: int, max_plies: int):
    for _ in range(rng.randint(min_plies, max_plies)):
        if board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(list(board.legal_moves)))


def analyze(engine, board: chess.Board, multipv: int, seconds: float):
    infos = engine.analyse(board, chess.engine.Limit(time=seconds), multipv=multipv)
    if isinstance(infos, dict):
        infos = [infos]
    candidates = []
    seen = set()
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        if move not in board.legal_moves or move.uci() in seen:
            continue
        score = info["score"].pov(board.turn)
        score_cp = score.score(mate_score=10_000)
        if score_cp is None:
            continue
        candidates.append({
            "move": move.uci(),
            "score_cp": int(score_cp),
            "mate_in": score.mate(),
            "depth": int(info.get("depth", 0) or 0),
        })
        seen.add(move.uci())
    # Scores and the value target always use side-to-move POV.
    candidates.sort(key=lambda item: (item["score_cp"], item["depth"]), reverse=True)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Generate the definitive Stockfish JSONL dataset")
    parser.add_argument("--engine", type=Path, default=STOCKFISH_PATH, help="Stockfish executable")
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument("--positions", type=int, default=10_000)
    parser.add_argument("--max-games", type=int, default=4_000)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--analysis-time", type=float, default=0.06)
    parser.add_argument("--play-time", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-opening-plies", type=int, default=2)
    parser.add_argument("--max-opening-plies", type=int, default=14)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing dataset after successful generation")
    parser.add_argument("--backup-dir", type=Path, default=Path(__file__).resolve().parents[2].parent / "ChessAI_backups", help="External backup directory")
    args = parser.parse_args()

    if args.positions < 1 or args.max_games < 1 or args.multipv < 1 or args.max_plies < 2:
        parser.error("positions, max-games, multipv and max-plies must be positive")
    if args.analysis_time <= 0 or args.play_time <= 0:
        parser.error("analysis-time and play-time must be positive")
    if args.min_opening_plies < 0 or args.max_opening_plies < args.min_opening_plies:
        parser.error("invalid random-opening ply range")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} already exists; use --overwrite to replace it")
    if not args.engine.exists() and args.engine.parent != Path("."):
        parser.error(f"Stockfish executable not found: {args.engine}")

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    quotas = {stage: round(args.positions * ratio) for stage, ratio in STAGE_RATIOS.items()}
    quotas["middlegame"] += args.positions - sum(quotas.values())
    counts = Counter()
    seen = set()
    skipped = 0
    games = 0

    print(f"Stockfish: {args.engine}")
    print(f"Target: {args.positions:,} positions; stage quotas: {quotas}")
    engine = chess.engine.SimpleEngine.popen_uci(str(args.engine))
    try:
        available = engine.options
        settings = {}
        if "Threads" in available:
            settings["Threads"] = args.threads
        if "Hash" in available:
            settings["Hash"] = args.hash_mb
        if settings:
            engine.configure(settings)

        with temporary.open("w", encoding="utf-8") as output:
            while sum(counts.values()) < args.positions and games < args.max_games:
                games += 1
                board = chess.Board()
                randomize_opening(board, rng, args.min_opening_plies, args.max_opening_plies)
                while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
                    stage = classify_stage(board)
                    key = position_key(board)
                    if counts[stage] < quotas[stage] and key not in seen:
                        try:
                            candidates = analyze(engine, board, args.multipv, args.analysis_time)
                            if candidates:
                                record = {
                                    "fen": board.fen(),
                                    "turn": "white" if board.turn == chess.WHITE else "black",
                                    "stage": stage,
                                    "candidates": candidates,
                                    "source": "Stockfish",
                                    "game_id": f"generated_{games:06d}",
                                    "ply": board.ply(),
                                }
                                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                                counts[stage] += 1
                                seen.add(key)
                            else:
                                skipped += 1
                        except (chess.engine.EngineError, ValueError, KeyError):
                            skipped += 1
                    try:
                        result = engine.play(board, chess.engine.Limit(time=args.play_time))
                    except chess.engine.EngineError:
                        break
                    if result.move is None:
                        break
                    board.push(result.move)
                if games % 25 == 0:
                    print(f"Games {games:4d} | total {sum(counts.values()):5,d}/{args.positions:,} | {dict(counts)}")

        total = sum(counts.values())
        if total < args.positions:
            print(f"Warning: reached {total:,}/{args.positions:,} positions after {games} games")
            print("Incomplete output was discarded; existing definitive dataset was not touched.")
            temporary.unlink(missing_ok=True)
            return 2
        # Atomic replacement; historical copy is stored outside the project tree.
        if args.output.exists():
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            backup = args.backup_dir / f"{args.output.stem}_{int(time.time())}.jsonl"
            shutil.copy2(args.output, backup)
            print(f"Previous dataset backup: {backup}")
        temporary.replace(args.output)
        print(f"Generated {total:,} unique positions in {games} games; skipped {skipped}")
        print(f"Stage distribution: {dict(counts)}")
        print(f"Dataset: {args.output}")
        return 0
    finally:
        engine.quit()
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
