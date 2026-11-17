"""Match evaluation against Stockfish skill levels; truncated games are not draws."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.engine

from agent.mcts import MCTS
from agent.policy import ChessPolicy
from config import CHECKPOINT_PATH, STOCKFISH_PATH
from training.generate_dataset import classify_stage


def safe_score_cp(info, pov):
    score = info.get("score")
    if score is None:
        return None
    value = score.pov(pov).score(mate_score=10_000)
    return None if value is None else int(value)


def choose_agent_move(policy, board, simulations):
    if simulations > 0:
        search = MCTS(policy=policy, simulations=simulations)
        return search.search(board, simulations=simulations)[0]
    return policy.select_move(board)[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the neural agent against Stockfish")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None, help="Optional compatible Policy+Value checkpoint for head-to-head comparison")
    parser.add_argument("--stockfish", type=Path, default=STOCKFISH_PATH)
    parser.add_argument("--games-per-level", type=int, default=4)
    parser.add_argument("--skill-levels", default="2,8,14", help="Comma-separated Stockfish Skill Level values (0–20)")
    parser.add_argument("--move-time", type=float, default=0.08)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=0, help="0 uses direct neural policy; >0 uses PUCT")
    parser.add_argument("--quality-depth", type=int, default=8, help="Set 0 to skip centipawn-loss analysis")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    if args.games_per_level < 1 or args.max_plies < 2 or args.move_time <= 0:
        parser.error("games-per-level, max-plies and move-time must be positive")
    if args.simulations < 0 or args.quality_depth < 0:
        parser.error("simulations and quality-depth cannot be negative")
    levels = [int(value.strip()) for value in args.skill_levels.split(",") if value.strip()]
    if not levels or any(level < 0 or level > 20 for level in levels):
        parser.error("skill-levels must be integers from 0 to 20")

    policy = ChessPolicy(args.checkpoint)
    baseline_policy = ChessPolicy(args.baseline_checkpoint) if args.baseline_checkpoint else None
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    results = []
    baseline_results = Counter()
    aggregate = Counter()
    decision_times = []
    cpl_by_stage = defaultdict(list)
    agreement_by_stage = defaultdict(list)
    try:
        options = engine.options
        for level in levels:
            if "Skill Level" in options:
                engine.configure({"Skill Level": level})
            print(f"\n=== Stockfish Skill Level {level} ===")
            for game_number in range(args.games_per_level):
                agent_color = chess.WHITE if game_number % 2 == 0 else chess.BLACK
                board = chess.Board()
                plies = 0
                game_decision_times = []
                while not board.is_game_over(claim_draw=True) and plies < args.max_plies:
                    if board.turn == agent_color:
                        stage = classify_stage(board)
                        quality_info = None
                        if args.quality_depth > 0:
                            quality_info = engine.analyse(board, chess.engine.Limit(depth=args.quality_depth))
                        start = time.perf_counter()
                        move = choose_agent_move(policy, board, args.simulations)
                        elapsed = time.perf_counter() - start
                        decision_times.append(elapsed)
                        game_decision_times.append(elapsed)
                        if quality_info is not None:
                            best_move = (quality_info.get("pv") or [None])[0]
                            best_score = safe_score_cp(quality_info, board.turn)
                            chosen_info = engine.analyse(
                                board, chess.engine.Limit(depth=args.quality_depth), root_moves=[move]
                            )
                            chosen_score = safe_score_cp(chosen_info, board.turn)
                            if best_move is not None:
                                agreement_by_stage[stage].append(float(move == best_move))
                            if best_score is not None and chosen_score is not None:
                                cpl_by_stage[stage].append(max(0, best_score - chosen_score))
                        board.push(move)
                    else:
                        result = engine.play(board, chess.engine.Limit(time=args.move_time))
                        if result.move is None:
                            break
                        board.push(result.move)
                    plies += 1

                if board.is_game_over(claim_draw=True):
                    outcome = board.outcome(claim_draw=True)
                    if outcome is None or outcome.winner is None:
                        result_label = "draw"
                    elif outcome.winner == agent_color:
                        result_label = "win"
                    else:
                        result_label = "loss"
                elif plies >= args.max_plies:
                    result_label = "truncated"
                else:
                    result_label = "aborted"

                aggregate[result_label] += 1
                entry = {
                    "skill_level": level,
                    "game": game_number + 1,
                    "agent_color": "white" if agent_color else "black",
                    "result": result_label,
                    "plies": plies,
                    "agent_decisions": len(game_decision_times),
                    "mean_decision_seconds": statistics.mean(game_decision_times) if game_decision_times else None,
                }
                results.append(entry)
                print(f"Game {game_number + 1:02d}: {result_label:9s} | {plies:3d} plies | "
                      f"{entry['mean_decision_seconds'] or 0:.3f}s/move")
        if baseline_policy is not None:
            print("\n=== Head-to-head vs compatible baseline ===")
            for game_number in range(args.games_per_level * 2):
                agent_color = chess.WHITE if game_number % 2 == 0 else chess.BLACK
                board = chess.Board()
                plies = 0
                while not board.is_game_over(claim_draw=True) and plies < args.max_plies:
                    active_policy = policy if board.turn == agent_color else baseline_policy
                    move = choose_agent_move(active_policy, board, args.simulations)
                    board.push(move)
                    plies += 1
                if board.is_game_over(claim_draw=True):
                    outcome = board.outcome(claim_draw=True)
                    if outcome is None or outcome.winner is None:
                        label = "draw"
                    elif outcome.winner == agent_color:
                        label = "win"
                    else:
                        label = "loss"
                else:
                    label = "truncated"
                baseline_results[label] += 1
                print(f"Baseline game {game_number + 1:02d}: {label} | {plies} plies")
    finally:
        engine.quit()

    official = aggregate["win"] + aggregate["loss"] + aggregate["draw"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "stockfish": str(args.stockfish),
        "skill_levels": levels,
        "games_per_level": args.games_per_level,
        "simulations": args.simulations,
        "max_plies": args.max_plies,
        "results": dict(aggregate),
        "baseline_head_to_head_results": dict(baseline_results),
        "baseline_checkpoint": str(args.baseline_checkpoint) if args.baseline_checkpoint else None,
        "official_games_excluding_truncation": official,
        "win_rate_score": (aggregate["win"] + 0.5 * aggregate["draw"]) / official if official else None,
        "mean_decision_seconds": statistics.mean(decision_times) if decision_times else None,
        "centipawn_loss_by_stage": {stage: statistics.mean(values) for stage, values in cpl_by_stage.items() if values},
        "stockfish_best_move_agreement_by_stage": {stage: statistics.mean(values) for stage, values in agreement_by_stage.items() if values},
        "games": results,
        "truncation_policy": "Games reaching max_plies are reported as truncated and excluded from win/draw/loss and win-rate calculations.",
    }
    print("\n=== SUMMARY ===")
    print(json.dumps({key: value for key, value in summary.items() if key != "games"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Full report saved: {args.output}")


if __name__ == "__main__":
    main()
