"""Single supervised training path: Stockfish policy targets + value targets."""
from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from agent.network import ChessResNet, count_parameters
from config import (
    BATCH_SIZE, CHECKPOINT_PATH, DATASET_PATH, DEVICE, EARLY_STOPPING_PATIENCE,
    EPOCHS, LEARNING_RATE, VALIDATION_FRACTION, WEIGHT_DECAY,
)
from training.dataset import ChessDataset, collate_chess_batch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grouped_stratified_split(dataset: ChessDataset, validation_fraction: float, seed: int):
    """Split by source game when available; legacy rows fall back to position groups."""
    grouped_indices = defaultdict(list)
    for index, group_id in enumerate(dataset.group_ids):
        grouped_indices[group_id].append(index)

    groups_by_stage = defaultdict(list)
    for group_id, indices in grouped_indices.items():
        stage_counts = Counter(dataset.stages[index] for index in indices)
        representative_stage = stage_counts.most_common(1)[0][0]
        groups_by_stage[representative_stage].append(group_id)

    rng = random.Random(seed)
    train_indices, validation_indices = [], []
    for stage, group_ids in sorted(groups_by_stage.items()):
        rng.shuffle(group_ids)
        stage_sample_count = sum(len(grouped_indices[group_id]) for group_id in group_ids)
        target_validation = max(1, round(stage_sample_count * validation_fraction)) if len(group_ids) > 1 else 0
        validation_count = 0
        for group_index, group_id in enumerate(group_ids):
            remaining_groups = len(group_ids) - group_index
            indices = grouped_indices[group_id]
            # Never move the final group of a stage out of training.
            if validation_count < target_validation and remaining_groups > 1:
                validation_indices.extend(indices)
                validation_count += len(indices)
            else:
                train_indices.extend(indices)

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    if not train_indices or not validation_indices:
        raise ValueError("Need at least two independent groups for train/validation split")

    train_groups = {dataset.group_ids[i] for i in train_indices}
    validation_groups = {dataset.group_ids[i] for i in validation_indices}
    group_overlap = train_groups & validation_groups
    if group_overlap:
        raise RuntimeError(f"Data leakage: {len(group_overlap)} source groups occur in both splits")

    train_keys = {dataset.position_keys[i] for i in train_indices}
    validation_keys = {dataset.position_keys[i] for i in validation_indices}
    position_overlap = train_keys & validation_keys
    if position_overlap:
        raise RuntimeError(f"Data leakage: {len(position_overlap)} position keys occur in both splits")
    return train_indices, validation_indices


def batch_loss(model, batch, device):
    states, candidate_ids, target_weights, values = batch
    states = states.to(device, non_blocking=True)
    candidate_ids = candidate_ids.to(device, non_blocking=True)
    target_weights = target_weights.to(device, non_blocking=True)
    values = values.to(device, non_blocking=True)

    logits, predicted_values = model(states)
    log_probs = F.log_softmax(logits, dim=1)
    candidate_log_probs = log_probs.gather(1, candidate_ids)
    policy_loss = -(target_weights * candidate_log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(predicted_values, values)
    total_loss = policy_loss + value_loss
    top_candidate = candidate_ids[:, 0]
    policy_accuracy = (logits.argmax(dim=1) == top_candidate).float().mean()
    value_mae = (predicted_values - values).abs().mean()
    return total_loss, policy_loss, value_loss, policy_accuracy, value_mae


def run_epoch(model, loader, optimizer, device, scaler=None, train=True, use_amp=False):
    model.train(train)
    totals = np.zeros(5, dtype=np.float64)
    count = 0
    for batch in loader:
        batch_size = batch[0].shape[0]
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                losses = batch_loss(model, batch, device)
            loss = losses[0]
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
        totals += np.asarray([item.detach().float().item() for item in losses]) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("Empty data loader")
    return dict(zip(("loss", "policy_loss", "value_loss", "policy_accuracy", "value_mae"), (totals / count).tolist()))


def restore_scaler_state(scaler, scaler_state, enabled: bool) -> bool:
    """Restore AMP state only when the saved checkpoint actually used AMP.

    CPU checkpoints serialize a disabled GradScaler as an empty dict. That
    empty state must not be loaded into an enabled CUDA GradScaler when
    resuming training on a GPU.
    """
    if not enabled or not isinstance(scaler_state, dict) or not scaler_state:
        return False
    scaler.load_state_dict(scaler_state)
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train the single Stockfish-supervised chess model")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--resume", action="store_true", help="Resume from the output checkpoint if compatible")
    parser.add_argument("--warm-start", type=Path, default=None, help="Initialize weights from a compatible earlier Policy+Value checkpoint")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")
    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error("validation-fraction must be between 0.05 and 0.40")

    seed_everything(args.seed)
    device = torch.device(DEVICE)
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"CUDA mixed precision: {'on' if use_amp else 'off'}")

    dataset = ChessDataset(args.dataset)
    train_ids, val_ids = grouped_stratified_split(dataset, args.validation_fraction, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_ids), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=collate_chess_batch, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_ids), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=collate_chess_batch, persistent_workers=args.workers > 0,
    )

    model = ChessResNet().to(device)
    warm_started_from = None
    if args.warm_start is not None:
        warm_start_data = torch.load(args.warm_start, map_location=device, weights_only=True)
        warm_state = warm_start_data.get("model_state_dict", warm_start_data)
        model.load_state_dict(warm_state, strict=True)
        warm_started_from = str(args.warm_start)
        print(f"Warm-started compatible Policy+Value weights from: {args.warm_start}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    start_epoch = 1
    best_loss = float("inf")
    resume_data = None
    if args.resume and args.checkpoint.is_file():
        resume_data = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if resume_data.get("dataset_sha256") != sha256_file(args.dataset):
            raise ValueError("Cannot resume: checkpoint was trained with a different dataset hash")
        model.load_state_dict(resume_data["model_state_dict"])
        if "optimizer_state_dict" in resume_data:
            optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        start_epoch = int(resume_data.get("epoch", 0)) + 1
        best_loss = float(resume_data.get("validation_metrics", {}).get("loss", best_loss))
        warm_started_from = resume_data.get("warm_started_from")
        print(f"Resuming at epoch {start_epoch}; best val loss {best_loss:.5f}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    if resume_data is not None and "scheduler_state_dict" in resume_data:
        scheduler.load_state_dict(resume_data["scheduler_state_dict"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if resume_data is not None:
        restore_scaler_state(scaler, resume_data.get("scaler_state_dict"), use_amp)
    checkpoint_path = args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_hash = sha256_file(args.dataset)
    no_improvement = 0
    print(f"Samples: {len(dataset)} | train: {len(train_ids)} | validation: {len(val_ids)}")
    print(f"Parameters: {count_parameters(model):,}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, scaler, train=True, use_amp=use_amp)
        with torch.inference_mode():
            val_metrics = run_epoch(model, val_loader, optimizer, device, train=False, use_amp=use_amp)
        scheduler.step(val_metrics["loss"])
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train {train_metrics['loss']:.4f} | "
            f"val {val_metrics['loss']:.4f} | policy {val_metrics['policy_loss']:.4f} | "
            f"value {val_metrics['value_loss']:.4f} | top1 {val_metrics['policy_accuracy']:.3f} | "
            f"value MAE {val_metrics['value_mae']:.3f} | lr {lr:.2e}"
        )
        if val_metrics["loss"] < best_loss - 1e-6:
            best_loss = val_metrics["loss"]
            no_improvement = 0
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "validation_metrics": val_metrics,
                "training_metrics": train_metrics,
                "model_config": {"input_channels": 18, "channels": 64, "residual_blocks": 4, "action_size": 20480},
                "dataset_sha256": dataset_hash,
                "dataset_samples": len(dataset),
                "seed": args.seed,
                "amp": use_amp,
                "encoding": "side_to_move_v1",
                "warm_started_from": warm_started_from,
            }
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
            print(f"  saved best checkpoint: {checkpoint_path}")
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            print("Early stopping triggered")
            break
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Model: {checkpoint_path}")


if __name__ == "__main__":
    main()
