"""Single source of runtime and training configuration."""
from pathlib import Path
import os
import torch

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "stockfish_dataset.jsonl"
CHECKPOINT_PATH = ROOT / "models" / "chess_model_best.pth"
STOCKFISH_PATH = Path(os.environ.get("STOCKFISH_EXECUTABLE", "stockfish"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CHANNELS = 18
ACTION_SIZE = 20_480
RESIDUAL_CHANNELS = 64
RESIDUAL_BLOCKS = 4
POLICY_EMBEDDING = 32
VALUE_HIDDEN = 128

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.15
EARLY_STOPPING_PATIENCE = 7
POLICY_TEMPERATURE_CP = 100.0
VALUE_SCALE_CP = 600.0
DEFAULT_MCTS_SIMULATIONS = 50
