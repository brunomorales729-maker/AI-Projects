import unittest

import numpy as np
import torch

from agent.encoding import ACTION_SIZE, encode_fen, encode_uci
from agent.network import ChessResNet, count_parameters
from training.dataset import ChessDataset
from training.train import grouped_stratified_split, restore_scaler_state

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class EncodingTests(unittest.TestCase):
    def test_initial_board_tensor(self):
        tensor = encode_fen(START_FEN)
        self.assertEqual(tensor.shape, (18, 8, 8))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(float(tensor[0].sum()), 8.0)
        self.assertEqual(float(tensor[6].sum()), 8.0)
        self.assertTrue(np.all(tensor[12] == 1.0))
        self.assertTrue(np.all(tensor[13] == 1.0))

    def test_side_to_move_canonicalizes_equivalent_pawn_push(self):
        white_e2e4 = encode_uci("e2e4", "w")
        black_e7e5 = encode_uci("e7e5", "b")
        self.assertEqual(white_e2e4, black_e7e5)

    def test_promotion_actions_are_distinct(self):
        actions = [encode_uci(f"a7a8{p}", "w") for p in "qrbn"]
        self.assertEqual(len(set(actions)), 4)
        self.assertTrue(all(0 <= action < ACTION_SIZE for action in actions))

    def test_bad_fen_rejected(self):
        with self.assertRaises(ValueError):
            encode_fen("not a fen")


class ModelTests(unittest.TestCase):
    def test_forward_shapes_and_parameter_budget(self):
        model = ChessResNet().eval()
        with torch.inference_mode():
            policy, value = model(torch.zeros(2, 18, 8, 8))
        self.assertEqual(tuple(policy.shape), (2, ACTION_SIZE))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertLess(count_parameters(model), 500_000)


class DatasetTests(unittest.TestCase):
    def test_final_dataset_and_leakage_free_split(self):
        dataset = ChessDataset()
        self.assertEqual(len(dataset), 10996)
        train_ids, validation_ids = grouped_stratified_split(dataset, 0.15, 42)
        self.assertFalse(set(dataset.position_keys[i] for i in train_ids) & set(dataset.position_keys[i] for i in validation_ids))
        self.assertEqual(len(train_ids) + len(validation_ids), len(dataset))
        self.assertEqual(len(dataset.group_ids), len(dataset))
        self.assertTrue(all(dataset.group_ids))
        train_groups = {dataset.group_ids[i] for i in train_ids}
        validation_groups = {dataset.group_ids[i] for i in validation_ids}
        self.assertFalse(train_groups & validation_groups)


class TrainingResumeTests(unittest.TestCase):
    def test_empty_cpu_scaler_state_is_skipped_for_cuda_resume(self):
        class FakeScaler:
            def __init__(self):
                self.loaded = None

            def load_state_dict(self, state):
                self.loaded = state

        scaler = FakeScaler()
        self.assertFalse(restore_scaler_state(scaler, {}, enabled=True))
        self.assertIsNone(scaler.loaded)
        self.assertFalse(restore_scaler_state(scaler, {"scale": 1024}, enabled=False))
        self.assertIsNone(scaler.loaded)
        self.assertTrue(restore_scaler_state(scaler, {"scale": 1024}, enabled=True))
        self.assertEqual(scaler.loaded, {"scale": 1024})


if __name__ == "__main__":
    unittest.main()
