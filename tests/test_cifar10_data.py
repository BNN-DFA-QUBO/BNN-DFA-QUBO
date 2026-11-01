"""Unit tests for CIFAR-10 data loading, transforms, and stratified splitting."""
import unittest
import numpy as np
import torch
from PIL import Image

from utils.CIFAR10.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    TransformedSubset,
    get_cifar10_split_indices,
    get_device,
    get_eval_transform,
    get_train_transform,
)


class MockCIFAR10Dataset:
    """Synthetic CIFAR-10 dataset for fast unit tests without network download."""

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        np.random.seed(42)
        # 32x32x3 uint8 array matching torchvision CIFAR10.data
        self.data = np.random.randint(0, 256, (num_samples, 32, 32, 3), dtype=np.uint8)
        # Balanced 10-class labels
        self.targets = [i % 10 for i in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.data[idx])
        return img, self.targets[idx]


class TestCIFAR10Data(unittest.TestCase):

    def setUp(self):
        self.synthetic_train = MockCIFAR10Dataset(num_samples=100)
        self.synthetic_test = MockCIFAR10Dataset(num_samples=20)

    def test_image_dimensions_and_channels(self):
        eval_transform = get_eval_transform()
        subset = TransformedSubset(self.synthetic_train, list(range(10)), eval_transform)
        img, target = subset[0]

        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (3, 32, 32), "Image must be 3x32x32 RGB tensor")
        self.assertIsInstance(target, int)
        self.assertIn(target, list(range(10)))

    def test_stratified_split(self):
        split = get_cifar10_split_indices(
            self.synthetic_train, self.synthetic_test, seed=42, validation_fraction=0.2
        )
        self.assertEqual(len(split["train"]), 80)
        self.assertEqual(len(split["validation"]), 20)
        self.assertEqual(len(split["test"]), 20)

        # Check stratification across 10 classes
        train_labels = [self.synthetic_train.targets[i] for i in split["train"]]
        val_labels = [self.synthetic_train.targets[i] for i in split["validation"]]

        train_counts = np.bincount(train_labels, minlength=10)
        val_counts = np.bincount(val_labels, minlength=10)

        self.assertTrue(np.all(train_counts == 8), "Train set must have exactly 8 per class")
        self.assertTrue(np.all(val_counts == 2), "Val set must have exactly 2 per class")

    def test_deterministic_eval_transform(self):
        eval_transform = get_eval_transform()
        img_pil = Image.fromarray(self.synthetic_train.data[0])

        t1 = eval_transform(img_pil)
        t2 = eval_transform(img_pil)

        self.assertTrue(torch.equal(t1, t2), "Eval transforms must be deterministic")

    def test_augmented_transform_behavior(self):
        torch.manual_seed(42)
        train_transform = get_train_transform()
        img_pil = Image.fromarray(self.synthetic_train.data[0])

        # Over multiple augmentations with random crop and flip, values should vary
        samples = [train_transform(img_pil) for _ in range(5)]
        any_diff = any(not torch.equal(samples[0], samples[i]) for i in range(1, 5))
        self.assertTrue(any_diff, "Training transforms should perform augmentation")

    def test_label_alignment_preserved(self):
        eval_transform = get_eval_transform()
        indices = [5, 12, 19, 42]
        subset = TransformedSubset(self.synthetic_train, indices, eval_transform)

        for i, real_idx in enumerate(indices):
            _, label = subset[i]
            expected_label = self.synthetic_train.targets[real_idx]
            self.assertEqual(label, expected_label, "Label must match underlying dataset index")

    def test_device_helper(self):
        dev = get_device()
        self.assertIsInstance(dev, torch.device)
        self.assertIn(dev.type, ["cpu", "cuda", "mps"])


if __name__ == "__main__":
    unittest.main()
