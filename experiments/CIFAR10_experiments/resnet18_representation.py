"""Extract frozen 512-D features from trained ResNet18 CIFAR-10 checkpoints."""

from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[int(self.indices[idx])]
        return self.transform(image), label


def load_or_extract_resnet18_representation(
    seed,
    output_dir=Path("data/xai_artifacts/resnet18_cifar10/representations"),
    checkpoint_dir=Path("data/resnet18_cifar10"),
    data_dir=Path("data/cifar10/raw"),
    device=None,
    overwrite=False,
):
    device = device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(checkpoint_dir) / f"resnet18_cifar10_seed_{seed}.pt"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"ResNet18 checkpoint missing for seed {seed}: {checkpoint}"
        )

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    artifact = output_dir / f"resnet18_seed_{seed}_features.pt"

    if artifact.exists() and not overwrite:
        cached = torch.load(artifact, map_location="cpu", weights_only=False)

        if cached.get("checkpoint_sha256") == digest:
            return cached, artifact, "cached_resnet18_checkpoint"

    checkpoint_data = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(checkpoint_data["state_dict"])
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    raw_train = datasets.CIFAR10(
        str(data_dir), train=True, download=True
    )
    raw_test = datasets.CIFAR10(
        str(data_dir), train=False, download=True
    )

    labels = np.asarray(raw_train.targets)

    train_idx, val_idx = train_test_split(
        np.arange(len(raw_train)),
        test_size=0.1,
        random_state=seed,
        stratify=labels,
    )

    datasets_to_process = [
        IndexedDataset(raw_train, train_idx, transform),
        IndexedDataset(raw_train, val_idx, transform),
        IndexedDataset(raw_test, np.arange(len(raw_test)), transform),
    ]

    features, targets = [], []

    with torch.inference_mode():
        for dataset in datasets_to_process:
            loader = DataLoader(
                dataset, batch_size=256, shuffle=False, num_workers=0
            )

            h_list, y_list = [], []

            for images, labels_batch in loader:
                h = model(images.to(device))
                h_list.append(h.cpu())
                y_list.append(labels_batch.cpu())

            features.append(torch.cat(h_list))
            targets.append(torch.cat(y_list))

    H_train, H_val, H_test = features

    # Match the original ResNet18 head-training protocol.
    mu = H_train.mean(0, keepdim=True)
    sd = H_train.std(0, keepdim=True).clamp_min(1e-5)

    H_train = (H_train - mu) / sd
    H_val = (H_val - mu) / sd
    H_test = (H_test - mu) / sd

    payload = {
        "H_train": H_train,
        "y_train": targets[0],
        "H_val": H_val,
        "y_val": targets[1],
        "H_test": H_test,
        "y_test": targets[2],
        "seed": seed,
        "feature_dim": 512,
        "checkpoint_sha256": digest,
        "checkpoint": str(checkpoint),
        "metadata": {
            "backbone": "ImageNet-pretrained ResNet18 fine-tuned on CIFAR-10",
            "feature_dim": 512,
            "seed": seed,
            "normalization": "ImageNet",
        },
    }

    torch.save(payload, artifact)

    return payload, artifact, "extracted_resnet18_checkpoint"