"""Controlled CIFAR-10 BNN-DFA Stabilization Experiment Runner.

Executes and benchmarks stabilized DFA training regimes:
1. opt_control: Learning rate 1e-4, Gradient clipping 1.0, Weight decay 1e-4.
2. latent_clamp: Latent weight clamping [-1, 1], LR 3e-4, Gradient clipping 1.0.
3. batch_norm: CIFAR_BNN_BN (architectural variant with BatchNorm).

Preserves all historical baseline files and records provenance, progression,
and comparative metrics separately in data/cifar10_stabilization/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.cifar_bnn import CIFAR_BNN, BinaryConv2d
from models.bnn import BinaryLinear
from models.cifar_dfa import CIFARDFA
from utils.CIFAR10.data import (
    DEFAULT_DATA_DIR,
    TransformedSubset,
    get_cifar10_raw_datasets,
    get_eval_transform,
    get_train_transform,
    get_device,
)
from utils.MNIST.seed import set_seed
from experiments.CIFAR10_experiments.cifar_bnn_dfa import collect_hidden_features
from experiments.CIFAR10_experiments.cifar_dfa_least_squares import fit_and_evaluate_ridge
from experiments.CIFAR10_experiments.cifar_dfa_binarized_ls import evaluate_binarized_head
from experiments.CIFAR10_experiments.cifar_dfa_direct_binary import evaluate_direct_binary_head as eval_direct_base
from experiments.CIFAR10_experiments.stabilization.stabilized_direct_binary import evaluate_direct_binary_stabilized as eval_direct_stab
from experiments.CIFAR10_experiments.cifar_dfa_ste_binary import evaluate_ste_head
from experiments.CIFAR10_experiments.tune_qubo_validation import tune_from_representations
from experiments.CIFAR10_experiments.cifar_dfa_qubo import evaluate_qubo
from utils.run_provenance import collect_run_provenance, stable_run_id

STAB_DATA_DIR = Path("data/cifar10_stabilization")
STAB_REP_DIR = STAB_DATA_DIR / "representations"
STAB_RESULTS_DIR = STAB_DATA_DIR / "results"
STAB_RESULTS_CSV = STAB_RESULTS_DIR / "cifar10_stabilization_results.csv"
PROGRESSION_CSV = STAB_RESULTS_DIR / "cifar10_representation_progression.csv"


class CIFAR_BNN_BN(nn.Module):
    """Architectural variant with Batch Normalization."""
    def __init__(self):
        super().__init__()
        self.conv1 = BinaryConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = BinaryConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = BinaryLinear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = BinaryLinear(128, 10)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.bn3(self.fc1(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.extract_features(x))

    def forward_with_intermediates(self, x: torch.Tensor):
        c1_pre = self.bn1(self.conv1(x))
        c1_act = torch.relu(c1_pre)
        c2_pre = self.bn2(self.conv2(c1_act))
        c2_act = torch.relu(c2_pre)
        gap_act = torch.flatten(self.gap(c2_act), start_dim=1)
        d_pre = self.bn3(self.fc1(gap_act))
        d_act = torch.relu(d_pre)
        out = self.fc2(d_act)
        return out, c1_pre, c1_act, c2_pre, c2_act, gap_act, d_pre, d_act


def train_stabilized_seed(
    seed: int,
    config: str = "opt_control",
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    clamp_weights: bool = False,
    device: str | torch.device | None = None,
) -> Dict[str, Any]:
    """Train a single seed under a specified stabilization protocol."""
    if device is None:
        device = get_device()
    dev = torch.device(device)
    set_seed(seed)

    tr, te = get_cifar10_raw_datasets(download=False)
    # Stratified 45,000 / 5,000 / 10,000 indices
    from utils.CIFAR10.data import get_cifar10_split_indices
    indices = get_cifar10_split_indices(tr, te, seed=seed, validation_fraction=0.1)

    train_loader = DataLoader(
        TransformedSubset(tr, indices["train"], get_train_transform()),
        batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_eval_loader = DataLoader(
        TransformedSubset(tr, indices["train"], get_eval_transform()),
        batch_size=256, shuffle=False,
    )
    val_loader = DataLoader(
        TransformedSubset(tr, indices["validation"], get_eval_transform()),
        batch_size=256, shuffle=False,
    )
    test_loader = DataLoader(
        TransformedSubset(te, indices["test"], get_eval_transform()),
        batch_size=256, shuffle=False,
    )

    use_bn = (config == "batch_norm")
    if config == "latent_clamp":
        clamp_weights = True
        lr = 3e-4

    if use_bn:
        model = CIFAR_BNN_BN().to(dev)
    else:
        model = CIFAR_BNN().to(dev)

    dfa = CIFARDFA(device=dev, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 1. Evaluate Epoch 0 (Random Init)
    print(f"\n--- Seed {seed} | Config: {config} | Epoch 0 (Random Init) ---")
    H_tr0, y_tr0 = collect_hidden_features(model, train_eval_loader, dev)
    H_val0, y_val0 = collect_hidden_features(model, val_loader, dev)
    H_te0, y_te0 = collect_hidden_features(model, test_loader, dev)
    _, acc0, _ = fit_and_evaluate_ridge(H_tr0, y_tr0, H_val0, y_val0, H_te0, y_te0)
    print(f"Epoch 00 (Random Init): Ridge LS Val={acc0['validation']:.2f}%, Test={acc0['test']:.2f}%, H_norm={H_tr0.norm(dim=1).mean().item():.2f}")

    progression_records = [{
        "seed": seed, "config": config, "epoch": 0,
        "ridge_val_acc": acc0["validation"], "ridge_test_acc": acc0["test"],
        "ridge_train_acc": acc0["train"],
        "H_mean": H_tr0.mean().item(), "H_std": H_tr0.std().item(),
        "H_norm": H_tr0.norm(dim=1).mean().item(),
        "H_sparsity": (H_tr0 == 0).float().mean().item() * 100.0,
        "bnn_train_loss": 2.3026, "bnn_train_acc": 10.0,
        "bnn_val_loss": 2.3026, "bnn_val_acc": 10.0,
    }]

    check_epochs = {1, 2, 5, 10, 20}
    current_epoch = 0
    t0_training = time.perf_counter()

    for target_ep in sorted(check_epochs):
        while current_epoch < target_ep:
            current_epoch += 1
            model.train()
            tot_loss, corr, tot = 0.0, 0, 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(dev), lbls.to(dev)
                optimizer.zero_grad()
                intermediates = model.forward_with_intermediates(imgs)
                outputs = intermediates[0]
                loss = F.cross_entropy(outputs, lbls)
                grads = dfa.compute_local_gradients(model, imgs, lbls, intermediates)
                dfa.apply_gradients(model, grads)

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                optimizer.step()

                if clamp_weights:
                    with torch.no_grad():
                        for name, p in model.named_parameters():
                            if "weight" in name and "bn" not in name:
                                p.clamp_(-1.0, 1.0)

                tot_loss += loss.item() * lbls.size(0)
                corr += (outputs.argmax(dim=1) == lbls).sum().item()
                tot += lbls.size(0)

            tr_loss = tot_loss / tot
            tr_acc = 100.0 * corr / tot

        # Evaluate model & representation at checkpoint epoch
        model.eval()
        v_corr, v_tot, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for v_img, v_lbl in val_loader:
                v_img, v_lbl = v_img.to(dev), v_lbl.to(dev)
                v_out = model(v_img)
                v_loss += F.cross_entropy(v_out, v_lbl).item() * v_lbl.size(0)
                v_corr += (v_out.argmax(dim=1) == v_lbl).sum().item()
                v_tot += v_lbl.size(0)
        v_acc = 100.0 * v_corr / v_tot
        v_loss = v_loss / v_tot

        H_tr, y_tr = collect_hidden_features(model, train_eval_loader, dev)
        H_val, y_val = collect_hidden_features(model, val_loader, dev)
        H_te, y_te = collect_hidden_features(model, test_loader, dev)
        _, acc_ep, _ = fit_and_evaluate_ridge(H_tr, y_tr, H_val, y_val, H_te, y_te)

        print(f"Epoch {current_epoch:02d}: BNN tr_loss={tr_loss:.3f}, tr_acc={tr_acc:.2f}%, val_loss={v_loss:.3f}, val_acc={v_acc:.2f}% | Ridge Val={acc_ep['validation']:.2f}%, Test={acc_ep['test']:.2f}% | H_norm={H_tr.norm(dim=1).mean().item():.2f}")

        progression_records.append({
            "seed": seed, "config": config, "epoch": current_epoch,
            "ridge_val_acc": acc_ep["validation"], "ridge_test_acc": acc_ep["test"],
            "ridge_train_acc": acc_ep["train"],
            "H_mean": H_tr.mean().item(), "H_std": H_tr.std().item(),
            "H_norm": H_tr.norm(dim=1).mean().item(),
            "H_sparsity": (H_tr == 0).float().mean().item() * 100.0,
            "bnn_train_loss": tr_loss, "bnn_train_acc": tr_acc,
            "bnn_val_loss": v_loss, "bnn_val_acc": v_acc,
        })

    total_training_sec = time.perf_counter() - t0_training

    # Save frozen representation artifact at epoch 20
    STAB_REP_DIR.mkdir(parents=True, exist_ok=True)
    rep_save_path = STAB_REP_DIR / f"rep_{config}_seed_{seed}_ep20.pt"
    torch.save({
        "H_train": H_tr.cpu(), "y_train": y_tr.cpu(),
        "H_validation": H_val.cpu(), "y_validation": y_val.cpu(),
        "H_test": H_te.cpu(), "y_test": y_te.cpu(),
        "config": config, "seed": seed, "epochs": epochs,
        "training_time_sec": total_training_sec,
    }, rep_save_path)
    print(f"[SAVED] Saved stabilized representation to {rep_save_path}")

    # Now evaluate all 5 classifier heads on this final frozen representation
    print(f"\nEvaluating all 5 heads on stabilized representation (Seed {seed})...")
    head_records = []

    # 1. Real LS
    w_ridge, acc_ridge, t_ridge = fit_and_evaluate_ridge(H_tr, y_tr, H_val, y_val, H_te, y_te)
    head_records.append({
        "config": config, "seed": seed, "method": "Real Least Squares",
        "train_acc": acc_ridge["train"], "val_acc": acc_ridge["validation"], "test_acc": acc_ridge["test"],
        "fit_time_sec": t_ridge,
    })

    # 2. Binarized LS
    _, acc_bin, t_bin = evaluate_binarized_head(H_tr, y_tr, H_val, y_val, H_te, y_te, w_ridge)
    head_records.append({
        "config": config, "seed": seed, "method": "Binarized Least Squares",
        "train_acc": acc_bin["train"], "val_acc": acc_bin["validation"], "test_acc": acc_bin["test"],
        "fit_time_sec": t_bin,
    })

    # 3. Direct Binary Baseline (Original Unscaled)
    _, acc_dir_base, t_dir_base = eval_direct_base(H_tr, y_tr, H_val, y_val, H_te, y_te, seed=seed)
    head_records.append({
        "config": config, "seed": seed, "method": "Direct Binary (Baseline)",
        "train_acc": acc_dir_base["train"], "val_acc": acc_dir_base["validation"], "test_acc": acc_dir_base["test"],
        "fit_time_sec": t_dir_base,
    })

    # 4. Direct Binary Stabilized (Scale/Bias Refit)
    _, acc_dir_stab, t_dir_stab = eval_direct_stab(H_tr, y_tr, H_val, y_val, H_te, y_te, seed=seed)
    head_records.append({
        "config": config, "seed": seed, "method": "Direct Binary (Stabilized)",
        "train_acc": acc_dir_stab["train"], "val_acc": acc_dir_stab["validation"], "test_acc": acc_dir_stab["test"],
        "fit_time_sec": t_dir_stab,
    })

    # 5. STE Binary
    _, acc_ste, t_ste = evaluate_ste_head(H_tr, y_tr, H_val, y_val, H_te, y_te, seed=seed)
    head_records.append({
        "config": config, "seed": seed, "method": "STE Binary",
        "train_acc": acc_ste["train"], "val_acc": acc_ste["validation"], "test_acc": acc_ste["test"],
        "fit_time_sec": t_ste,
    })

    # 6. QUBO Binary (Validation Grid Tuned)
    winner, _ = tune_from_representations(H_tr, y_tr, H_val, y_val, seed=seed)
    qubo_test = evaluate_qubo(H_te, y_te, winner["_weights"], winner["_scales"], winner["_bias"])
    qubo_train = evaluate_qubo(H_tr, y_tr, winner["_weights"], winner["_scales"], winner["_bias"])
    head_records.append({
        "config": config, "seed": seed, "method": "QUBO Binary",
        "train_acc": qubo_train, "val_acc": winner["validation_accuracy"], "test_acc": qubo_test,
        "fit_time_sec": winner["tuning_total_seconds"],
        "notes": f"reads={winner['num_reads']}, sweeps={winner['num_sweeps']}, mult={winner['scale_multiplier']}",
    })

    return {
        "progression": progression_records,
        "heads": head_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Run CIFAR-10 BNN-DFA stabilization experiment.")
    parser.add_argument("--config", type=str, default="opt_control", choices=["opt_control", "latent_clamp", "batch_norm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 999])
    parser.add_argument("--single-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    seeds = [args.single_seed] if args.single_seed is not None else args.seeds
    STAB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_progression = []
    all_heads = []

    for seed in seeds:
        res = train_stabilized_seed(seed=seed, config=args.config, epochs=args.epochs)
        all_progression.extend(res["progression"])
        all_heads.extend(res["heads"])

    # Append to progression CSV
    df_prog = pd.DataFrame(all_progression)
    if PROGRESSION_CSV.exists():
        df_old_prog = pd.read_csv(PROGRESSION_CSV)
        df_prog = pd.concat([df_old_prog, df_prog], ignore_index=True).drop_duplicates(subset=["config", "seed", "epoch"])
    df_prog.to_csv(PROGRESSION_CSV, index=False)
    print(f"\n[UPDATED] Progression records written to {PROGRESSION_CSV}")

    # Append to results CSV
    df_heads = pd.DataFrame(all_heads)
    if STAB_RESULTS_CSV.exists():
        df_old_heads = pd.read_csv(STAB_RESULTS_CSV)
        df_heads = pd.concat([df_old_heads, df_heads], ignore_index=True).drop_duplicates(subset=["config", "seed", "method"])
    df_heads.to_csv(STAB_RESULTS_CSV, index=False)
    print(f"[UPDATED] Head results written to {STAB_RESULTS_CSV}")

    # Summary table
    summary = df_heads.groupby(["config", "method"])["test_acc"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    print("\n" + "=" * 80)
    print(f"Summary for Config: {args.config}")
    print("-" * 80)
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
