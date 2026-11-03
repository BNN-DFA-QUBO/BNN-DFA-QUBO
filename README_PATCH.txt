# ResNet18 + QUBO CIFAR-10 experimental patch

## What this adds
This is a **separate transfer-learning experiment**. It does not overwrite or relabel the existing BNN-DFA v4 pipeline.

- ImageNet-pretrained ResNet18, fine-tuned on CIFAR-10.
- Stratified train/validation split from the official training set; official test set remains held out.
- A single shared 512-dimensional penultimate representation is extracted and standardized using training-set statistics only.
- The same representation is evaluated using Real LS, Binarized LS, Direct Binary, STE Binary, and QUBO Binary heads.
- Best backbone checkpoint is selected using validation accuracy.
- Optional QUBO parameter selection uses validation only.

## Apply
From the repository root, extract the patch ZIP:
```bash
unzip -o ~/Downloads/resnet18_qubo_patch.zip -d .
```

## Run (recommended first run)
```bash
python -m experiments.CIFAR10_experiments.run_resnet18_qubo --seed 42 --epochs 20 --batch-size 128 --qubo-tune
```

First run downloads CIFAR-10 and the torchvision ImageNet ResNet18 weights if not already cached. Internet access is required for those downloads. On Apple Silicon, MPS is selected automatically. If memory pressure occurs, use `--batch-size 64`.

For a fixed QUBO setting without tuning:
```bash
python -m experiments.CIFAR10_experiments.run_resnet18_qubo --seed 42 --epochs 20 --batch-size 128 --num-reads 50 --num-sweeps 500
```

Results and checkpoint are saved to `data/resnet18_cifar10/`.

## Scientific limitations
- This is **not** the original BNN-DFA backbone. It is a pretrained conventional CNN fine-tuned on CIFAR-10.
- The five classifier heads share exactly the same frozen features, but the backbone is supervised transfer learning.
- Accuracy targets (QUBO >=90%, other heads >=75-80%) are goals, not guaranteed outcomes. A one-seed result is preliminary; repeat with multiple seeds.
- The QUBO optimizer is classical simulated annealing (dwave-neal), not quantum hardware.
- Do not tune against the official test set.
