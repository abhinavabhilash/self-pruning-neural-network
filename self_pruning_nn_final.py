import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 3.0))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates          = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))

    def get_sparsity_loss(self) -> torch.Tensor:
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(torch.sigmoid(module.gate_scores).flatten())
        return torch.cat(all_gates).mean()

    def get_sparsity_level(self, threshold: float = 0.05) -> float:
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().flatten())
        return (torch.cat(all_gates) < threshold).float().mean().item() * 100.0

    def get_all_gates(self) -> np.ndarray:
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().flatten().cpu().numpy())
        return np.concatenate(all_gates)

    def gate_params(self):
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module.gate_scores

    def weight_params(self):
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module.weight
                yield module.bias
            elif isinstance(module, nn.BatchNorm1d):
                yield from module.parameters()


def get_cifar10_loaders(batch_size: int = 128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, lambda_val, device, epoch, total_epochs):
    model.train()
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0
    ce_last = sp_last = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits        = model(images)
        ce_loss       = criterion(logits, labels)
        sparsity_loss = model.get_sparsity_loss()
        loss          = ce_loss + lambda_val * sparsity_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        ce_last, sp_last = ce_loss.item(), sparsity_loss.item()

    avg_loss = total_loss / len(loader)
    sparsity = model.get_sparsity_level()
    print(f"  Epoch [{epoch:2d}/{total_epochs}]  "
          f"CE: {ce_last:.4f}  "
          f"GateMean: {sp_last:.4f}  "
          f"Sparsity: {sparsity:.1f}%  "
          f"AvgLoss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds    = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total * 100.0


def plot_gate_distribution(model, lambda_val, filename):
    gates      = model.get_all_gates()
    pruned_pct = (gates < 0.05).mean() * 100
    active_pct = (gates >= 0.5).mean()  * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(gates, bins=60, color='steelblue',
                                edgecolor='white', linewidth=0.3)

    for patch, left in zip(patches, bins[:-1]):
        if left < 0.05:
            patch.set_facecolor('#e74c3c')

    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=1.5,
               label='Sparsity threshold (0.05)')

    ax.set_xlabel("Gate Value (Sigmoid Output)", fontsize=12)
    ax.set_ylabel("Number of Weights", fontsize=12)
    ax.set_title(
        f"Gate Distribution  |  λ = {lambda_val}  |  "
        f"Pruned (<0.05): {pruned_pct:.1f}%   Active (≥0.5): {active_pct:.1f}%",
        fontsize=12, pad=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Plot saved → {filename}")


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    EPOCHS     = 20
    BATCH_SIZE = 128
    LAMBDAS = [1.0, 5.0, 20.0]

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)
    results = []

    for lambda_val in LAMBDAS:
        print("=" * 65)
        print(f"  λ = {lambda_val}")
        print("=" * 65)

        model = SelfPruningNet().to(device)

        optimizer = torch.optim.Adam([
            {'params': list(model.weight_params()), 'lr': 1e-3},
            {'params': list(model.gate_params()),   'lr': 1e-2},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=EPOCHS)

        for epoch in range(1, EPOCHS + 1):
            train_epoch(model, train_loader, optimizer,
                        lambda_val, device, epoch, EPOCHS)
            scheduler.step()

        acc      = evaluate(model, test_loader, device)
        sparsity = model.get_sparsity_level(threshold=0.05)

        print(f"\n  → Test Accuracy : {acc:.2f}%")
        print(f"  → Sparsity Level: {sparsity:.2f}%\n")
        results.append((lambda_val, acc, sparsity))

        plot_gate_distribution(
            model, lambda_val,
            filename=f"gate_distribution_lambda_{lambda_val}.png"
        )

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>14}")
    print("-" * 44)
    for lam, acc, sp in results:
        print(f"{lam:<12} {acc:>14.2f}% {sp:>13.2f}%")

    print("""
=============================================================
REPORT — Self-Pruning Neural Network
=============================================================

WHY L1 ON SIGMOID GATES ENCOURAGES SPARSITY
---------------------------------------------
The sigmoid function maps each gate_score to the range (0, 1).
Because all gate values are positive, their L1 norm equals
their sum (or equivalently, their mean × count).

The optimizer minimizes:
  total_loss = CE_loss + λ × mean(all gates)

This gives every gate_score a constant gradient of magnitude
  λ × sigmoid(s) × (1 - sigmoid(s)) / N
pointing in the direction of decreasing s (toward -∞, gate→0).

This is the core property of L1 regularization: the gradient
does NOT vanish as the value approaches zero (unlike L2, where
gradient shrinks proportionally to the value). L1 keeps pushing
gates to exactly 0, creating true sparsity.

Once a gate collapses below ~0.05, the weight it multiplies is
effectively removed from the network, while the system remains
fully differentiable throughout training.

SPARSITY vs ACCURACY TRADE-OFF
--------------------------------
Higher λ → stronger constant pressure to zero all gates →
more connections pruned → smaller effective network →
lower test accuracy (capacity reduced).

Lower λ → gates mostly stay near 1.0 → full network capacity
→ better accuracy but no compression benefit.
=============================================================
""")


if __name__ == "__main__":
    main()