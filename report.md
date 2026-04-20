# REPORT — Self-Pruning Neural Network

## WHY L1 ON SIGMOID GATES ENCOURAGES SPARSITY
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

## SPARSITY vs ACCURACY TRADE-OFF
Higher λ → stronger constant pressure to zero all gates →
more connections pruned → smaller effective network →
lower test accuracy (capacity reduced).

Lower λ → gates mostly stay near 1.0 → full network capacity
→ better accuracy but no compression benefit.
