# model/quantum_layer.py
# Switchable feature-interaction layer.
#
# HadamardLayer (default, INTERACTION_MODE="hadamard"):
#   Fixed 4×4 Hadamard matrix mixes all 4 bottleneck features together,
#   so every output depends on every input.
#   One learnable scale vector (4 params) controls per-feature weighting.
#   This is the quantum-inspired layer: it models different feature interactions.
#
# VQCLayer (INTERACTION_MODE="vqc"):
#   Real 4-qubit PennyLane VQC — angle embedding + variational RY/RZ +
#   ring CNOT + Pauli-Z measurement.
#   Install: pip install pennylane pennylane-lightning

import torch
import torch.nn as nn
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as C


# ── Hadamard matrix (normalised, 4×4) ─────────────────────────────────────
def _hadamard_4() -> torch.Tensor:
    """Returns the 4×4 normalised Hadamard matrix as a fixed float32 tensor."""
    H = torch.tensor([
        [1.,  1.,  1.,  1.],
        [1., -1.,  1., -1.],
        [1.,  1., -1., -1.],
        [1., -1., -1.,  1.],
    ]) / 2.0   # normalisation: H @ H.T = I
    return H


class HadamardLayer(nn.Module):
    """
    Quantum-inspired feature interaction via fixed Hadamard transform
    followed by learnable per-feature scaling.

    Forward: x [B, 4] → H·x (fixed mix) → scale (learned) → Tanh → [B, 4]

    Why this models lesion interactions:
    - Each output is a ±sum of all inputs → "looks at" all lesion channels
    - The learned scale lets the model weight which interactions matter
    - Ablation: set scale=ones → pure Hadamard, no learned weighting
    """
    def __init__(self, dim: int = C.BOTTLENECK_DIM):
        super().__init__()
        assert dim == 4, "HadamardLayer requires BOTTLENECK_DIM=4"
        self.register_buffer("H", _hadamard_4())  # fixed, not trained
        self.scale = nn.Parameter(torch.ones(dim))  # 4 learnable weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4]
        mixed = x @ self.H.T          # [B, 4]  — Hadamard transform
        out   = mixed * self.scale     # [B, 4]  — per-feature weighting
        return torch.tanh(out)         # [B, 4]  — bounded output


# ── VQC Layer ──────────────────────────────────────────────────────────────

class VQCLayer(nn.Module):
    """
    4-qubit Variational Quantum Circuit.
    Only instantiated when INTERACTION_MODE="vqc".
    Requires: pip install pennylane pennylane-lightning
    """
    def __init__(self, dim: int = C.BOTTLENECK_DIM,
                 n_layers: int = C.NUM_Q_LAYERS):
        super().__init__()
        self.dim      = dim
        self.n_layers = n_layers
        self._build_qnode()
        self.q_weights = nn.Parameter(
            torch.randn(n_layers, dim, 2) * 0.01)

    def _build_qnode(self):
        import pennylane as qml
        dev = qml.device("lightning.qubit", wires=self.dim)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            # Angle embedding
            for i in range(self.dim):
                qml.RY(torch.arctan(inputs[i]), wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.dim):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # Ring CNOT entanglement
                for i in range(self.dim):
                    qml.CNOT(wires=[i, (i + 1) % self.dim])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.dim)]

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outputs = []
        for i in range(B):
            result = self._circuit(x[i], self.q_weights)
            outputs.append(torch.stack(result))
        return torch.stack(outputs).float()   # [B, dim]


# ── Factory ────────────────────────────────────────────────────────────────

def build_interaction_layer(mode: Optional[str] = None) -> nn.Module:
    """Returns the correct layer based on explicit mode or config.INTERACTION_MODE."""
    mode = (mode or C.INTERACTION_MODE).lower()
    if mode == "hadamard":
        layer = HadamardLayer(C.BOTTLENECK_DIM)
        print(f"  Interaction layer: HadamardLayer (dim={C.BOTTLENECK_DIM}, fast)")
    elif mode == "vqc":
        layer = VQCLayer(C.BOTTLENECK_DIM, C.NUM_Q_LAYERS)
        print(f"  Interaction layer: VQCLayer (qubits={C.BOTTLENECK_DIM}, "
              f"layers={C.NUM_Q_LAYERS})")
    else:
        raise ValueError(f"Unknown INTERACTION_MODE: {mode!r}. "
                         f"Choose 'hadamard' or 'vqc'.")
    return layer
