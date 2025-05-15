import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationEnsemble(nn.Module):
    def __init__(self, negative_slope=0.01, elu_alpha=1.0):
        super().__init__()
        # 7 activations → learnable weights
        self.weights = nn.Parameter(torch.ones(7))
        self.negative_slope = negative_slope
        self.elu_alpha = elu_alpha

    def forward(self, x):
        activations = []

        # 1. Sigmoid
        activations.append(torch.sigmoid(x))

        # 2. Tanh
        activations.append(torch.tanh(x))

        # 3. ReLU
        activations.append(F.relu(x))

        # 4. Leaky ReLU
        activations.append(F.leaky_relu(x, negative_slope=self.negative_slope))

        # 5. GELU
        activations.append(F.gelu(x))

        # 6. ELU
        activations.append(F.elu(x, alpha=self.elu_alpha))

        # 7. SiLU (Swish)
        activations.append(F.silu(x))

        # Softmax-normalized weights
        w = F.softmax(self.weights, dim=0)

        # Weighted sum
        out = sum(w[i] * act for i, act in enumerate(activations))
        return out

class AdaNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(AdaNorm, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * self.norm(x)

class Adapter(nn.Module):
    def __init__(self, dim, adapter_dim):
        super(Adapter, self).__init__()
        self.down = nn.Linear(dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(adapter_dim, dim)

    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))

class MemoryModule(nn.Module):
    def __init__(self, dim, memory_slots=10):
        super(MemoryModule, self).__init__()
        self.memory_slots = memory_slots
        self.dim = dim
        self.register_buffer("memory", torch.zeros(1, memory_slots, dim))

    def forward(self, x):
        # x: (batch, seq_len, dim)
        batch_size = x.size(0)
        memory = self.memory.expand(batch_size, -1, -1)  # Expand for batch

        # Concatenate input with memory
        x_cat = torch.cat([memory, x], dim=1)

        return x_cat, memory

    def update_memory(self, new_info):
        # new_info: (batch, seq_len, dim) -> mean over seq
        new_mem = new_info.mean(dim=1, keepdim=True)  # (batch, 1, dim)
        self.memory = torch.cat([new_mem.detach(), self.memory[:, :-1]], dim=1)


class PM4DA2EFourierHartley(nn.Module):
    """
    Replaces the attention mechanism by applying a 2D Fourier transform to the input.
    Operates over the sequence length and feature dimensions.
    """
    def __init__(self):
        super(PM4DA2EFourierHartley, self).__init__()

    def forward(self, x):
        # Apply 2D Fourier transform (real part only, as in PM4DA2E)
        x_ft = torch.fft.fft2(x.float(), dim=(-2, -1))
        return x_ft.real


class PM4DA2EBlock(nn.Module):
    """
    A single PM4DA2E encoder block with Fourier mixing and feedforward network.
    """
    def __init__(self, dim, hidden_dim, adapter_dim, dropout=0.02):
        super(PM4DA2EBlock, self).__init__()
        self.fourier = PM4DA2EFourierHartley()
        self.activation = ActivationEnsemble()
        self.norm1 = AdaNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = AdaNorm(dim)
        self.adapter = Adapter(dim, adapter_dim)

    def forward(self, x):
        # Fourier mixing + residual
        x = x + self.fourier(self.norm1(x))
        # Feedforward + residual
        x = x + self.ff(self.norm2(x))
        x = self.adapter(x)
        return x


class PM4DA2EModel(nn.Module):
    def __init__(self, vocab_size, seq_len, dim, hidden_dim, num_blocks, num_classes, dropout=0.1):
        super(PM4DA2EModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.blocks = nn.Sequential(*[
            PM4DA2EBlock(dim, hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding
        x = self.blocks(x)
        # Pool over sequence (mean)
        x = x.mean(dim=1)
        return self.classifier(x)


class PM4DA2EWithMemory(nn.Module):
    def __init__(self, vocab_size, seq_len, dim, hidden_dim, adapter_dim, num_blocks, num_classes, memory_slots=10, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + memory_slots, dim))
        self.memory_module = MemoryModule(dim, memory_slots)

        self.blocks = nn.ModuleList([
            PM4DA2EBlockWithMemory(dim, hidden_dim, adapter_dim, dropout) for _ in range(num_blocks)
        ])

        self.norm = AdaNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Embed + positional encoding
        x = self.embedding(x)
        x, memory = self.memory_module(x)  # (batch, memory + seq_len, dim)
        x = x + self.pos_embedding[:, :x.size(1), :]

        for block in self.blocks:
            x = block(x)

        self.memory_module.update_memory(x[:, -x.size(1)//2:])  # Update with last half

        x = self.norm(x).mean(dim=1)
        return self.classifier(x)
