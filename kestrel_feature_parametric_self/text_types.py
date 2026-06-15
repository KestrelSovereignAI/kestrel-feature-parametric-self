"""Text-native LoRA training config for the parametric self.

`TextLoRAConfig` is the text counterpart to core's image-bound
`kestrel_sovereign.features.training.types.TrainingConfig` — same dataclass
style (defaults + ``to_dict``/``from_dict``), but text-native fields only. It
deliberately carries **no** ``trigger_word`` / ``resolution`` / ``flux_version``
(Strategy B, epic #1): the image config is not overloaded to carry text.

Field names map directly onto the ``mlx_lm.lora`` CLI so the adapter can build
its command without translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# The base model proven in the feasibility run (docs/TWO_BRAIN_ARCHITECTURE.md §7).
DEFAULT_BASE_MODEL = "gemma-4-31B-it-mlx-4bit"


@dataclass
class TextLoRAConfig:
    """Configuration for a local MLX LoRA run on the agent's own corpus."""

    # Model + data
    base_model: str = DEFAULT_BASE_MODEL  # MLX model dir or HF id
    data_dir: Optional[str] = None        # dir holding train.jsonl / valid.jsonl
    adapter_path: Optional[str] = None    # where the trained adapter is written

    # LoRA / optimisation (mlx_lm.lora flags)
    num_layers: int = 16
    iters: int = 400
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_seq_length: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "data_dir": self.data_dir,
            "adapter_path": self.adapter_path,
            "num_layers": self.num_layers,
            "iters": self.iters,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextLoRAConfig":
        return cls(
            base_model=data.get("base_model", DEFAULT_BASE_MODEL),
            data_dir=data.get("data_dir"),
            adapter_path=data.get("adapter_path"),
            num_layers=data.get("num_layers", 16),
            iters=data.get("iters", 400),
            batch_size=data.get("batch_size", 4),
            learning_rate=data.get("learning_rate", 1e-4),
            max_seq_length=data.get("max_seq_length", 2048),
        )
