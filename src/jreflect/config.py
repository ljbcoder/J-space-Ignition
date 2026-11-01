"""Small, strict configuration; no ML imports needed to inspect a run."""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str = "main"
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_new_tokens: int = 32
    max_length: int = 2048


@dataclass
class DataConfig:
    name: str = "openai/gsm8k"
    subset: str = "main"
    revision: str = "main"
    train_count: int = 7473
    test_count: int = 1319
    dev_size: int = 500
    gate_size: int = 200


@dataclass
class ReadoutConfig:
    layers: list[int] = field(default_factory=lambda: [3, 7, 11, 15, 19, 23])
    middle: list[int] = field(default_factory=lambda: [11, 15])
    late: list[int] = field(default_factory=lambda: [19, 23])
    top_k: int = 20
    concepts: int = 5


@dataclass
class TrainingConfig:
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: int = 2
    micro_batch: int = 1
    effective_batch: int = 16
    warmup_ratio: float = 0.0
    gradient_checkpointing: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 10000
    confidence: float = 0.95


@dataclass
class Config:
    experiment: str = "qwen25-gsm8k-initial"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self):
        if not 0 < self.data.gate_size < self.data.dev_size < self.data.train_count:
            raise ValueError("Require 0 < gate_size < dev_size < train_count")
        r = self.readout
        if not r.layers or min(r.layers) < 0 or len(set(r.layers)) != len(r.layers):
            raise ValueError("Readout layers must be distinct nonnegative block indices")
        if not r.middle or not r.late or not set(r.middle + r.late) <= set(r.layers):
            raise ValueError("Both readout bands must be nonempty subsets of layers")
        if set(r.middle) & set(r.late):
            raise ValueError("Readout bands must be disjoint")
        if not 0 < r.concepts <= r.top_k:
            raise ValueError("Require 0 < concepts <= top_k")
        t = self.training
        if min(t.micro_batch, t.effective_batch, t.epochs, t.rank, t.alpha) <= 0:
            raise ValueError("Training counts must be positive")
        if t.effective_batch % t.micro_batch:
            raise ValueError("effective_batch must be divisible by micro_batch")
        if not 0 <= t.dropout < 1 or not 0 <= t.warmup_ratio < 1 or t.learning_rate <= 0:
            raise ValueError("Invalid training rates")
        if not t.seeds or len(set(t.seeds)) != len(t.seeds):
            raise ValueError("Specify distinct training seeds")
        if self.model.dtype not in {"bfloat16", "float32", "float16"}:
            raise ValueError("Unsupported model dtype")
        if not 0 < self.model.max_new_tokens < self.model.max_length:
            raise ValueError("Invalid model length limits")
        if self.evaluation.bootstrap_samples < 1 or not 0 < self.evaluation.confidence < 1:
            raise ValueError("Invalid bootstrap settings")
        return self

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, typ in {
            "model": ModelConfig,
            "data": DataConfig,
            "readout": ReadoutConfig,
            "training": TrainingConfig,
            "evaluation": EvaluationConfig,
        }.items():
            if key in values:
                values[key] = typ(**values[key])
        return cls(**values).validate()


def load_config(path):
    with Path(path).open("rb") as stream:
        return Config.from_dict(tomllib.load(stream))
