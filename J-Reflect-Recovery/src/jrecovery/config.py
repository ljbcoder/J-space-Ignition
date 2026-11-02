"""Small, strict configuration; resolved revisions are part of the run identity."""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Model:
    name: str = "Qwen/Qwen3.5-9B"
    revision: str = "main"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 64
    max_context: int = 4096
    temperature: float = 0.0
    enable_thinking: bool = False


@dataclass
class Lens:
    repo: str = "koayon/jpp-lenses"
    revision: str = "main"
    filename: str = "qwen3.5-9b/lens.pt"
    layers: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 28])
    top_k: int = 10
    save_activations: bool = True


@dataclass
class Data:
    dev_count: int = 200
    test_count: int = 1000
    tail_steps: int = 3


@dataclass
class Analysis:
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    minimum_dev_errors: int = 10


@dataclass
class Config:
    name: str = "qwen35-jpp-recovery-pilot"
    seed: int = 42
    model: Model = field(default_factory=Model)
    lens: Lens = field(default_factory=Lens)
    data: Data = field(default_factory=Data)
    analysis: Analysis = field(default_factory=Analysis)

    def validate(self):
        if self.model.temperature != 0 or self.model.enable_thinking:
            raise ValueError(
                "This pilot freezes greedy, answer-only decoding with thinking disabled"
            )
        if self.model.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Unsupported model dtype")
        if not 0 < self.model.max_new_tokens < self.model.max_context:
            raise ValueError("Invalid context or generation budget")
        if not self.lens.layers or sorted(set(self.lens.layers)) != self.lens.layers:
            raise ValueError("Lens layers must be sorted, unique, and nonempty")
        if self.lens.layers[0] < 0 or not 1 <= self.lens.top_k <= 100:
            raise ValueError("Invalid readout geometry")
        if min(self.data.dev_count, self.data.test_count) < 6:
            raise ValueError("Each split needs at least six problems")
        if not 0 <= self.data.tail_steps <= 8:
            raise ValueError("tail_steps must be between 0 and 8")
        if self.analysis.bootstrap_samples < 100 or not 0 < self.analysis.confidence < 1:
            raise ValueError("Invalid bootstrap settings")
        if self.analysis.minimum_dev_errors < 1:
            raise ValueError("Development gate needs a positive minimum error count")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, constructor in [
            ("model", Model),
            ("lens", Lens),
            ("data", Data),
            ("analysis", Analysis),
        ]:
            values[key] = constructor(**values.get(key, {}))
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(tomllib.loads(Path(path).read_text()))
