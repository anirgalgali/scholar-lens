from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Configuration for data preprocessing"""

    dataset_identifier: str = "TimSchopf/arxiv_categories"
    load_from_cache: bool = False
    subjects: List[str] = field(
        default_factory=lambda: ["Physics", "Mathematics", "Computer Science"]
    )
    num_categories_per_subject: int = 8

    @property
    def version_id(self) -> str:
        sorted_subjects = sorted(self.subjects)
        subjects_str = "-".join(sorted_subjects[:4])
        return f"subjs_{subjects_str}_numcatspersubj_{self.num_categories_per_subject}"


@dataclass
class BaselineModelConfig:
    """Default config for baseline model"""

    ngram_range: tuple = (1, 2)  # tuple that tells you the min and max n-grams.
    min_df: int = 2
    max_df: float = 0.95
    strip_accents: str = "unicode"
    stop_words: str = "english"
    token_pattern: str = r"\b[\w\-]+\b"
    max_features: int = 20000
    C: float = 1.0
    multi_class: str = "multinomial"
    class_weight: str = "balanced"
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_state: int = 42


@dataclass
class ModelConfig:
    "Config for pretrained model"

    num_classes: int = (
        157  # N, for N-way classification, default=all possible categories
    )
    model_name: str = "distilbert-base-uncased"
    dynamic_padding: bool = True  # This ensures padding happens at batch level
    max_tokens = 512  # This is set by the model architecture
    truncate_tokens: bool = True

    @property
    def tokenized_version(self) -> str:
        return f"{self.model_name}_maxtokens={self.max_tokens}_prepad={not self.dynamic_padding}"


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 128
    resume_from_checkpoint: str = None
    reset_scheduler_on_load: bool = False

@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
