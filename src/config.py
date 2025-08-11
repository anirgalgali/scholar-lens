from dataclasses import dataclass
from typing import List
import hashlib


@dataclass
class DataConfig:

    """ Configuration for data preprocessing"""

    dataset_identifier: str = "TimSchopf/arxiv_categories"
    load_from_cache: bool = False
    subjects: List[str] = ['Physics', 'Mathematics', 'Computer Science']
    num_categories_per_subject: int = 8

    @property
    def version_id(self)->str:
        sorted_subjects = sorted(self.subjects)
        subjects_str = "-".join(sorted_subjects[:4])
        return f"subjs_{subjects_str}_numcatspersubj_{self.num_categories_per_subject}"
    

@dataclass
class BaselineModelConfig:
        
        """Default config for baseline model"""
        ngram_range: tuple = (1,2) # tuple that tells you the min and max n-grams.
        min_df: int = 2
        max_df: float = 0.95
        strip_accents: str = 'unicode'
        stop_words: str = 'english'
        token_pattern:str = r'\b[\w\-]+\b'
        max_features: int = 20000
        C:float = 1.0
        multi_class:str = 'multinomial'
        class_weight: str = 'balanced'
        solver: str = 'lbfgs'
        max_iter: int = 1000
        random_state: int = 42          