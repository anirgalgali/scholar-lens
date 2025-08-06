import pytest
from datasets import load_dataset
from src.data_processing import get_tag_dict


@pytest.fixture(scope="package")
def tag_dict():
    ds_cats = load_dataset("TimSchopf/arxiv_categories",  "arxiv_category_descriptions",split="arxiv_category_descriptions")
    return get_tag_dict(ds_cats)