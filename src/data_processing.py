import os
from pathlib import Path
import re
from collections import OrderedDict, defaultdict, Counter
from itertools import chain
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
import json
from datasets import Dataset, DatasetDict, load_dataset
import random
from src.config import DataConfig, ModelConfig
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# random.seed(42)


def match_category_pattern(input_string: str) -> str:

    # This regex seems to work for most examples. Will require further testing

    # This needs to be in every string else it should return a value error
    base_pattern = re.compile(r"Archive->")
    if base_pattern.search(input_string) is None:
        raise ValueError("invalid category string")
    else:
        pattern = re.compile(r"([\w\s]*)Archive(->)([A-Za-z.-]+)(>|\w*)?([A-Za-z.-]*)")
        # Selecting the second and fourth groups, but also 3rd just in case
        matched_pattern = pattern.sub(r"\3\4\5", input_string)
        return matched_pattern


def match_subject_pattern(input_string: str) -> str:

    # This regex seems to work for most examples. Will require further testing

    # This needs to be in every string else it should return a value error
    base_pattern = re.compile(r"Archive->")
    if base_pattern.search(input_string) is None:
        raise ValueError("invalid category string")
    else:
        pattern = re.compile(r"([\w\s]*)Archive(->)([A-Za-z.-]+)(>|\w*)?([A-Za-z.-]*)")
        # Selecting the second and fourth groups, but also 3rd just in case
        matched_pattern = pattern.sub(r"\1", input_string)
        return matched_pattern


def get_tag_dict(ds: Dataset) -> OrderedDict:
    all_categories = OrderedDict()
    for i, c in enumerate(ds["tag"]):
        all_categories[i] = c
    return all_categories


def get_category_tags(
    example: Dict[str, List[str]], all_tags: OrderedDict
) -> Dict[str, None | str]:

    # This function should be compatible with the map() method in datasets

    all_tags_ = all_tags.values()
    # Handling the fact that there maybe more than one ctaegory
    categories = []
    subjects = []
    for string in example["categories"]:
        try:
            matched_pattern = match_category_pattern(string)
            # Sometimes the regex picks up a -> which needs to be removed
            matched_pattern = matched_pattern.split("->")
            categories.append(matched_pattern)
            subject_pattern = match_subject_pattern(string)
            subjects.append([subject_pattern] * len(matched_pattern))
        except ValueError as e:
            continue

    if len(categories) == 0:
        return dict(primary_category=None, subject=None)

    # This removes repeated elements such as:
    # astro-ph->astro-ph->astro-ph.CO returing only [astro-ph, astro-ph.CO]
    categories = [list(dict.fromkeys(c)) for c in categories]

    categories = list(chain.from_iterable(categories))
    subjects = list(chain.from_iterable(subjects))

    # The parse may have resulted in substrings that do not appear in our original
    # tag dictionary. Gotta remove those. For example:
    # non-lin->non-lin.CD will result in two elements [non-lin, non-lin.CD], only the
    # latter appears in the tag dict.
    invalid_categories = []
    for c, s in zip(categories, subjects):
        if c not in all_tags_:
            invalid_categories.append(c)

    category_subjects = [
        (c, s) for c, s in zip(categories, subjects) if c not in invalid_categories
    ]
    category_subjects = list(map(list, zip(*category_subjects)))
    categories = category_subjects[0]
    subjects = category_subjects[1]
    # print(categories)
    # print(subjects)
    assert len(categories) == len(subjects)

    # Selecting the first occurence as the label. In case the first occurence is a
    # complete substring of subsequentt element then you want to select the subsequent
    # element as that would be a more specific label

    if len(categories) > 1:
        if any([categories[0] in c for c in categories[1:]]):
            return dict(primary_category=categories[1], subject=subjects[0].rstrip(" "))
        else:
            return dict(primary_category=categories[0], subject=subjects[0].rstrip(" "))

    return dict(primary_category=categories[0], subject=subjects[0].rstrip(" "))


def get_category_label(
    example: List[str], all_tags: OrderedDict
) -> Dict[str, None | int]:

    try:
        label_ = [k for k, v in all_tags.items() if example["category"] == v]
    except KeyError as e:
        print(f"dataset needs category feature for this to work")
        return

    if len(label_) != 1:
        raise ValueError("Each datapoint can have only a sinle label")

    return dict(label=label_)


def get_topk_categories_by_subject(
    categories: List[str], subjects: List[str], k: int = 5
):

    category_subject_pairs = [(c, s) for (c, s) in zip(categories, subjects)]
    category_subject_counts = dict(Counter(category_subject_pairs))
    # We first sort based on counts and then based on the subject
    category_subject_counts = dict(
        sorted(
            category_subject_counts.items(),
            key=lambda item: (item[1], item[0][-1]),
            reverse=True,
        )
    )

    subject_count = defaultdict(int)
    topk_category_count_by_subject = defaultdict(dict)
    for key, val in category_subject_counts.items():
        if subject_count[key[-1]] > k - 1:
            continue
        else:
            topk_category_count_by_subject[key[-1]].update({key[0]: val})
            subject_count[key[-1]] += 1

    return topk_category_count_by_subject


def create_input_from_abstract_title(example: Dict[str, List[str]]) -> Dict[str, str]:

    input_text = example["title"] + ". " + example["abstract"]
    input_text = input_text.lower()
    input_text = BeautifulSoup(input_text, "html.parser").get_text()  # remove html tags
    input_text = re.sub(r"https?//\S+|www\.\S+", "", input_text)  # remove website names
    input_text = re.sub(r"\S+@\S*\s?", "", input_text)  # remove email addresses
    return dict(input=" ".join(input_text.split()))


def get_category_label(
    example: List[str], category_label_mapping: Dict[str, int]
) -> Dict[str, None | int]:

    try:
        label_ = [
            v for (k, v) in category_label_mapping.items() if example["category"] == k
        ]
    except KeyError as e:
        print(e)
        return
    except AssertionError as e:
        print(e)

    if len(label_) > 1:
        raise ValueError("Each datapoint can have only a single label")
    elif len(label_) == 0:
        raise ValueError(
            "The category tag for this datapoint must have been filtered out"
        )

    return dict(label=label_[0])


def extract_category_stats_for_relabeling(train_dataset: Dataset):

    df = train_dataset.to_pandas()
    unique_labels = df["primary_category"].unique()

    parent_labels = set()
    for label_a in unique_labels:
        for label_b in unique_labels:
            if label_a != label_b and label_b.startswith(label_a + "."):
                parent_labels.add(label_a)

    print(f"Found parent categories: {list(parent_labels)}")

    relabel_plan = {}
    for parent in parent_labels:
        # Filter for rows where the label is a specific sub-category of the parent
        children_df = df[df["primary_category"].str.startswith(parent + ".")]
        if not children_df.empty:
            proportions = children_df["primary_category"].value_counts(normalize=True)
            relabel_plan[parent] = {
                "children": proportions.index.tolist(),
                "probs": proportions.values.tolist(),
            }

    return relabel_plan


def apply_relabeling_batch(batch, relabel_plan, rng=np.random.default_rng()):
    """
    A function to be used with .map() to apply the re-labeling plan.
    """

    new_labels = []
    # 'primary_label' is the column of initially parsed labels
    for label in batch["primary_category"]:
        if label in relabel_plan:
            # This is a incomplete parent label, so we sample a child from subcategories
            plan = relabel_plan[label]
            new_labels.append(rng.choice(plan["children"], p=plan["probs"]))
        else:
            # This is not a incomplete label, so we keep the original
            new_labels.append(label)

    batch["category"] = new_labels
    return batch


def _process_dataset(data_config: DataConfig) -> Tuple[DatasetDict, Dict]:

    rng = np.random.default_rng(data_config.random_state)

    try:
        ds_raw = load_dataset(data_config.dataset_identifier, "default")
        ds_categories = load_dataset(
            data_config.dataset_identifier,
            "arxiv_category_descriptions",
            split="arxiv_category_descriptions",
        )
    except ValueError as e:
        print(e)

    all_tag_labels = get_tag_dict(ds_categories)
    print(f" Extracting category and subject-level information")
    ds_raw = ds_raw.map(
        lambda x: get_category_tags(x, all_tag_labels),
        load_from_cache_file=data_config.load_from_cache,
    )

    relabel_plan = extract_category_stats_for_relabeling(ds_raw["train"])
    print(f"Generated Re-labeling Plan: {relabel_plan}")

    # 2. Apply the plan to all splits (train, validation, and test)
    ds_raw = ds_raw.map(
        apply_relabeling_batch,
        batched=True,
        fn_kwargs={
            "relabel_plan": relabel_plan,
            "rng": rng,
        },  # Pass the plan to the map function
    )

    print(f" Cleaning input text ")
    ds_raw = ds_raw.map(
        lambda x: create_input_from_abstract_title(x),
        load_from_cache_file=data_config.load_from_cache,
    )

    num_categories_to_retain_per_subject = data_config.num_categories_per_subject
    subjects_to_retain = data_config.subjects
    print(
        f"Retaining top { num_categories_to_retain_per_subject } categories in subjects: {', '.join(subjects_to_retain)}..."
    )

    # You want to maje sure that you select the top-k categories based only on the
    # training data distribution

    topk_category_counts_by_subject = get_topk_categories_by_subject(
        list(ds_raw["train"]["category"]),
        list(ds_raw["train"]["subject"]),
        k=num_categories_to_retain_per_subject,
    )

    categories_retained = []
    for subj, (tag_counts) in topk_category_counts_by_subject.items():
        if subj in subjects_to_retain:
            categories_retained.append([k for (k, _) in tag_counts.items()])
    categories_retained = list(chain.from_iterable(categories_retained))

    # Shuffling the categories to ensure that they aren't organized by subject
    rng.shuffle(categories_retained)
    category_labels = dict()
    for i, tag in enumerate(categories_retained):
        category_labels[tag] = i

    ds_filtered = ds_raw.filter(lambda x: x["category"] in categories_retained)
    print(f" Categories retained: {', '.join(categories_retained)}")
    print(f"Mapping filtered examples to labels")
    ds_filtered = ds_filtered.map(lambda x: get_category_label(x, category_labels))
    return ds_filtered, category_labels


def _tokenize_dataset(
    processed_data: DatasetDict, model_config: ModelConfig
) -> DatasetDict:

    print(f"Tokenizing using {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    if model_config.dynamic_padding:
        padding = False
    else:
        padding = "max_length"

    return processed_data.map(
        lambda example: tokenizer(
            example["input"],
            padding=padding,
            truncation=model_config.truncate_tokens,
            max_length=model_config.max_tokens,
        ),
        batched=True,
    )


def _process_and_tokenize_dataset(
    data_config: DataConfig, model_config: ModelConfig
) -> DatasetDict:
    processed, category_label_dict = _process_dataset(data_config)
    tokenized = _tokenize_dataset(processed, model_config)
    return tokenized, processed, category_label_dict


def get_processed_dataset(data_config: DataConfig) -> DatasetDict:
    cache_path = f"./data/processed/{data_config.version_id}"

    if Path(cache_path).exists():
        print(f"Loading processed dataset from cache path : {cache_path}")
        return DatasetDict.load_from_disk(cache_path)
    else:
        print(f" Loading and preprocessing {data_config.dataset_identifier}...")
        dataset, category_label_mapping = _process_dataset(data_config)
        print(f" Saving processed data to {cache_path}...")
        dataset.save_to_disk(cache_path)
        with open(os.path.join(cache_path, "id2label.json"), "w") as f:
            json.dump(category_label_mapping, f)
        return dataset


def get_tokenized_dataset(
    data_config: DataConfig, model_config=ModelConfig
) -> DatasetDict:

    cache_path = (
        f"./data/tokenized/{data_config.version_id}/{model_config.tokenized_version}"
    )
    processed_cache_path = f"./data/processed/{data_config.version_id}"
    if Path(cache_path).exists():
        print(f"Loading already tokenized dataset from cache path : {cache_path}")
        return DatasetDict.load_from_disk(cache_path)
    elif Path(processed_cache_path).exists():
        print(f"Loading processed dataset from cache path : {processed_cache_path}")
        dataset = DatasetDict.load_from_disk(processed_cache_path)
        tokenized_dataset = _tokenize_dataset(dataset, model_config)
        print(f" Saving processed and tokenized data to {cache_path}...")
        tokenized_dataset.save_to_disk(cache_path)
        return tokenized_dataset
    else:
        print(
            f" Loading, preprocessing and tokenizing {data_config.dataset_identifier}..."
        )
        tokenized_dataset, processed_dataset, category_label_mapping = (
            _process_and_tokenize_dataset(data_config, model_config)
        )

        print(f" Saving processed data to {processed_cache_path}...")
        processed_dataset.save_to_disk(processed_cache_path)
        with open(os.path.join(processed_cache_path, "id2label.json"), "w") as f:
            json.dump(category_label_mapping, f)

        print(f" Saving tokenized data to {cache_path}...")
        tokenized_dataset.save_to_disk(cache_path)

        return tokenized_dataset


if __name__ == "__main__":

    data_config = DataConfig(num_categories_per_subject=15)
    num_classes = len(data_config.subjects) * data_config.num_categories_per_subject
    model_config = ModelConfig(num_classes=num_classes)
    tokenized_dataset = get_tokenized_dataset(data_config, model_config)
