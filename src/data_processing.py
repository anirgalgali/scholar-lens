import re
from collections import OrderedDict
from itertools import chain
from datasets import Dataset
from typing import List, Dict

def match_category_pattern(input_string:str) -> str:
     
     # This regex seems to work for most examples. Will require further testing
     pattern = re.compile(r'[\w\s]*Archive(->)([A-Za-z.-]+)(>|\w*)?([A-Za-z.-]*)')
     # Selecting the second and fourth groups, but also 3rd just in case
     matched_pattern = pattern.sub(r'\2\3\4', input_string)
     return matched_pattern

def get_tag_dict(ds: Dataset) -> OrderedDict:
    all_categories = OrderedDict()
    for i, c in enumerate(ds["tag"]):
        all_categories[i] = c
    return all_categories

def get_category_tags(example: Dict[str,List[str]], all_tags: OrderedDict) \
      -> Dict[str,None|str]:
    
    # This function should be compatible with the map() method in datasets
    
    all_tags_ = all_tags.values()
    # Handling the fact that there maybe more than one ctaegory
    categories = []
    for string in  example['categories']:
        matched_pattern  = match_category_pattern(string)
        # Sometimes the regex picks up a -> which needs to be removed 
        matched_pattern = matched_pattern.split('->')
        categories.append(matched_pattern)

    # IN case there is more than one category, unpack that nested list and 
    # remove duplicates. This removes repeated elements such as:
    # astro-ph->astro-ph->astro-ph.Co returing only [astro-ph, astro-ph.CO]
    categories = set(chain.from_iterable(categories))
    
    # The parse may have resulted in substrings that do not appear in our original
    # tag dictionary. Gott remove those. For example:
    # non-lin->non-lin.CD will result in two elements [non-lin, non-lin.CD], only the
    # latter appears in the tag dict.

    invalid_categories = set()
    for c in categories:
        if c not in all_tags_:
            invalid_categories.add(c)
    categories -= invalid_categories
    categories = list(categories)
    # Selecting the first occurence as the label. In case the first occurence is a
    # complete substring
    if len(categories) == 0:
        return dict(category = None)
    
    if len(categories) > 1:
        if any([categories[0] in c for c in categories[1:]]):
            return dict(category = categories[1])
        else:
            return dict(category = categories[0])
    
    return dict(category = categories[0])

def get_category_label(example: List[str], all_tags: OrderedDict) -> Dict[str,None|int]:

    try:
        label_ = [k for k , v in all_tags.items() if example['category'] == v]
    except KeyError as e:
        print(f'dataset needs category feature for this to work')
        return

    if len(label_) != 1:
        raise ValueError('Each datapoint can have only a sinle label')

    return dict(label = label_)