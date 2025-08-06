import pytest
from src.data_processing import match_category_pattern, get_category_tags

# TESTING THE REGEX MATCH

def test_match_category_pattern_single_level() -> None:
    # This tests a single level heirarchy of the following
    # form : ... Archive->xx', where xx is typically letters and dots
    input_string = 'Computer Science Archive->cs.CV'
    expected = 'cs.CV'
    result = match_category_pattern(input_string)
    assert result == expected

def test_match_category_pattern_invalid_input()-> None:
    input_string = 'This is not a valid pattern'
    with pytest.raises(ValueError):
        result = match_category_pattern(input_string)

def test_match_category_pattern_single_level_arbitrary_prefix()-> None:
    input_string = "blah blah blah blah Archive->cs.AI"
    expected = 'cs.AI'
    result = match_category_pattern(input_string)
    assert result == expected

def test_match_category_pattern_double_level_simple() -> None:
    # This tests a double level heirarchy of the following
    # form : Archive->xx->yy' where xx is only letters and 
    # yy can contian only dots or letters
    input_string = 'Physics Archive->nlin->nlin.CD'
    expected = 'nlin->nlin.CD'
    result = match_category_pattern(input_string)
    assert result == expected

def test_match_category_pattern_double_level_withhyphen() -> None:
    # This tests a double level heirarchy of the following
    # form : Archive->xx->yy' where xx is only letters and 
    # yy can contian letters and hyphens
    input_string = 'Physics Archive->hep->hep-ph'
    expected = 'hep->hep-ph'
    result = match_category_pattern(input_string)
    assert result == expected

def test_match_category_pattern_double_level_withhyphendouble() -> None:
    # This tests a double level heirarchy of the following
    # form : Archive->xx->yy' where xx is only letters and 
    # yy can contian letters,  hyphens and dots
    input_string = 'Physics Archive->cond-mat->cond-mat.mtrl-sci'
    expected = 'cond-mat->cond-mat.mtrl-sci'
    result = match_category_pattern(input_string)
    assert result == expected


# TESTING THE CATEGORY EXTRACTION

def test_get_category_tags_single(tag_dict) -> None:
    # This tests when the categories column in the dataset
    # only contains a single entry
    example = dict(categories = ['Computer Science Archive->cs.CV'])
    result = get_category_tags(example, tag_dict)
    expected = dict(category = 'cs.CV')
    assert result == expected

def test_get_category_tags_fail(tag_dict) -> None:
    # This tests when the categories column in the dataset
    # only contains a single entry
    example = dict(categories = ['This is not valid'])
    result = get_category_tags(example, tag_dict)
    expected = dict(category = None)
    assert result == expected
     
def test_get_category_tags_double(tag_dict) -> None:
    # This tests when the categories column in the dataset
    # contains two entries each one being different. The 
    # function should  return the first listed category tag
    example = dict(categories =  ['Computer Science Archive->cs.AI', 'Computer Science Archive->cs.CV'])
    result = get_category_tags(example, tag_dict)
    expected = dict(category = 'cs.AI')
    assert result == expected 

def test_get_category_tags_single_withmissing(tag_dict) -> None:
    # This tests when the categories column in the dataset
    # only contains a single entries but the parse results in an
    # intermediate string that is not present in tag_dict
    example = dict(categories =  ['Physics Archive->nlin->nlin.CD'])
    result = get_category_tags(example, tag_dict)
    expected = dict(category = 'nlin.CD')
    assert result == expected 

def test_get_category_tags_single_repeatingsubstring(tag_dict) -> None:
    # This tests when the categories column in the dataset
    # contains two entries each one being different. The 
    # function should  return the first listed category tag
    example = dict(categories =  ['Physics Archive->astro-ph->astro-ph.SR'])
    result = get_category_tags(example, tag_dict)
    expected = dict(category = 'astro-ph.SR')
    assert result == expected 

def test_get_category_tags_double_repeatingsubstring(tag_dict) -> None:
    # This tests when the categories column in the dataset
    # contains two entries each one being different. The 
    # function should  return the first listed category tag
    example = dict(categories =  ['Physics Archive->astro-ph->astro-ph.CO', 'Physics Archive->astro-ph->astro-ph.SR'])
    result = get_category_tags(example, tag_dict)
    expected = dict(category = 'astro-ph.CO')
    assert result == expected 

def test_get_category_tags_double_repeatingsubstring_missing(tag_dict) -> None:
    # This tests when the categories column in the dataset
    # contains two entries each one being different. The 
    # function should  return the first listed category tag
    example = dict(categories =  ['Physics Archive->astro-ph->astro-ph.XY', 'Physics Archive->astro-ph->astro-ph.SR'])
    result = get_category_tags(example, tag_dict)
    expected = dict(category = 'astro-ph.SR')
    assert result == expected

def test_get_category_tags_triple(tag_dict) -> None:
    # This tests when the categories column in the dataset
    # contains two entries each one being different. The 
    # function should  return the first listed category tag
    example = dict(categories =  ['Physics Archive->gr-qc',
  'Physics Archive->hep->hep-th',
  'Physics Archive->quant-ph'])
    result = get_category_tags(example, tag_dict)
    expected = dict(category = 'gr-qc')
    assert result == expected


 





    
