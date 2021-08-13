from typing import Dict, List, Tuple

import pandas as pd

from tokenize_ner_sentences import tokenize_sentence


def adapt_input_data(yaml_file: dict) -> pd.DataFrame:
    """
    The DataLoader expects the tags to be a list of the same length as the text.
    Lists are converted to strings in order to be compatible with literal_eval in the Dataset
    """

    text_list = []
    tag_list = []

    for key, value in yaml_file.items():
        text = _convert_text(key)
        split_text, tag = _convert_tags(value)

        # Text and split_text must match
        assert text == split_text, "{} does not match {}".format(text, split_text)

        # Populate lists
        text_list.append(str(text))
        tag_list.append(str(tag))

    return pd.DataFrame(list(zip(text_list, tag_list)), columns=["text", "tag"])


def _convert_text(text: str) -> str:
    """
    Split sentence by spaces
    TODO: split also by apostrophes. It will be defined in an utility function that will also be used by search_data_processing.
    """

    return tokenize_sentence(text)


def _convert_tags(tags: List[Dict[str, str]]) -> Tuple[str, str]:

    """
    Split list of dicts in text and tag. We return split_text as a sanity check later on.
    We also convert words to lowercase, so we don;t need to lowercase all the evaluation datasets by hand.
    """
    split_text = []
    tag = []

    for element in tags:
        split_text.extend([word.lower() for word in element.keys()])
        tag.extend(list(element.values()))

    return split_text, tag
