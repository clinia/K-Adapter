import re
from typing import List, Optional


def tokenize_sentence(text: str) -> List[str]:

    """
    TODO: [SEAR-756] split by apostrophes too, but only when they are at the beggining of the word.
    """

    text = text.lower()
    text = re.split("([\w\-']+)", text, flags=re.UNICODE)  # Keeps dashes and apostrophes in the token
    text = [clean_token for clean_token in (process_token(token) for token in text) if clean_token is not None]

    return text


def process_token(token: str) -> Optional[str]:
    token = token.strip()
    token = token if (token != " ") and (token != "") else None

    return token
