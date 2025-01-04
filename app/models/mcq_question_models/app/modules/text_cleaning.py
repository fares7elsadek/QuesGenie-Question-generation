import re

def clean_text(text: str) -> str:
    cleaned_text = _remove_brackets(text)
    cleaned_text = _remove_square_brackets(cleaned_text)
    cleaned_text = _remove_multiple_spaces(cleaned_text)
    cleaned_text = _replace_weird_hyphen(cleaned_text)
    return cleaned_text
    

def _remove_brackets(text: str) -> str:
    return re.sub(r'\((.*?)\)', lambda L: '', text)


def _remove_square_brackets(text: str) -> str:
    return re.sub(r'\[(.*?)\]', lambda L: '', text)


def _remove_multiple_spaces(text: str) -> str:
    return re.sub(' +', ' ', text)


def _replace_weird_hyphen(text: str) -> str:
    return text.replace('–', '-')