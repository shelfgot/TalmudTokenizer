import re
from typing import Optional


def remove_nekudot(text: str) -> str:
    nekudot_pattern = r'[\u0591-\u05C7]'
    return re.sub(nekudot_pattern, '', text)


def remove_punctuation(text: str, keep_basic: bool = False) -> str:
    if keep_basic:
        #remove most punctuation, yet keep basic sentence/clause markers; !?;:()[]{}""'' etc.
        punct_pattern = r'[!?;:()[\]{}"\'`]'
        return re.sub(punct_pattern, '', text)
    else:
        #,. removed
        punct_pattern = r'[^\w\s]'
        return re.sub(punct_pattern, '', text)


def clean_corpus_formatting(text: str) -> str:
    #removal of::
    #parentheses around words
    text = re.sub(r'\(([^)]+)\)', r'\1', text)
    
    #brackets
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    #for last char...
    return text.strip()


def normalize_text(text: str, 
                   remove_nekudot_flag: bool = False,
                   remove_punctuation_flag: bool = False,
                   keep_basic_punctuation: bool = False,
                   normalize_whitespace_flag: bool = True) -> str:
    result = text
    
    if remove_nekudot_flag:
        result = remove_nekudot(result)
    
    if remove_punctuation_flag:
        result = remove_punctuation(result, keep_basic=keep_basic_punctuation)
    
    if normalize_whitespace_flag:
        result = normalize_whitespace(result)
    
    return result


def batch_normalize(texts: list, **kwargs) -> list:
    return [normalize_text(text, **kwargs) for text in texts]
