"""
Base tokenizer interface/abstract class that all others inherit from
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
import json


class BaseTokenizer(ABC):
    """ABC for all tokenizers"""
    
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2):
        
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.is_trained = False
    
    @abstractmethod
    def train(self, texts: List[str], **kwargs):
        pass
    
    @abstractmethod
    def encode(self, text: str, dropout: bool = False) -> List[int]:
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        pass
    
    def encode_batch(self, texts: List[str], dropout: bool = False) -> List[List[int]]:
        return [self.encode(text, dropout=dropout) for text in texts]
    
    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in token_ids_batch]
    
    def save(self, path: Path):
        #saves tokenizer to disk
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'is_trained': self.is_trained,
            'class_name': self.__class__.__name__,
        }
        
        #saves additional algorithm data
        data.update(self._get_save_data())
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved tokenizer to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'BaseTokenizer':
        #pkl good enough for now TODO
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'], 
                       min_frequency=data['min_frequency'])
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = data['inverse_vocab']
        tokenizer.is_trained = data['is_trained']
        
        #loads algo specific data
        tokenizer._load_save_data(data)
        
        return tokenizer
    
    def _get_save_data(self) -> Dict:
        return {}
    
    def _load_save_data(self, data: Dict):
        pass
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def token_to_id(self, token: str) -> Optional[int]:
        return self.vocab.get(token)
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        return self.inverse_vocab.get(token_id)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        from text_normalization import clean_corpus_formatting, remove_nekudot, normalize_whitespace
        
        text = clean_corpus_formatting(text)
        text = remove_nekudot(text)
        text = normalize_whitespace(text)
        
        return text
    
    def compute_compression_ratio(self, texts: List[str]) -> float:
        total_tokens = 0
        total_chars = 0
        
        for text in texts:
            tokens = self.encode(text)
            total_tokens += len(tokens)
            total_chars += len(text)
        
        return total_tokens / total_chars if total_chars > 0 else 0.0
    
    def compute_fertility(self, texts: List[str]) -> float:
        total_tokens = 0
        total_words = 0
        
        for text in texts:
            tokens = self.encode(text)
            words = text.split()
            total_tokens += len(tokens)
            total_words += len(words)
        
        return total_tokens / total_words if total_words > 0 else 0.0


class TokenizerWrapper:
    def __init__(self, base_tokenizer: BaseTokenizer, strategy: str):
        self.base_tokenizer = base_tokenizer
        self.strategy = strategy
        self.language_tokenizers: Optional[Dict[str, BaseTokenizer]] = None
    
    def train(self, texts: List[str], language_labels: Optional[List[str]] = None, **kwargs):
        texts = [self.base_tokenizer.preprocess_text(text) for text in texts]
        
        if self.strategy == 'unified':
            self.base_tokenizer.train(texts, **kwargs)
        
        elif self.strategy == 'partitioned':
            if language_labels is None:
                raise ValueError("Partitioned strategy requires language_labels")

            from config import Language
            texts_by_lang = {lang.value: [] for lang in Language}
            for text, label in zip(texts, language_labels):
                texts_by_lang[label].append(text)

            if Language.BIBLICAL_HEBREW.value in texts_by_lang:
                try:
                    from biblical_text_loader import get_biblical_text_enrichment
                    biblical_enrichment = get_biblical_text_enrichment(
                        source='both',
                        filepath=None,
                        sefaria_books=None
                    )
                    if biblical_enrichment:
                        biblical_enrichment = [self.base_tokenizer.preprocess_text(text) for text in biblical_enrichment]
                        original_count = len(texts_by_lang[Language.BIBLICAL_HEBREW.value])
                        texts_by_lang[Language.BIBLICAL_HEBREW.value].extend(biblical_enrichment)
                        enriched_count = len(texts_by_lang[Language.BIBLICAL_HEBREW.value])
                        print(f"more BH training data: {original_count} → {enriched_count} segments (+{enriched_count - original_count} from Bible)")
                except Exception as e:
                    print(f"couldnt enrich BH data: {e}")
            
            self.language_tokenizers = {}
            for lang, lang_texts in texts_by_lang.items():
                if lang_texts:
                    lang_texts = [self.base_tokenizer.preprocess_text(text) for text in lang_texts]
                    tokenizer = self.base_tokenizer.__class__(
                        vocab_size=self.base_tokenizer.vocab_size,
                        min_frequency=self.base_tokenizer.min_frequency
                    )
                    tokenizer.train(lang_texts, **kwargs)
                    self.language_tokenizers[lang] = tokenizer
        
        elif self.strategy == 'language_informed':
            #this essentially prepends language tags and train unified. texts are already preprocessed above
            tagged_texts = []
            for text, label in zip(texts, language_labels):
                tagged_texts.append(f"<LID:{label}> {text}")
            self.base_tokenizer.train(tagged_texts, **kwargs)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def encode(self, text: str, language: Optional[str] = None, dropout: bool = False) -> List[int]:
        if self.strategy == 'unified':
            return self.base_tokenizer.encode(text, dropout=dropout)
        
        elif self.strategy == 'partitioned':
            if language is None:
                raise ValueError("Beep. partitioned strategy requires language parameter")
            return self.language_tokenizers[language].encode(text, dropout=dropout)
        
        elif self.strategy == 'language_informed':
            if language:
                text = f"<LID:{language}> {text}"
            return self.base_tokenizer.encode(text, dropout=dropout)
    
    def decode(self, token_ids: List[int], language: Optional[str] = None) -> str:
        if self.strategy == 'unified':
            return self.base_tokenizer.decode(token_ids)
        
        elif self.strategy == 'partitioned':
            if language is None:
                raise ValueError("beep. Partitioned strategy requires language parameter")
            return self.language_tokenizers[language].decode(token_ids)
        
        elif self.strategy == 'language_informed':
            decoded = self.base_tokenizer.decode(token_ids)
            #get rid of lang tag
            import re
            decoded = re.sub(r'<LID:[A-Z]+>\s*', '', decoded)
            return decoded
