"""
loads Talmudic data and analyzes basic stats
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import re
from config import Language


@dataclass
class TalmudSentence:
    text: str
    language: Language
    #see thesis for these definitions
    mesekhta: Optional[str] = None 
    daf: Optional[str] = None
    
    def with_language_tag(self) -> str:
        return f"<LID:{self.language.value}> {self.text}"


class TalmudCorpus:
    
    def __init__(self, corpus_path: str, classifier_path: Optional[str] = None,
                 normalize_text: bool = False,
                 remove_nekudot: bool = False,
                 remove_punctuation: bool = False,
                 keep_basic_punctuation: bool = False):
        
        self.corpus_path = Path(corpus_path)
        self.classifier_path = classifier_path
        self.normalize_text = normalize_text
        self.remove_nekudot = remove_nekudot
        self.remove_punctuation = remove_punctuation
        self.keep_basic_punctuation = keep_basic_punctuation
        self.sentences: List[TalmudSentence] = []
        self.language_counts: Dict[Language, int] = {lang: 0 for lang in Language}
        
    def load(self):
        print(f"Loading corpus from {self.corpus_path}")
        
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"corpus file not found at {self.corpus_path}")
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                #format: LANGUAGE|MESEKHTA|DAF|TEXT
                parts = line.split('|')
                if len(parts) == 4:
                    lang_str, mesekhta, daf, text = parts
                    try:
                        language = Language(lang_str)
                    except ValueError:
                        language = self._classify_language(text)
                else:
                    text = line
                    language = self._classify_language(text)
                    mesekhta = None
                    daf = None
                
                from text_normalization import clean_corpus_formatting, remove_nekudot, normalize_whitespace
                
                text = clean_corpus_formatting(text)
                text = remove_nekudot(text)
                text = normalize_whitespace(text)
                
                if self.normalize_text:
                    from text_normalization import normalize_text
                    text = normalize_text(
                        text,
                        remove_nekudot_flag=False,
                        remove_punctuation_flag=self.remove_punctuation,
                        keep_basic_punctuation=self.keep_basic_punctuation,
                        normalize_whitespace_flag=False 
                    )
                
                sentence = TalmudSentence(
                    text=text,
                    language=language,
                    mesekhta=mesekhta,
                    daf=daf
                )
                self.sentences.append(sentence)
                self.language_counts[language] += 1
        
        print(f"Loaded {len(self.sentences)} sentences")
        print("Language distribution:")
        for lang, count in self.language_counts.items():
            pct = 100 * count / len(self.sentences)
            print(f"  {lang.value}: {count} ({pct:.1f}%)")
    
    def _classify_language(self, text: str) -> Language:
        pass #TODO!!!
    
    def get_texts(self, with_language_tags: bool = False) -> List[str]:
        if with_language_tags:
            return [s.with_language_tag() for s in self.sentences]
        return [s.text for s in self.sentences]
    
    def get_texts_by_language(self) -> Dict[Language, List[str]]:
        result = {lang: [] for lang in Language}
        for sentence in self.sentences:
            result[sentence.language].append(sentence.text)
        return result
    
    def get_balanced_sample(self, aramaic_factor: float = 2.0) -> List[str]:
        """
        upsample aramaic by aramaic_factor times.
        """
        texts_by_lang = self.get_texts_by_language()
        result = []
        
        for lang in [Language.BIBLICAL_HEBREW, Language.MISHNAIC_HEBREW]:
            result.extend(texts_by_lang[lang])
        
        aramaic_texts = texts_by_lang[Language.JEWISH_BABYLONIAN_ARAMAIC]
        n_aramaic_samples = int(len(aramaic_texts) * aramaic_factor)
        
        if n_aramaic_samples > len(aramaic_texts):
            indices = np.random.choice(len(aramaic_texts), n_aramaic_samples, replace=True)
            result.extend([aramaic_texts[i] for i in indices])
        else:
            result.extend(aramaic_texts)
        
        np.random.shuffle(result)
        return result
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
              test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        texts = self.get_texts()
        n = len(texts)
        
        #seed lets us reproduce results
        np.random.seed(seed)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        test_texts = [texts[i] for i in test_idx]
        
        return train_texts, val_texts, test_texts, train_idx, val_idx, test_idx
    
    def compute_statistics(self) -> Dict:
        texts = self.get_texts()
        
        char_counts = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        unique_chars = set(''.join(texts))
        
        lang_stats = {}
        for lang in Language:
            lang_texts = [s.text for s in self.sentences if s.language == lang]
            if lang_texts:
                lang_stats[lang.value] = {
                    'count': len(lang_texts),
                    'avg_chars': np.mean([len(t) for t in lang_texts]),
                    'avg_words': np.mean([len(t.split()) for t in lang_texts]),
                }
        
        return {
            'total_sentences': len(texts),
            'total_characters': sum(char_counts),
            'total_words': sum(word_counts),
            'avg_chars_per_sentence': np.mean(char_counts),
            'std_chars_per_sentence': np.std(char_counts),
            'avg_words_per_sentence': np.mean(word_counts),
            'std_words_per_sentence': np.std(word_counts),
            'unique_characters': len(unique_chars),
            'language_distribution': lang_stats,
        }
    
    def save_splits(self, output_dir: Path, train_texts: List[str], 
                   val_texts: List[str], test_texts: List[str]):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, texts in [('train', train_texts), ('val', val_texts), ('test', test_texts)]:
            path = output_dir / f'{name}.txt'
            with open(path, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')
            print(f"Saved {len(texts)} sentences to {path}")


class FullTalmudCorpus:

    def __init__(self, 
                 tractates: Optional[List[str]] = None,
                 cache_dir: Optional[str] = None,
                 max_words_per_segment: int = 40,
                 min_words_per_segment: int = 3):

        self.tractates = tractates
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_words_per_segment = max_words_per_segment
        self.min_words_per_segment = min_words_per_segment
        self.texts: List[str] = []
        
    def load(self):
        print("Loading full Babylonian Talmud from Sefaria API...")
        
        from talmud_loader import prepare_talmud_corpus
        
        self.texts = prepare_talmud_corpus(
            tractates=self.tractates,
            max_words_per_segment=self.max_words_per_segment,
            min_words_per_segment=self.min_words_per_segment,
            cache_dir=self.cache_dir,
            use_cache=True,
            force_refresh=False
        )
        
        print(f"Loaded {len(self.texts)} text segments")
    
    def get_texts(self) -> List[str]:
        return self.texts
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
              test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
 
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        if not self.texts:
            raise ValueError(
                "Corpus has to be loaded before splitting; call corpus.load() first."
            )
        
        n = len(self.texts)
        
        #set seed so we can reproduce results
        np.random.seed(seed)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        train_texts = [self.texts[i] for i in train_idx]
        val_texts = [self.texts[i] for i in val_idx]
        test_texts = [self.texts[i] for i in test_idx]
        
        return train_texts, val_texts, test_texts, train_idx, val_idx, test_idx
    
    def compute_statistics(self) -> Dict:
        char_counts = [len(text) for text in self.texts]
        word_counts = [len(text.split()) for text in self.texts]
        
        unique_chars = set(''.join(self.texts))
        
        return {
            'total_sentences': len(self.texts),
            'total_characters': sum(char_counts),
            'total_words': sum(word_counts),
            'avg_chars_per_sentence': np.mean(char_counts),
            'std_chars_per_sentence': np.std(char_counts),
            'avg_words_per_sentence': np.mean(word_counts),
            'std_words_per_sentence': np.std(word_counts),
            'unique_characters': len(unique_chars),
        }


class CorpusStatistics:
    
    @staticmethod
    def compute_heaps_law_params(texts: List[str]) -> Tuple[float, float]:
        """
        V = K * N^beta where V is vocabulary size, N is corpus size. See thesis.
        """
        vocab_sizes = []
        corpus_sizes = []
        
        vocab = set()
        n_tokens = 0
        
        for text in texts:
            tokens = text.split()
            n_tokens += len(tokens)
            vocab.update(tokens)
            
            vocab_sizes.append(len(vocab))
            corpus_sizes.append(n_tokens)
        
        #implements Log-linear regression
        log_N = np.log(corpus_sizes)
        log_V = np.log(vocab_sizes)
        
        #use polyfit (log(V) = log(K) + beta * log(N))
        coeffs = np.polyfit(log_N, log_V, 1)
        beta = coeffs[0]
        K = np.exp(coeffs[1])
        
        return K, beta
    
    @staticmethod
    def compute_type_token_ratio(texts: List[str]) -> float:
        """vocabulary size over total tokens"""
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.split())
        
        types = len(set(all_tokens))
        tokens = len(all_tokens)
        
        return types / tokens if tokens > 0 else 0.0