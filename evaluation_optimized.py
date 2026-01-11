"""
wrapper for optimized versions of the evaluation framework, utilized by parallelization schemes elsewhere in codebase
"""
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
import hashlib
import pickle
from pathlib import Path


class EncodingCache:
    """
    cache encodings in order to reduce time spent retraining
    """
    def __init__(self, cache_dir: Path = None, max_size: int = 10000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self._cache = {}
        self._hits = 0
        self._misses = 0
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / 'encoding_cache.pkl'
            self._load_cache()
    
    def _get_key(self, text: str, tokenizer_id: str, use_dropout: bool = False) -> str:
        """make a cache key from text and tokenizer."""
        content = f"{tokenizer_id}:dropout_{use_dropout}:{text}"
        try:
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        except (UnicodeEncodeError, UnicodeDecodeError):
            return hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()
    
    def _load_cache(self):
        if self.cache_dir and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"  Loaded {len(self._cache)} cached encodings")
            except Exception as e:
                print(f"  error - couldn't load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        if self.cache_dir:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self._cache, f)
            except Exception as e:
                print(f"  error: couldn't save cache: {e}")
    
    def get(self, text: str, tokenizer_id: str, encoder_func, use_dropout: bool = False):
        """
        gets the encoding from cache or, if is empty, compute and cache it.
        """
        key = self._get_key(text, tokenizer_id, use_dropout)
        
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        else:
            self._misses += 1
            tokens = encoder_func(text)
            
            if len(self._cache) < self.max_size:
                self._cache[key] = tokens
            elif self.miss_rate > 0.5 and self._cache:
                #FIFO regime (TODO probably could use better cache hit-miss algo)
                first_key = next(iter(self._cache))
                del self._cache[first_key]
                self._cache[key] = tokens
            
            return tokens
    
    def batch_encode(self, texts: List[str], tokenizer_id: str, encoder_func, use_dropout: bool = False):
        results = []
        for text in texts:
            results.append(self.get(text, tokenizer_id, encoder_func, use_dropout))
        return results
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate
    
    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        if self.cache_dir and self.cache_file.exists():
            self.cache_file.unlink()
    
    def save(self):
        self._save_cache()
    
    def stats(self) -> Dict:
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate
        }


def batch_encode_texts(texts: List[str], tokenizer, use_dropout: bool = False) -> List[List[int]]:
    return [tokenizer.encode(text, dropout=use_dropout) for text in texts]


class OptimizedTokenizerEvaluator:
    def __init__(self, tokenizer, test_texts: List[str], 
                 cache: EncodingCache = None,
                 tokenizer_id: str = None,
                 use_dropout: bool = False,
                 random_seed: int = 42):

        if not test_texts or len(test_texts) == 0:
            raise ValueError("test_texts cannot be empty")
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        
        self.tokenizer = tokenizer
        self.test_texts = test_texts
        self.cache = cache
        self.tokenizer_id = tokenizer_id or str(id(tokenizer))
        self.use_dropout = use_dropout
        
        if use_dropout:
            np.random.seed(random_seed)
            import random
            random.seed(random_seed)

        def encode_text(text: str):
            """Encode a single text - defined as function for multiprocessing compatibility."""
            return self.tokenizer.encode(text, dropout=self.use_dropout)

        if self.cache:
            self._encoded_texts = self.cache.batch_encode(
                test_texts,
                self.tokenizer_id,
                encode_text,
                use_dropout=self.use_dropout
            )
        else:
            self._encoded_texts = batch_encode_texts(test_texts, tokenizer, use_dropout=self.use_dropout)
        
        #precompute all of the tokens for statistics
        self._all_tokens = []
        for tokens in self._encoded_texts:
            self._all_tokens.extend(tokens)

    #all of the following use the precomputed tokens
    
    def compute_renyi_entropy(self, alpha: float = 2.5) -> float:
        token_counts = Counter(self._all_tokens)
        total = sum(token_counts.values())
        probabilities = [count / total for count in token_counts.values()]
        
        if alpha == 1.0:
            return -sum(p * np.log2(p) for p in probabilities if p > 0)
        else:
            sum_p_alpha = sum(p ** alpha for p in probabilities)
            return (1.0 / (1 - alpha)) * np.log2(sum_p_alpha)
    
    def compute_nsl(self) -> float:
        total_tokens = len(self._all_tokens)
        total_chars = sum(len(text) for text in self.test_texts)
        return total_tokens / total_chars if total_chars > 0 else 0.0
    
    def compute_fertility(self) -> float:
        total_tokens = len(self._all_tokens)
        total_words = sum(len(text.split()) for text in self.test_texts)
        return total_tokens / total_words if total_words > 0 else 0.0
    
    def compute_zipfian_alignment(self) -> float:
        from scipy import stats
        
        token_counts = Counter(self._all_tokens)
        
        if len(token_counts) < 2:
            return 0.0
        
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        frequencies = [count for _, count in sorted_tokens]
        ranks = list(range(1, len(frequencies) + 1))
        
        log_freq = np.log(frequencies)
        log_rank = np.log(ranks)
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_rank, log_freq)
            r_squared = r_value ** 2
            return max(0.0, min(1.0, r_squared))
        except (ValueError, np.linalg.LinAlgError):
            return 0.0
    
    def compute_token_distribution_stats(self) -> Dict:
        token_counts = Counter(self._all_tokens)
        counts = list(token_counts.values())
        
        def _gini_coefficient(frequencies: List[int]) -> float:
            sorted_freq = sorted(frequencies)
            n = len(sorted_freq)
            if n == 0 or sum(sorted_freq) == 0:
                return 0.0
            cumsum = np.cumsum(sorted_freq)
            return (2 * np.sum((i + 1) * freq for i, freq in enumerate(sorted_freq))) / (n * cumsum[-1]) - (n + 1) / n
        
        unique_tokens_used = len(token_counts)
        total_tokens = len(self._all_tokens)
        
        #get the vocabulary size from tokenizer
        vocab_size = 0
        if hasattr(self.tokenizer, 'vocab') and self.tokenizer.vocab:
            vocab_size = len(self.tokenizer.vocab)
        elif hasattr(self.tokenizer, 'vocab_size'):
            vocab_size = self.tokenizer.vocab_size
        
        #calc vocabulary usage percentage
        vocab_usage_pct = (unique_tokens_used / vocab_size * 100) if vocab_size > 0 else 0.0
        
        return {
            'unique_tokens_used': unique_tokens_used,
            'total_tokens': total_tokens,
            'vocab_size': vocab_size,
            'vocab_usage_percentage': vocab_usage_pct,
            'mean_frequency': np.mean(counts) if counts else 0.0,
            'std_frequency': np.std(counts) if counts else 0.0,
            'max_frequency': max(counts) if counts else 0,
            'min_frequency': min(counts) if counts else 0,
            'gini_coefficient': _gini_coefficient(counts),
        }
    
    def evaluate_all(self, morphological_boundaries: Dict[str, List[int]] = None) -> Dict:
        """
        runs all intrinsic evaluations (optimized version, v2)"""

        results = {
            'renyi_entropy': self.compute_renyi_entropy(),
            'nsl': self.compute_nsl(),
            'fertility': self.compute_fertility(),
            'zipfian_alignment': self.compute_zipfian_alignment(),
            'distribution_stats': self.compute_token_distribution_stats(),
        }
        
        return results