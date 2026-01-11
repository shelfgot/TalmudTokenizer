"""
Embedding-based downstream evaluation for tokenizers.

This module trains word embeddings on tokenized corpora and evaluates
the quality of the resulting semantic spaces through various tasks.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import hashlib
import random
try:
    from tqdm.auto import tqdm  # Auto-detects notebook vs terminal
except ImportError:
    from tqdm import tqdm


class SimpleWord2Vec:
    """
    a simple Word2Vec implementation; uses skip-gram with negative sampling.
    (going to use more comments bc I am very out of my depth here)
    """
    def __init__(self, embedding_dim: int = 100, window_size: int = 5,
                 negative_samples: int = 15, learning_rate: float = 0.01):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        
        #for core model state
        self.vocab: Dict[str, int] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.context_embeddings: Optional[np.ndarray] = None
        
        #for sampling statistics
        self.token_counts: Dict[str, int] = {}
        self.token_freqs: Optional[np.ndarray] = None 
        self.neg_sampling_cdf: Optional[np.ndarray] = None  
        
        #subsampling of very frequent tokens
        self.subsample_t: float = 1e-5
        self.use_subsampling: bool = True
        
        #SGD + momentum (thanks prof Alex Wong!!)
        self.use_momentum: bool = True
        self.momentum: float = 0.9
        self.center_momentum: Optional[np.ndarray] = None
        self.context_momentum: Optional[np.ndarray] = None
    
    def build_vocab(self, token_sequences: List[List[str]], min_count: int = 1):
        """make vocabulary from token sequences."""
        special_tokens = {'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'}
        
        token_counts = defaultdict(int)
        for sequence in token_sequences:
            for token in sequence:
                if token and token not in special_tokens:
                    token_counts[token] += 1
        
        filtered_tokens = [(token, count) for token, count in sorted(token_counts.items()) if count >= min_count]
        
        #map tokens -> index (0-indexed and sequential)
        self.vocab = {token: idx for idx, (token, count) in enumerate(filtered_tokens)}
        
        if len(self.vocab) == 0:
            raise ValueError("No tokens found after filtering. Try lowering min_count or check tokenization.")
        
        vocab_size = len(self.vocab)
        #the smaller initialization range helps prevent early overflow
        limit = 0.5 / self.embedding_dim
        self.embeddings = np.random.uniform(-limit, limit, (vocab_size, self.embedding_dim)).astype(np.float32)
        self.context_embeddings = np.random.uniform(-limit, limit, (vocab_size, self.embedding_dim)).astype(np.float32)
        
        #init momentum buffers (for optimizer) to zeros
        if self.use_momentum:
            self.center_momentum = np.zeros_like(self.embeddings)
            self.context_momentum = np.zeros_like(self.context_embeddings)
        
        #we store counts aligned with vocab indexes + compute relative frequencies
        self.token_counts = {token: token_counts[token] for token in self.vocab.keys()}
        total_count = float(sum(self.token_counts.values()))
        if total_count > 0:
            freqs = np.zeros(vocab_size, dtype=np.float64)
            for token, idx in self.vocab.items():
                freqs[idx] = self.token_counts[token] / total_count
            self.token_freqs = freqs
        else:
            self.token_freqs = None
        
        #unigram^0.75 cdf for negative sampling
        if self.token_freqs is not None:
            adjusted = np.power(self.token_freqs, 0.75)
            total_adjusted = adjusted.sum()
            if total_adjusted > 0:
                neg_probs = adjusted / total_adjusted
                self.neg_sampling_cdf = np.cumsum(neg_probs)
            else:
                self.neg_sampling_cdf = None
        else:
            self.neg_sampling_cdf = None
        
        print(f"Built vocabulary: {vocab_size} tokens")
    
    def _sample_negative(self) -> int:
        """
        samples negative token index using unigram^0.75 distribution
        if statistics are unavailable, just performs uniform IID sampling
        """
        if self.neg_sampling_cdf is None or len(self.neg_sampling_cdf) == 0:
            return np.random.randint(0, len(self.vocab))
        
        r = np.random.rand()
        idx = int(np.searchsorted(self.neg_sampling_cdf, r, side="right"))
        if idx >= len(self.vocab):
            idx = len(self.vocab) - 1
        return idx
    
    def train(self, token_sequences: List[List[str]], epochs: int = 5, 
              early_stopping_patience: Optional[int] = None):
    
        print(f"Training Word2Vec for {epochs} epochs...")
        
        best_loss = float('inf')
        patience_counter = 0
        min_delta = 1e-4 
        #i.e. improvement
        
        epoch_bar = tqdm(range(epochs), desc="Training embeddings", unit="epoch")
        for epoch in epoch_bar:
            #LR decay and min learning rate
            self.learning_rate = self.initial_learning_rate * (1.0 - epoch / epochs)
            self.learning_rate = max(0.0001, self.learning_rate)
            
            total_loss = 0
            n_samples = 0
            
            sequence_bar = tqdm(token_sequences, desc=f"  └─ Sequences in epoch", 
                               unit="seq", leave=False, mininterval=0.5)
            for sequence in sequence_bar:
                indices = []
                for token in sequence:
                    if token in self.vocab:
                        idx = self.vocab[token]
                        if not (0 <= idx < len(self.vocab)):
                            continue
                        
                        if self.use_subsampling and self.token_freqs is not None:
                            freq = self.token_freqs[idx]
                            if freq > 0:
                                #P(discard) = 1 - sqrt(t / f)
                                prob_discard = 1.0 - np.sqrt(self.subsample_t / freq)
                                if prob_discard > 0 and np.random.rand() < prob_discard:
                                    continue
                        
                        indices.append(idx)
                
                if len(indices) < 2:
                    continue
                
                #now for the skipgram training
                for i, center_idx in enumerate(indices):
                    #we need center_idx is in bounds
                    if not (0 <= center_idx < len(self.vocab)):
                        continue
                    
                    #find the context words
                    start = max(0, i - self.window_size)
                    end = min(len(indices), i + self.window_size + 1)
                    
                    for j in range(start, end):
                        if i == j:
                            continue
                        
                        context_idx = indices[j]
                        
                        #show context_idx is in bounds
                        if not (0 <= context_idx < len(self.vocab)):
                            continue
                        
                        #get positive sample
                        loss = self._train_pair(center_idx, context_idx, label=1)
                        total_loss += loss
                        n_samples += 1
                        
                        # Negative samples (unigram^0.75)
                        for _ in range(self.negative_samples):
                            neg_idx = self._sample_negative()
                            loss = self._train_pair(center_idx, neg_idx, label=0)
                            total_loss += loss
                            n_samples += 1
                
                # Update sequence progress bar with current loss (throttled by mininterval)
                if n_samples > 0:
                    current_loss = total_loss / n_samples
                    sequence_bar.set_postfix({"loss": f"{current_loss:.4f}", "samples": n_samples})
            
            avg_loss = total_loss / n_samples if n_samples > 0 else 0
            epoch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Early stopping logic
            if early_stopping_patience is not None:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        epoch_bar.write(f"  Early stopping: loss hasn't improved for {early_stopping_patience} epochs")
                        break
    
    def _train_pair(self, center_idx: int, context_idx: int, label: int) -> float:
        """Train on a single '(center, context)' pair."""
        #check bounds and skip invalid parts
        vocab_size = len(self.vocab)
        if not (0 <= center_idx < vocab_size) or not (0 <= context_idx < vocab_size):
            return 0.0
        
        #check for corruption in embedding vals, i.e. NaN/inf
        center_emb = self.embeddings[center_idx].copy()
        context_emb = self.context_embeddings[context_idx].copy()
        
        #replace Nan/Inf w/0s
        if np.any(np.isnan(center_emb)) or np.any(np.isinf(center_emb)):
            center_emb = np.nan_to_num(center_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            self.embeddings[center_idx] = center_emb
        
        if np.any(np.isnan(context_emb)) or np.any(np.isinf(context_emb)):
            context_emb = np.nan_to_num(context_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            self.context_embeddings[context_idx] = context_emb
        
        #clipping for dot products so no overflow by setting max L2 norm
        max_emb_norm = 10.0
        center_norm = np.linalg.norm(center_emb)
        if center_norm > max_emb_norm:
            center_emb = center_emb * (max_emb_norm / center_norm)
            self.embeddings[center_idx] = center_emb
        
        context_norm = np.linalg.norm(context_emb)
        if context_norm > max_emb_norm:
            context_emb = context_emb * (max_emb_norm / context_norm)
            self.context_embeddings[context_idx] = context_emb
        
        #Now for the forwards and the backwards passes:
        #forwards
        score = np.dot(center_emb, context_emb)
        #clip again and then use sigmoid and calculate loss
        score = np.clip(score, -10.0, 10.0)
        pred = 1.0 / (1.0 + np.exp(-score))
        loss = -label * np.log(pred + 1e-10) - (1 - label) * np.log(1 - pred + 1e-10)
        
        #backwards pass
        grad = pred - label
        
        #clip gradient, once again, to prevent large updates
        grad = np.clip(grad, -1.0, 1.0)
        
        #calculate SGD update direction
        update_center = self.learning_rate * grad * context_emb
        update_context = self.learning_rate * grad * center_emb
        
        #use momentum for the optimizer
        if self.use_momentum:
            if self.center_momentum is None or self.context_momentum is None:
                vocab_size = len(self.vocab)
                self.center_momentum = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
                self.context_momentum = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
            
            self.center_momentum[center_idx] = (
                self.momentum * self.center_momentum[center_idx] + update_center
            )
            self.context_momentum[context_idx] = (
                self.momentum * self.context_momentum[context_idx] + update_context
            )
            
            center_update = self.center_momentum[center_idx]
            context_update = self.context_momentum[context_idx]
        else:
            center_update = update_center
            context_update = update_context
        
        # clip update magnitude for stability
        center_update_norm = np.linalg.norm(center_update)
        if center_update_norm > 1.0:
            center_update = center_update * (1.0 / center_update_norm)
        
        context_update_norm = np.linalg.norm(context_update)
        if context_update_norm > 1.0:
            context_update = context_update * (1.0 / context_update_norm)
        
        self.embeddings[center_idx] -= center_update
        self.context_embeddings[context_idx] -= context_update
        
        #after update ensure no Nan/Inf
        self.embeddings[center_idx] = np.nan_to_num(
            self.embeddings[center_idx], nan=0.0, posinf=1.0, neginf=-1.0
        )
        self.context_embeddings[context_idx] = np.nan_to_num(
            self.context_embeddings[context_idx], nan=0.0, posinf=1.0, neginf=-1.0
        )
        
        return loss
    
    def get_embedding(self, token: str) -> Optional[np.ndarray]:
        """Get embedding for a token in word2vec"""
        if token not in self.vocab:
            return None
        return self.embeddings[self.vocab[token]]
    
    def _find_fuzzy_match(self, query: str) -> Optional[str]:
        """
        Find token in vocab by removing nekudot (Hebrew diacritics) if exact match fails"""
        if query in self.vocab:
            return query
        
        #remove nekudot for fuzzy matching
        try:
            from text_normalization import remove_nekudot
            query_no_nekudot = remove_nekudot(query)
            
            #check tokens that match after removing nekudot
            for token in self.vocab.keys():
                token_no_nekudot = remove_nekudot(token)
                if query_no_nekudot == token_no_nekudot and query_no_nekudot:
                    return token
        except ImportError:
            pass
        
        return None
    
    def get_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """
        suggestions for "close-enough" words
        """
        suggestions = []
        
        fuzzy_match = self._find_fuzzy_match(query)
        if fuzzy_match:
            suggestions.append(fuzzy_match)
        
        #query as substring of another word, or vice versa?
        query_no_nekudot = query
        try:
            from text_normalization import remove_nekudot
            query_no_nekudot = remove_nekudot(query)
        except ImportError:
            pass
        
        for token in self.vocab.keys():
            if len(suggestions) >= max_suggestions:
                break
                
            if token == query or token in suggestions:
                continue
            
            token_no_nekudot = token
            try:
                from text_normalization import remove_nekudot
                token_no_nekudot = remove_nekudot(token)
            except ImportError:
                pass
            
            if query_no_nekudot and token_no_nekudot:
                if query_no_nekudot in token_no_nekudot or token_no_nekudot in query_no_nekudot:
                    if token not in suggestions:
                        suggestions.append(token)
        
        return suggestions[:max_suggestions]
    
    def most_similar(self, token: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        give most similar token to given token
        """
        actual_token = token
        if actual_token not in self.vocab:
            fuzzy_match = self._find_fuzzy_match(token)
            if fuzzy_match:
                actual_token = fuzzy_match
            else:
                return []
        
        query_emb = self.get_embedding(actual_token)
        if query_emb is None:
            return []
        
        #clean query embedding
        query_emb = np.nan_to_num(query_emb, nan=0.0, posinf=0.0, neginf=0.0)
        
        #norm of query embedding for cos similarity
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb = query_emb / query_norm
        
        #use vector to normalize (better than loop)
        all_embeddings = self.embeddings
        
        all_embeddings = np.nan_to_num(all_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 
        all_embeddings_norm = all_embeddings / norms
        
        #dot prod = cosine similarity
        similarities_vec = np.dot(all_embeddings_norm, query_emb)
        
        similarities = []
        query_idx = self.vocab.get(actual_token)
        for token, idx in self.vocab.items():
            if idx == query_idx:
                continue
            sim = float(similarities_vec[idx])
            if not np.isnan(sim) and not np.isinf(sim):
                similarities.append((token, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analogy(self, a: str, b: str, c: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve analogy, i.e. question of form a is to b as c is to ?
        returns a List of (token, score) tuples
        """
        if not all(t in self.vocab for t in [a, b, c]):
            return []
        
        #theoretically the embedding should follow vector arithmetic: b - a + c
        emb_a = self.get_embedding(a)
        emb_b = self.get_embedding(b)
        emb_c = self.get_embedding(c)
        #note: this, to me, Shlomi Helfgot, is absolute magic. it's crazy that any of the analogies are correct!
        target = emb_b - emb_a + emb_c
        
        #get closest tokens
        similarities = []
        for token, idx in self.vocab.items():
            if token in [a, b, c]:
                continue
            
            emb = self.embeddings[idx]
            sim = 1 - cosine(target, emb)
            similarities.append((token, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, filepath: Path) -> None:
        import pickle
        save_data = {
            'embedding_dim': self.embedding_dim,
            'window_size': self.window_size,
            'negative_samples': self.negative_samples,
            'initial_learning_rate': self.initial_learning_rate,
            'vocab': self.vocab,
            'embeddings': self.embeddings,
            'context_embeddings': self.context_embeddings,
            'token_counts': self.token_counts,
            'token_freqs': self.token_freqs,
            'neg_sampling_cdf': self.neg_sampling_cdf,
            'subsample_t': self.subsample_t,
            'use_subsampling': self.use_subsampling,
            'momentum': self.momentum,
            'use_momentum': self.use_momentum,
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved Word2Vec model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'SimpleWord2Vec':
        import pickle
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        model = cls(
            embedding_dim=save_data.get('embedding_dim', 100),
            window_size=save_data.get('window_size', 5),
            negative_samples=save_data.get('negative_samples', 5),
            learning_rate=save_data.get('initial_learning_rate', 0.01)
        )
        
        #restore core state, optimizer state, momentum, etc
        model.vocab = save_data['vocab']
        model.embeddings = save_data['embeddings']
        model.context_embeddings = save_data['context_embeddings']
        
        model.token_counts = save_data.get('token_counts', {})
        model.token_freqs = save_data.get('token_freqs', None)
        model.neg_sampling_cdf = save_data.get('neg_sampling_cdf', None)
        model.subsample_t = save_data.get('subsample_t', 1e-5)
        model.use_subsampling = save_data.get('use_subsampling', True)
        model.momentum = save_data.get('momentum', 0.9)
        model.use_momentum = save_data.get('use_momentum', True)
        
        #reinitialize momentum buffers
        vocab_size = len(model.vocab)
        if model.use_momentum:
            model.center_momentum = np.zeros((vocab_size, model.embedding_dim), dtype=np.float32)
            model.context_momentum = np.zeros((vocab_size, model.embedding_dim), dtype=np.float32)
        else:
            model.center_momentum = None
            model.context_momentum = None
        
        print(f"Word2Vec model was loaded from {filepath} (vocab size: {len(model.vocab)})")
        return model


class EmbeddingEvaluator:
    """
    evaluates tokenizer quality with the above embedding-based tasks.
    """
    
    _tokenized_cache: Dict[str, List[List[str]]] = {}
    
    def __init__(self, tokenizer, texts: List[str], embedding_dim: int = 100,
                 tokenizer_id: Optional[str] = None, use_cache: bool = True,
                 pre_tokenized_path: Optional[Path] = None):
        self.tokenizer = tokenizer
        self.texts = texts
        self.embedding_dim = embedding_dim
        self.model: Optional[SimpleWord2Vec] = None
        self.tokenizer_id = tokenizer_id or (str(id(tokenizer)) if tokenizer else "pre_tokenized")
        self.use_cache = use_cache
        self._cached_token_sequences: Optional[List[List[str]]] = None
        self.pre_tokenized_path = Path(pre_tokenized_path) if pre_tokenized_path else None
    
    #now we need to map surface-form words to embedding vocab tokens
    def _surface_to_vocab_token(self, word: str) -> Optional[str]:
        """
        we need to map to a word extant in the vocab
        """
        if not self.model or not getattr(self.model, "vocab", None):
            return None

            
        if word in self.model.vocab:
            return word

        #or get tokenizer to get token strings, then try to match those
        token_ids = self.tokenizer.encode(word) if hasattr(self, "tokenizer") else []
        token_strs: List[str] = []
        for tid in token_ids:
            token_str = (
                self.tokenizer.inverse_vocab.get(tid)
                if hasattr(self.tokenizer, "inverse_vocab")
                else self.tokenizer.id_to_token(tid)
            )
            if token_str:
                token_strs.append(token_str)

        #ANY token string which already exists in vocab is preferred
        for tok in token_strs:
            if tok in self.model.vocab:
                return tok

        # for WordPiece / TM, try stripping leading markers like '▁' or '##'
        for tok in token_strs:
            stripped = tok.lstrip("▁")
            if stripped.startswith("##"):
                stripped = stripped[2:]
            if stripped in self.model.vocab:
                return stripped

        #fuzzy matching
        if hasattr(self.model, "_find_fuzzy_match"):
            candidate = self.model._find_fuzzy_match(word)
            if candidate in self.model.vocab:
                return candidate

        return None

    def _get_embedding_for_surface(self, word: str) -> Optional[np.ndarray]:
        """wrapper over surface-to-vocab-token"""
        if not self.model:
            return None
        token = self._surface_to_vocab_token(word)
        if token is None:
            return None
        return self.model.get_embedding(token)
    
    def _get_tokenized_sequences(self, max_samples: Optional[int] = None) -> List[List[str]]:
        if self.pre_tokenized_path and self.pre_tokenized_path.exists():
            print(f"Loading pre-tokenized sequences from {self.pre_tokenized_path}...")
            import pickle
            try:
                with open(self.pre_tokenized_path, 'rb') as f:
                    save_data = pickle.load(f)
                
                token_sequences = save_data['token_sequences']
                print(f" loaded {len(token_sequences)} pre-tokenized sequences")
                print(f"  tokenizer ID: {save_data.get('tokenizer_id', 'unknown')}")
                print(f"  amount of original texts: {save_data.get('num_texts', 'unknown')}")
                
                if max_samples and len(token_sequences) > max_samples:
                    import random
                    token_sequences = random.sample(token_sequences, max_samples)
                    print(f"   Subsampled to: {len(token_sequences)} sequences")
                
                return token_sequences
            except Exception as e:
                print(f" warning: Could not load pre-tokenized sequences: {e}")
                print(f"   Falling back to tokenization...")
        elif self.pre_tokenized_path:
            print(f"error: Pre-tokenized path {self.pre_tokenized_path} does not exist")
            print(f"   Falling back to tokenization...")
        
        
        texts_to_tokenize = self.texts
        if max_samples and len(texts_to_tokenize) > max_samples:
            pass
        
        cache_key = None
        if self.use_cache:
            corpus_hash = hashlib.md5(('\n'.join(self.texts[:100]) if len(self.texts) > 100 else '\n'.join(self.texts)).encode('utf-8')).hexdigest()[:16]
            cache_key = f"{self.tokenizer_id}:{len(self.texts)}:{corpus_hash}:{max_samples or 'all'}"
            
            if cache_key in EmbeddingEvaluator._tokenized_cache:
                print(f"Using cached tokenized sequences ({len(EmbeddingEvaluator._tokenized_cache[cache_key])} sequences)")
                return EmbeddingEvaluator._tokenized_cache[cache_key]
        
        if max_samples and len(texts_to_tokenize) > max_samples:
            texts_to_tokenize = random.sample(texts_to_tokenize, max_samples)
            print(f"Subsampling corpus: {len(texts_to_tokenize)}/{len(self.texts)} texts")
        
        print(f"Tokenizing {len(texts_to_tokenize)} texts...")
        special_tokens = {'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'}
        
        token_sequences = []
        for text in tqdm(texts_to_tokenize, desc="Tokenizing", unit="text", leave=False):
            token_ids = self.tokenizer.encode(text, dropout=self.use_dropout)
            token_strs = []
            for tid in token_ids:
                token_str = (self.tokenizer.inverse_vocab.get(tid) if hasattr(self.tokenizer, 'inverse_vocab') 
                            else self.tokenizer.id_to_token(tid))
                if token_str and token_str not in special_tokens:
                    token_strs.append(token_str)
            if token_strs:
                token_sequences.append(token_strs)
        
        if self.use_cache and cache_key:
            if len(EmbeddingEvaluator._tokenized_cache) > 50:
                #as above, use FIFO (#TODO - make into better cache for hit miss)
                oldest_key = next(iter(EmbeddingEvaluator._tokenized_cache))
                del EmbeddingEvaluator._tokenized_cache[oldest_key]
            EmbeddingEvaluator._tokenized_cache[cache_key] = token_sequences
        
        return token_sequences
    
    def train_embeddings(self, epochs: int = 5, max_samples: Optional[int] = None, 
                        early_stopping_patience: Optional[int] = None,
                        embedding_save_path: Optional[Path] = None,
                        force_retrain: bool = False):
        """
        Train word embeddings on tokenized corpus."""

        if embedding_save_path and not force_retrain:
            embedding_path = Path(embedding_save_path)
            if embedding_path.exists():
                try:
                    print(f"Loading saved embeddings from {embedding_path}...")
                    self.model = SimpleWord2Vec.load(embedding_path)
                    print("Loaded saved embeddings (skipping training)")
                    return
                except Exception as e:
                    print(f"warning: Could not load saved embeddings: {e}")
                    print("   Falling back to training...")
        
        token_sequences = self._get_tokenized_sequences(max_samples=max_samples)
        
        if not token_sequences:
            print("error: No valid token sequences found after tokenization")
            return
        
        print(f"Training embeddings on {len(token_sequences)} sequences...")
        self.model = SimpleWord2Vec(
            embedding_dim=self.embedding_dim,
            negative_samples=15,
            window_size=5
        )
        self.model.build_vocab(token_sequences, min_count=1)
        self.model.train(token_sequences, epochs=epochs, early_stopping_patience=early_stopping_patience)
        
        if embedding_save_path:
            self.save_embeddings(Path(embedding_save_path))
    
    def save_embeddings(self, filepath: Path) -> None:
        if self.model is None:
            raise ValueError("No trained model to save. Train embeddings first.")
        
        self.model.save(filepath)
    
    def load_embeddings(self, filepath: Path) -> None:
        self.model = SimpleWord2Vec.load(filepath)
    
    def evaluate_similarity_task(self, word_pairs: List[Tuple[str, str, float]]) -> float:
        """
        evaluates on word similarity task and returns correlation (Spearman) with human judgments
        """
        if not self.model:
            raise ValueError("Must train embeddings first")
        
        model_scores = []
        human_scores = []
        
        for word1, word2, human_sim in word_pairs:
            emb1 = self.model.get_embedding(word1)
            emb2 = self.model.get_embedding(word2)
            
            if emb1 is None or emb2 is None:
                continue
            
            model_sim = 1 - cosine(emb1, emb2)
            model_scores.append(model_sim)
            human_scores.append(human_sim)
        
        if len(model_scores) < 2:
            return 0.0
        
        correlation, _ = spearmanr(model_scores, human_scores)
        return correlation
    
    def evaluate_analogy_task(self, analogies: List[Tuple[str, str, str, str]]) -> float:
        """
        Evaluate on analogy task and returns accuracy"""
        if not self.model:
            raise ValueError("Must train embeddings first")
        
        correct = 0
        total = 0
        
        for a, b, c, d in analogies:
            mapped = [
                self._surface_to_vocab_token(x) 
                for x in (a, b, c, d)
            ]
            if any(t is None for t in mapped):
                continue

            ma, mb, mc, md = mapped  
            predictions = self.model.analogy(ma, mb, mc, top_k=5)
            
            if not predictions:
                continue
            
            #see if d is in top 5 predictions
            pred_tokens = [token for token, _ in predictions]
            if md in pred_tokens:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def visualize_semantic_space(self, words: List[str], output_path: Path, 
                                 title: Optional[str] = None,
                                 group_by_language: bool = False,
                                 language_labels: Optional[List[str]] = None):
        """
        t-SNE to visualize 
        """

        if not self.model:
            raise ValueError("Must train embeddings first")
        
        embeddings = []
        valid_words = []
        valid_labels = []
        
        for i, word in enumerate(words):
            emb = self.model.get_embedding(word)
            if emb is not None:
                embeddings.append(emb)
                valid_words.append(word)
                if group_by_language and language_labels and i < len(language_labels):
                    valid_labels.append(language_labels[i])
        
        if len(embeddings) < 2:
            print("Not enough valid words for visualization")
            return
        
        embeddings = np.array(embeddings)
        
        #normalize embeddings for stability; NaN and inf values, replace with 0
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        #prevent numerical overflow in t-SNE by using L2 normalization per embedding vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 
        embeddings_normalized = embeddings / norms
        
        #scale to reasonable range to prevent extreme values, as above
        embeddings_normalized = np.clip(embeddings_normalized, -10, 10)
        
        #use t-SNE with adaptive perplexity
        from sklearn.manifold import TSNE
        n_samples = len(embeddings_normalized)
        perplexity = min(30, n_samples // 3)
        #needs to be at least 1
        perplexity = max(1, perplexity)
        
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity,
            early_exaggeration=12.0,
            learning_rate='auto',
            n_iter=1000,
            min_grad_norm=1e-7,
            verbose=0
        )
        embeddings_2d = tsne.fit_transform(embeddings_normalized)
        
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(14, 10))
        
        if group_by_language and valid_labels:
            from config import Language
            color_map = {
                'BH': '#ef4444',  
                'MH': '#22c55e',  
                'JBA': '#3b82f6'  
            }
            colors = [color_map.get(label, '#6b7280') for label in valid_labels]
            
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=colors, alpha=0.7, s=150, edgecolors='black', linewidths=1.5)
            
            unique_labels = list(set(valid_labels))
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color_map.get(label, '#6b7280'), 
                                    label=label) for label in unique_labels]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        else:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=range(len(valid_words)), cmap='viridis',
                               alpha=0.7, s=150, edgecolors='white', linewidths=1.5)
        
        for i, word in enumerate(valid_words):
            ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=12, alpha=0.9, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               alpha=0.7, edgecolor='gray', linewidth=0.5))
        
        plot_title = title or f'Semantic Space Visualization (t-SNE)\n{len(valid_words)} words'
        ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='medium')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='medium')
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved visualization to {output_path}")
    
    def analyze_morphological_coherence(self, root_families: Dict[str, List[str]]) -> Dict[str, float]:
        """
        test to see whether morphological variants cluster together"""
        if not self.model:
            raise ValueError("please train embeddings first")
        
        results = {}
        
        for root, variants in root_families.items():
            embeddings = []
            valid_variants = []
            
            for variant in variants:
                emb = self._get_embedding_for_surface(variant)  
                if emb is not None:
                    embeddings.append(emb)
                    valid_variants.append(variant)
            
            if len(embeddings) < 2:
                continue
            
            #compute average pairwise similarity within the root family
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            results[root] = avg_similarity
        
        return results
    
    def compare_cross_lingual_coherence(self, translation_pairs: List[Tuple[str, str]]) -> float:
        """
        Measure how well Hebrew-Aramaic translation pairs cluster"""
        if not self.model:
            raise ValueError("train embeddings first!")
        
        similarities = []
        
        for hebrew, aramaic in translation_pairs:
            emb_heb = self._get_embedding_for_surface(hebrew)
            emb_ara = self._get_embedding_for_surface(aramaic)
            
            if emb_heb is None or emb_ara is None:
                continue
            
            sim = 1 - cosine(emb_heb, emb_ara)
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def compare_tokenizer_embeddings(evaluators: Dict[str, 'EmbeddingEvaluator'], 
                                     words: List[str],
                                     output_path: Path):
        """
        compares semantic evaluations for multiple tokenizers.
        """

        n_tokenizers = len(evaluators)
        if n_tokenizers == 0:
            print("No evaluator dimensions provided for comparison")
            return
        
        fig, axes = plt.subplots(1, n_tokenizers, figsize=(6 * n_tokenizers, 6))
        if n_tokenizers == 1:
            axes = [axes]
        
        for idx, (name, evaluator) in enumerate(evaluators.items()):
            if not evaluator.model:
                print(f"warning: {name} has no trained model! we are skipping")
                continue
            
            #get embeddings for valid words
            embeddings = []
            valid_words = []
            
            for word in words:
                if hasattr(evaluator, "_get_embedding_for_surface"):
                    emb = evaluator._get_embedding_for_surface(word) 
                else:
                    emb = evaluator.model.get_embedding(word)
                if emb is not None:
                    embeddings.append(emb)
                    valid_words.append(word)
            
            if len(embeddings) < 2:
                print(f"warning: not enough valid words for {name}")
                continue
            
            embeddings = np.array(embeddings)
            
            #normalize embeddings - numerical stability
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings_normalized = embeddings / norms
            embeddings_normalized = np.clip(embeddings_normalized, -10, 10)
            
            from sklearn.manifold import TSNE
            n_samples = len(embeddings_normalized)
            perplexity = min(30, n_samples // 3)
            perplexity = max(1, perplexity)  # Ensure at least 1
            tsne = TSNE(
                n_components=2, 
                random_state=42,
                perplexity=perplexity,
                learning_rate='auto',
                verbose=0
            )
            embeddings_2d = tsne.fit_transform(embeddings_normalized)
            
            ax = axes[idx]
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               alpha=0.7, s=100, c=range(len(valid_words)), 
                               cmap='viridis')
            
            for i, word in enumerate(valid_words):
                ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                           fontsize=10, alpha=0.8, ha='center')
            
            ax.set_title(f'{name}\n({len(valid_words)} words)', fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Semantic Space Comparison Across Tokenizers', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison visualization to {output_path}")
    
    def plot_similarity_matrix(self, words: List[str], output_path: Path):

        if not self.model:
            raise ValueError("need to train embeddings first")
        
        embeddings = []
        valid_words = []
        
        for word in words:
            emb = self.model.get_embedding(word)
            if emb is not None:
                embeddings.append(emb)
                valid_words.append(word)
        
        if len(valid_words) < 2:
            print("Not enough valid words for similarity matrix")
            return
        
        embeddings = np.array(embeddings)
        
        similarity_matrix = np.zeros((len(valid_words), len(valid_words)))
        for i in range(len(valid_words)):
            for j in range(len(valid_words)):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    similarity_matrix[i, j] = sim
        
        plt.figure(figsize=(max(10, len(valid_words) * 0.8), max(8, len(valid_words) * 0.7)))
        sns.heatmap(similarity_matrix, 
                   xticklabels=valid_words, 
                   yticklabels=valid_words,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Cosine Similarity'},
                   square=True,
                   linewidths=0.5)
        
        plt.title('Word Similarity Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Words', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved similarity matrix to {output_path}")

if __name__ == "__main__":
    print("Embedding-based Tokenizer Evals Demo")
    print("=" * 50)
    print("\nThis little demo allows you to:")
    print("1.Train word embeddings on tokenized corpus")
    print("2.Evaluate semantic similarity")
    print("3.Test analogies")
    print("4.Analyze morphological coherence")
    print("5.Measure cross-lingual alignment")
    print("\nSee the main experiment framework for full usage.")