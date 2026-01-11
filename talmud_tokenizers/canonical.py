from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter, deque
import re
import math
import multiprocessing as mp
from talmud_tokenizers.tokenizer_base import BaseTokenizer
import numpy as np


class BPETokenizer(BaseTokenizer):
        
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 5, 
                 dropout_rate: float = 0.0):
        super().__init__(vocab_size, min_frequency)
        self.merges: List[Tuple[str, str]] = []
        self.dropout_rate = dropout_rate
        self.char_vocab: Set[str] = set()
    
    def train(self, texts: List[str], **kwargs):
        print("Training BPE tokenizer...")
        
        texts = [self.preprocess_text(text) for text in texts]
        
        word_freqs = self._get_word_frequencies(texts)
        self.char_vocab = self._build_char_vocab(word_freqs)
        
        splits = {
            word: [c for c in word]
            for word in word_freqs.keys()
        }
        
        self.merges = []
        current_vocab_size = len(self.char_vocab)
        initial_vocab_size = current_vocab_size
        print(f"  Starting with {initial_vocab_size} characters, target vocab size: {self.vocab_size}")
        
        while current_vocab_size < self.vocab_size:
            pair_freqs = self._compute_pair_frequencies_parallel(splits, word_freqs)
            
            if not pair_freqs:
                print(f"  No more pairs to merge. Stopping at vocab size: {current_vocab_size}")
                break
            
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            if pair_freqs[best_pair] < self.min_frequency:
                print(f"  Best pair frequency ({pair_freqs[best_pair]}) below min_frequency ({self.min_frequency}). Stopping at vocab size: {current_vocab_size}")
                break
            
            self.merges.append(best_pair)
            splits = self._merge_pair(best_pair, splits)
            current_vocab_size += 1
            
            progress_interval = 100 if self.vocab_size < 1000 else 1000
            if len(self.merges) % progress_interval == 0:
                print(f"  Learned {len(self.merges)} merges, vocab size: {current_vocab_size}/{self.vocab_size}")
        
        self._build_vocab(splits, word_freqs)
        self.is_trained = True
        
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
    
    def _get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        word_freqs = Counter()
        for text in texts:
            #tha basic whitespace tokenization
            words = text.split()
            word_freqs.update(words)
        return dict(word_freqs)
    
    def _build_char_vocab(self, word_freqs: Dict[str, int]) -> Set[str]:
        chars = set()
        for word in word_freqs.keys():
            chars.update(word)
        return chars
    
    def _compute_pair_frequencies(self, splits: Dict[str, List[str]], 
                                  word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """if we REALLY need to do single-process. otherwise see below"""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) < 2:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return dict(pair_freqs)

    def _compute_pair_frequencies_parallel(self, splits: Dict[str, List[str]], 
                                           word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        #if running inside a daemon process move :<(
        try:
            if mp.current_process().daemon:
                return self._compute_pair_frequencies(splits, word_freqs)
        except Exception:
            return self._compute_pair_frequencies(splits, word_freqs)

        items = list(word_freqs.items())
        if len(items) < 10000: 
            return self._compute_pair_frequencies(splits, word_freqs)
        
        def worker(chunk: List[Tuple[str, int]]) -> Dict[Tuple[str, str], int]:
            local = defaultdict(int)
            for word, freq in chunk:
                split = splits.get(word, [])
                if len(split) < 2:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    local[pair] += freq
            return local
        
        cpu = max(1, mp.cpu_count() - 1)
        chunk_size = max(1, len(items) // cpu)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        with mp.Pool(processes=cpu) as pool:
            results = pool.map(worker, chunks)
        
        merged = defaultdict(int)
        for res in results:
            for k, v in res.items():
                merged[k] += v
        return dict(merged)
    
    def _merge_pair(self, pair: Tuple[str, str], 
                    splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """need to merge pair in ALL word splits."""
        new_splits = {}
        
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(pair[0] + pair[1])
                    i += 2
                    #remember to skip a word!! hence the 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        
        return new_splits
    
    def _build_vocab(self, splits: Dict[str, List[str]], word_freqs: Dict[str, int]):
        """for final vocab"""
        vocab_set = set(self.char_vocab)
        
        for split in splits.values():
            vocab_set.update(split)
        
        #need special tokens for OOV and other issues; see paper
        vocab_set.add('[PAD]')
        vocab_set.add('[UNK]')
        vocab_set.add('[CLS]')
        vocab_set.add('[SEP]')
        vocab_set.add('[MASK]')
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
    
    def encode(self, text: str, dropout: bool = False) -> List[int]:
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        text = self.preprocess_text(text)
        words = text.split()
        token_ids = []
        
        for word in words:
            tokens = [c for c in word]
            tokens = self._apply_merges(tokens)
            
            #make IDs
            for token in tokens:
                token_id = self.vocab.get(token, self.vocab.get('[UNK]', 0))
                token_ids.append(token_id)
        
        return token_ids
    
    def _apply_merges(self, tokens: List[str]) -> List[str]:
        #gets applied to token-seq
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, '[UNK]') for tid in token_ids]
        
        #no need for special tokens
        tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]', '[MASK]']]
        
        #join tokens (very very basic; could use in future special glyphs TODO)
        return ' '.join(tokens)
    
    def _get_save_data(self) -> Dict:
        return {
            'merges': self.merges,
            'dropout_rate': self.dropout_rate,
            'char_vocab': list(self.char_vocab),
        }
    
    def _load_save_data(self, data: Dict):
        self.merges = data.get('merges', [])
        self.dropout_rate = data.get('dropout_rate', 0.1)
        self.char_vocab = set(data.get('char_vocab', []))


class TrieNode:
    """
    trie node for WordPiece vocabulary (using Google's LinMaxMatch algorithm); each node is a prefix in the vocabulary
    """
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_terminal = False
        #next is only if terminal
        self.token: Optional[str] = None
        #Aho-Corasick (see paper)
        self.failure_link: Optional['TrieNode'] = None
        self.failure_pops: List[str] = []


class WordPieceTokenizer(BaseTokenizer):
    """
    WordPiece tokenizer; using LinMaxMatch algorithm for O(n) comp
    """
    
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 5,
                 dropout_rate: float = 0.1):
        super().__init__(vocab_size, min_frequency)
        self.merges: List[Tuple[str, str]] = []
        self.continuing_subword_prefix = "##"
        self.dropout_rate = dropout_rate
        self.trie_root: Optional[TrieNode] = None
        self.trie_root_sharp: Optional[TrieNode] = None
    
    def train(self, texts: List[str], **kwargs):
        print("Initializing training for WordPiece tokenizer...")
        
        # Preprocess
        texts = [self.preprocess_text(text) for text in texts]
        
        # Get word frequencies
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            word_freqs.update(words)
        
        print(f"  Starting with {len(word_freqs):,} unique words")
        
        char_vocab = set()
        for word in word_freqs.keys():
            char_vocab.update(word)
        
        print(f"  Char vocabulary: {len(char_vocab)} chars")
        
        #initially all chars separate
        splits = {}
        for word in word_freqs.keys():
            chars = list(word)
            if len(chars) > 1:
                #first part of word normal, rest have prefix, so 1:
                splits[word] = [chars[0]] + [f"{self.continuing_subword_prefix}{c}" for c in chars[1:]]
            else:
                splits[word] = chars
        
        #track the vocab
        vocab_set = set()
        for split in splits.values():
            vocab_set.update(split)
        vocab_set.update(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        
        print(f"  Initial vocabulary size: {len(vocab_set)}")
        
        self.merges = []
        iteration = 0
        max_iterations = self.vocab_size
        
        while len(vocab_set) < self.vocab_size and iteration < max_iterations:
            iteration += 1
            pair_scores = self._compute_pair_scores_fixed_parallel(splits, word_freqs)
            
            if not pair_scores:
                print(f"  No more pairs to merge at iteration {iteration}")
                break
            best_pair = max(pair_scores, key=pair_scores.get)
            best_score = pair_scores[best_pair]
            epsilon = 1e-6
            #no magic numbers here!!
            if best_score < epsilon:
                print(f"  Stopping: the score is too low ({best_score:.2e})")
                break
            
            self.merges.append(best_pair)
            splits = self._merge_pair(best_pair, splits)
            
            vocab_set = set()
            for split in splits.values():
                vocab_set.update(split)
            vocab_set.update(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
            
            if iteration % 1000 == 0:
                print(f"  Iteration {iteration}: {len(self.merges)} merges, "
                    f"vocab size: {len(vocab_set)}, best score: {best_score:.4f}")
        
        self._build_vocab(splits)
        
        print("  Building trie - fast encoding...")
        self._build_trie()
        self._precompute_failure_links()
        
        self.is_trained = True
        
        print(f"Training complete:")
        print(f"  Merges: {len(self.merges):,}")
        print(f"  Final vocabulary size: {len(self.vocab):,}")
        print(f"  Iterations: {iteration:,}")

    def _compute_pair_scores_fixed(self, splits: Dict[str, List[str]], 
                                word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], float]:
        pair_freqs = defaultdict(int)
        token_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            split = splits[word]
            for token in split:
                token_freqs[token] += freq
            if len(split) >= 2:
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] += freq
        
        return self._score_pairs(pair_freqs, token_freqs)

    def _compute_pair_scores_fixed_parallel(self, splits: Dict[str, List[str]], 
                                word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], float]:
        #as above if daemon process get out
        try:
            if mp.current_process().daemon:
                return self._compute_pair_scores_fixed(splits, word_freqs)
        except Exception:
            return self._compute_pair_scores_fixed(splits, word_freqs)

        items = list(word_freqs.items())
        if len(items) < 10000:
            return self._compute_pair_scores_fixed(splits, word_freqs)

        def worker(chunk: List[Tuple[str, int]]):
            pf = defaultdict(int)
            tf = defaultdict(int)
            for word, freq in chunk:
                split = splits.get(word, [])
                for token in split:
                    tf[token] += freq
                if len(split) >= 2:
                    for i in range(len(split) - 1):
                        pair = (split[i], split[i + 1])
                        pf[pair] += freq
            return pf, tf

        cpu = max(1, mp.cpu_count() - 1)
        chunk_size = max(1, len(items) // cpu)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        with mp.Pool(processes=cpu) as pool:
            results = pool.map(worker, chunks)

        pair_freqs = defaultdict(int)
        token_freqs = defaultdict(int)
        for pf, tf in results:
            for k, v in pf.items():
                pair_freqs[k] += v
            for k, v in tf.items():
                token_freqs[k] += v

        return self._score_pairs(pair_freqs, token_freqs)

    def _score_pairs(self, pair_freqs: Dict[Tuple[str, str], int], token_freqs: Dict[str, int]) -> Dict[Tuple[str, str], float]:
        scores = {}
        for pair, pair_freq in pair_freqs.items():
            if pair_freq < self.min_frequency:
                continue
            a_freq = token_freqs[pair[0]]
            b_freq = token_freqs[pair[1]]
            if a_freq == 0 or b_freq == 0:
                continue
            score = math.log(pair_freq) - math.log(a_freq) - math.log(b_freq)
            scores[pair] = score
        return scores
        
    def _merge_pair(self, pair: Tuple[str, str], 
                    splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        new_splits = {}
        merged_token = pair[0] + pair[1].replace(self.continuing_subword_prefix, '')
        
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(merged_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        
        return new_splits
    
    def _build_vocab(self, splits: Dict[str, List[str]]):
        vocab_set = set()
        for split in splits.values():
            vocab_set.update(split)
        
        vocab_set.update(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_set))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
    
    def _build_trie(self):
        """
        trie structure from vocabulary for LinMaxMatch algorithm; two roots: one for regular tokens / one for tokens starting with ##.
        all tokens added to both tries  for matching
        """
        self.trie_root = TrieNode()
        self.trie_root_sharp = TrieNode()
        
        #roots = empty string match
        self.trie_root.is_terminal = True
        self.trie_root_sharp.is_terminal = True
        
        for token in self.vocab.keys():
            if token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
                continue
            
            #token gets added to reg root
            if not token.startswith(self.continuing_subword_prefix):
                current = self.trie_root
                chars_to_process = list(token)
                
                for char in chars_to_process:
                    if char not in current.children:
                        current.children[char] = TrieNode()
                    current = current.children[char]
                
                current.is_terminal = True
                current.token = token
            
            #continuing subword!
            if token.startswith(self.continuing_subword_prefix):
                remaining = token[len(self.continuing_subword_prefix):]
                current = self.trie_root_sharp
                chars_to_process = list(remaining)
            else:
                #could appear after ##
                current = self.trie_root_sharp
                chars_to_process = list(token)
            
            for char in chars_to_process:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            
            current.is_terminal = True
            current.token = token
    
    def _precompute_failure_links(self):
        """
        Precompute failure links and failure pops for LinMaxMatch based on Aho-Corasick algo
        """
        if self.trie_root is None or self.trie_root_sharp is None:
            return
        
        for root in [self.trie_root, self.trie_root_sharp]:
            queue = deque()
            
            root.failure_link = None
            root.failure_pops = []
            
            for char, child in root.children.items():
                child.failure_link = root
                child.failure_pops = []
                #root doesn't have token
                queue.append(child)
            
            while queue:
                current = queue.popleft()
                
                for char, child in current.children.items():
                    fail_node = current.failure_link
                    #keep going until we find char
                    while fail_node is not None:
                        if char in fail_node.children:
                            child.failure_link = fail_node.children[char]
                            break
                        fail_node = fail_node.failure_link
                    
                    if fail_node is None or char not in fail_node.children:
                        #no match so link to root
                        child.failure_link = root
                    
                    #failure pops - collect all terminal tokens on path from root to failure link
                    child.failure_pops = []
                    temp_node = child.failure_link
                    #collect tokens on failure link path
                    if child.failure_link is not None and child.failure_link.is_terminal:
                        if child.failure_link.token:
                            child.failure_pops.append(child.failure_link.token)
                    
                    queue.append(child)
    
    def encode(self, text: str, dropout: bool = False) -> List[int]:
        """Encode text using WordPiece with LinMaxMatch algorithm."""
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained")
        
        text = self.preprocess_text(text)
        
        words = text.split()
        token_ids = []
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
                continue
            if dropout:
                tokens = self._linmaxmatch_encode_with_dropout(word)
            else:
                tokens = self._linmaxmatch_encode(word)
            
            for token in tokens:
                token_ids.append(self.vocab.get(token, self.vocab.get('[UNK]', 0)))
        
        return token_ids
    
    def _h_function(self, node: Optional[TrieNode], char: str) -> Tuple[Optional[TrieNode], List[str]]:
        """
        h(u, c) transition function, returns (next_node, tokens_to_collect)
        """
        if node is None:
            return None, []
        
        #direct edge
        if char in node.children:
            next_node = node.children[char]
            tokens = []
            #tokens from failure pops while transition
            if next_node.failure_link is not None:
                tokens = next_node.failure_pops[:]
            return next_node, tokens
        
        #no edge, follow failure link
        fail_node = node.failure_link
        tokens = []
        
        while fail_node is not None:
            #collect from failure pops
            if fail_node.failure_pops:
                tokens.extend(fail_node.failure_pops)
            
            if char in fail_node.children:
                next_node = fail_node.children[char]
                #and from next_node's failure pops
                if next_node.failure_link is not None:
                    tokens.extend(next_node.failure_pops)
                return next_node, tokens
            
            fail_node = fail_node.failure_link
        
        #no matches
        return None, tokens
    
    def _match_loop(self, text: str, start_pos: int) -> Tuple[Optional[TrieNode], List[str], int]:
        """
        MatchLoop as described in LinMaxMatch paper
        """
        if self.trie_root is None:
            return None, [], start_pos
        
        tokens = []
        current_node = self.trie_root
        i = start_pos
        
        while i < len(text) and current_node is not None:
            char = text[i]
            
            #h(u,c) for transition
            next_node, collected = self._h_function(current_node, char)
            
            if next_node is not None:
                tokens.extend(collected)
                current_node = next_node
                i += 1
            else:
                #cant continue
                break
        
        return current_node, tokens, i
    
    def _linmaxmatch_encode(self, word: str) -> List[str]:
        if self.trie_root is None:
            return self._longest_match_fallback(word)
        
        #easiest to use longest-match with trie for fast lookups (TODO - full linmaxmatch requires checking failure links)
        return self._longest_match_trie_based(word)
    
    def _longest_match_trie_based(self, word: str) -> List[str]:
        tokens = []
        start = 0
        
        while start < len(word):
            if start == 0:
                current_root = self.trie_root
            else:
                current_root = self.trie_root_sharp
            
            longest_match = None
            longest_length = 0
            current_node = current_root
            
            #find longest match
            for i in range(start, len(word)):
                char = word[i]
                
                if char in current_node.children:
                    current_node = current_node.children[char]
                    if current_node.is_terminal and current_node.token:
                        longest_match = current_node.token
                        longest_length = i - start + 1
                else:
                    break
            
            if longest_match:
                tokens.append(longest_match)
                start += longest_length
            else:
                #use single char (no longest match)
                if start == 0:
                    char_token = word[start]
                else:
                    char_token = f"{self.continuing_subword_prefix}{word[start]}"
                
                #check whether chartok exists in vocab
                if char_token in self.vocab:
                    tokens.append(char_token)
                else:
                    tokens.append('[UNK]')
                start += 1
        
        return tokens if tokens else ['[UNK]']
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, '[UNK]') for tid in token_ids]
        
        #remove special tokens and doublesharp prefix
        result = []
        for token in tokens:
            if token in ['[PAD]', '[CLS]', '[SEP]', '[MASK]']:
                continue
            if token.startswith(self.continuing_subword_prefix):
                token = token[len(self.continuing_subword_prefix):]
            result.append(token)
        
        return ' '.join(result)
    
    def _get_save_data(self) -> Dict:
        return {
            'merges': self.merges,
            'prefix': self.continuing_subword_prefix,
            'dropout_rate': self.dropout_rate
        }
    
    def _load_save_data(self, data: Dict):
        self.merges = data.get('merges', [])
        self.continuing_subword_prefix = data.get('prefix', '##')
        self.dropout_rate = data.get('dropout_rate', 0.1)
        
        #rebuilds trie after learning vocab
        if self.is_trained and self.vocab:
            self._build_trie()
            self._precompute_failure_links()