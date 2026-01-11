"""
the advanced tokenizer implementations
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import multiprocessing as mp
from tokenizer_base import BaseTokenizer
import numpy as np
import math
import re
import random


class UnigramTokenizer(BaseTokenizer):
    """
    UnigramLM tokenizer, based on Kudo2018
    """
    
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 5,
                 dropout_rate: float = 0.0):
        super().__init__(vocab_size, min_frequency)
        self.token_probs: Dict[str, float] = {}
        self.max_ngram_length: int = 3
        self.dropout_rate = dropout_rate
    
    def train(self, texts: List[str], **kwargs):
        """
        1. init large character n-gram vocabulary (1-4 chars)
        2. E-M algorithm employed to estimate probabilities
        3. lastly, iteratively prune the tokens with lowest probabilities
        """
        print("Training Unigram tokenizer...")
        
        texts = [self.preprocess_text(text) for text in texts]
        
        initial_vocab = self._initialize_vocab(texts)
        print(f"  Initial vocabulary size: {len(initial_vocab)}")

        #IID distribution
        self.token_probs = {token: 1.0 / len(initial_vocab) for token in initial_vocab}
        
        #EM algo follows:
        num_iterations = 4
        for iteration in range(num_iterations):
            #expectation (word by word)
            expected_counts = self._em_e_step(texts, initial_vocab)
            
            #maximization update
            total_count = sum(expected_counts.values())
            if total_count > 0:
                self.token_probs = {token: count / total_count 
                                   for token, count in expected_counts.items()}
            
            if (iteration + 1) % 2 == 0:
                print(f"  EM iteration {iteration + 1}/{num_iterations}")
        
        #iterative pruning per Kudo2018
        current_vocab = set(initial_vocab)
        special_tok_space = 5
        target_size = self.vocab_size - special_tok_space
        
        pruning_iteration = 0
        while len(current_vocab) > target_size:
            tokens_to_remove = self._prune_vocab(current_vocab, target_size)
            if not tokens_to_remove:
                break
            
            current_vocab -= tokens_to_remove
            
            #renorm probabilities
            total_prob = sum(self.token_probs.get(t, 0) for t in current_vocab)
            if total_prob > 0:
                self.token_probs = {t: self.token_probs.get(t, 0) / total_prob 
                                  for t in current_vocab}
            
            #redo EM to make more accurate, every PERIOD
            PERIOD = 10
            pruning_iteration += 1
            if pruning_iteration % PERIOD == 0:
                print(f"  Re-estimating probabilities (vocab size: {len(current_vocab)})...")
                for em_iter in range(2): 
                    expected_counts = self._em_e_step(texts, current_vocab)
                    total_count = sum(expected_counts.values())
                    if total_count > 0:
                        self.token_probs = {token: count / total_count 
                                           for token, count in expected_counts.items()}
            
            if len(current_vocab) % 1000 == 0:
                print(f"  pruned to {len(current_vocab)} tokens")
        
        #sort and add the special tokens
        vocab_list = sorted(current_vocab)
        vocab_list.extend(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        #update tokens; use tiny for the special
        tiny_epsilon = 1e-10
        for token in vocab_list:
            if token not in self.token_probs:
                self.token_probs[token] = tiny_epsilon
        
        self.is_trained = True
        print(f"Training complete; vocabulary size: {len(self.vocab)}")
    
    def _initialize_vocab(self, texts: List[str]) -> Set[str]:
        """init n-gram vocab w/frequency filtering/cap"""
        from collections import Counter
        ngram_counts = Counter()
        
        for text in texts:
            for n in range(1, self.max_ngram_length + 1):
                for i in range(len(text) - n + 1):
                    ngram = text[i:i+n]
                    ngram_counts[ngram] += 1
        
        top_ngram_number = 200_000
        filtered = [(ng, c) for ng, c in ngram_counts.items() if c >= self.min_frequency]
        filtered.sort(key=lambda x: x[1], reverse=True)
        top_ngrams = [ng for ng, _ in filtered[:top_ngram_number]]
        return set(top_ngrams)
    
    def _em_e_step(self, texts: List[str], vocab: Set[str]) -> Dict[str, float]:
        """
        forward-backward expectation - processes words separately (split on whitespace)
        -forward pass gets probability of reaching each position
        -backward pass gets probability of continuing from each position
        -expected counts gives expected value of count variable
        """
        expected_counts = defaultdict(float)
        
        for text in texts:
            if not text:
                continue
            
            words = text.split()
            for word in words:
                if not word:
                    continue
                
                word_counts = self._forward_backward_expected_counts(word, vocab)
                for token, count in word_counts.items():
                    expected_counts[token] += count
        
        return dict(expected_counts)
    
    def _forward_backward_expected_counts(self, text: str, vocab: Set[str]) -> Dict[str, float]:
        """
        dict: mapping token ----> expected count for text
        """
        n = len(text)
        if n == 0:
            return {}
        
        alpha = [-float('inf')] * (n + 1)
        alpha[0] = 0.0
        
        #tokens_at[j][i] will mean token text[j:i] exists in vocab
        tokens_at = {}
        
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_ngram_length), i):
                candidate = text[j:i]
                
                #check validity
                if candidate in vocab or len(candidate) == 1:
                    if (j, i) not in tokens_at:
                        tokens_at[(j, i)] = candidate
                    
                    #log probability
                    prob = self.token_probs.get(candidate, 1e-10)
                    log_prob = math.log(prob) if prob > 0 else -100.0
                    
                    #forward probability update (exponentiation becomes addition) 
                    log_sum = alpha[j] + log_prob
                    if alpha[i] == -float('inf'):
                        alpha[i] = log_sum
                    else:
                        #log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
                        max_val = max(alpha[i], log_sum)
                        alpha[i] = max_val + math.log(1.0 + math.exp(-abs(alpha[i] - log_sum)))
        
        #log-space prob
        total_log_prob = alpha[n]

        if total_log_prob == -float('inf') or not math.isfinite(total_log_prob):
            return {}
        
        #backwards - log probability of all segs from i to end
        beta = [-float('inf')] * (n + 1)
        beta[n] = 0.0 
        
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, min(n + 1, i + self.max_ngram_length + 1)):
                if (i, j) in tokens_at:
                    candidate = tokens_at[(i, j)]
                    
                    prob = self.token_probs.get(candidate, 1e-10)
                    log_prob = math.log(prob) if prob > 0 else -100.0
                    
                    log_sum = beta[j] + log_prob
                    if beta[i] == -float('inf'):
                        beta[i] = log_sum
                    else:
                        max_val = max(beta[i], log_sum)
                        beta[i] = max_val + math.log(1.0 + math.exp(-abs(beta[i] - log_sum)))
        
        expected_counts = defaultdict(float)
        
        for (j, i), token in tokens_at.items():
            # P(token at [j:i]) = a[j] * P(token) * b[i] / total_probability
            
            token_prob = self.token_probs.get(token, 1e-10)
            log_token_prob = math.log(token_prob) if token_prob > 0 else -100.0
            
            #that is, log probability of segmentation with this token
            log_seg_prob = alpha[j] + log_token_prob + beta[i]
            
            #linear/norm
            if math.isfinite(log_seg_prob) and log_seg_prob > -float('inf'):
                seg_prob = math.exp(log_seg_prob - total_log_prob)
                expected_counts[token] += seg_prob
        
        return dict(expected_counts)
    
    def _prune_vocab(self, vocab: Set[str], target_size: int) -> Set[str]:
        if len(vocab) <= target_size:
            return set()
        
        #sort by probability
        token_probs_list = [(t, self.token_probs.get(t, 0)) for t in vocab]
        token_probs_list.sort(key=lambda x: x[1])
        
        #boot the lowest
        num_to_remove = len(vocab) - target_size
        tokens_to_remove = {t for t, _ in token_probs_list[:num_to_remove]}
        
        return tokens_to_remove
    
    def _viterbi_decode(self, text: str, vocab: Set[str], dropout: bool = False) -> List[str]:
        """Viterbi decoder for segmentation"""
        if not text:
            return []
        
        if dropout:
            return self._viterbi_decode_with_dropout(text, vocab)
        
        n = len(text)

        #for this dp problem, dp[i] = (best_score, best_segmentation)
        dp = [(-float('inf'), []) for _ in range(n + 1)]
        dp[0] = (0.0, [])
        
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_ngram_length), i):
                candidate = text[j:i]
                if candidate in vocab or len(candidate) == 1:
                    #log to avoid underflow
                    prob = self.token_probs.get(candidate, 1e-10)
                    log_prob = math.log(prob) if prob > 0 else -100
                    
                    score = dp[j][0] + log_prob
                    if score > dp[i][0]:
                        segmentation = dp[j][1] + [candidate]
                        dp[i] = (score, segmentation)
        #if not send whole text
        return dp[n][1] if dp[n][1] else [text]

    def encode(self, text: str, dropout: bool = False) -> List[int]:
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        text = self.preprocess_text(text)
        
        words = text.split()
        token_ids = []
        
        for word in words:
            segmentation = self._viterbi_decode(word, set(self.vocab.keys()), dropout=dropout)
            
            for token in segmentation:
                token_id = self.vocab.get(token, self.vocab.get('[UNK]', 0))
                token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, '[UNK]') for tid in token_ids]
        
        tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]', '[MASK]']]
        
        return ''.join(tokens)
    
    def _get_save_data(self) -> Dict:
        return {
            'token_probs': self.token_probs,
            'max_ngram_length': self.max_ngram_length,
        }
    
    def _load_save_data(self, data: Dict):
        self.token_probs = data.get('token_probs', {})
        self.max_ngram_length = data.get('max_ngram_length', 3)


class TokenMonsterTokenizer(BaseTokenizer):
    """
    TokenMonster, based on optimal transport
    """
    
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 5,
                 dropout_rate: float = 0.0):
        super().__init__(vocab_size, min_frequency)
        self.capcode_markers: Dict[str, str] = {}
        self.token_costs: Dict[str, float] = {}
        #don't make this much larger...
        self.max_token_length: int = 8
        self.dropout_rate = dropout_rate
    
    def train(self, texts: List[str], **kwargs):
        """
        build vocab from frequent substrings, use opt-trans scoring, and play out multiple segmentation paths so non-greedy
        """
        print("Training TokenMonster tokenizer...")
        
        texts = [self.preprocess_text(text) for text in texts]
        
        vocab, substring_freqs = self._build_optimal_vocab(texts)
        print(f"  Built vocabulary with {len(vocab)} tokens")
        
        chars = set()
        for text in texts:
            chars.update(text)
        vocab.update(chars) 
        
        #cost = 1 / (frequency + 1) [so no divide by zero].lower cost is better
        max_freq = max(substring_freqs.values()) if substring_freqs else 1
        self.token_costs = {}
        for token in vocab:
            freq = substring_freqs.get(token, 1)
            self.token_costs[token] = 1.0 / (freq + 1)

        vocab_list = sorted(vocab)
        vocab_list.extend(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        FULL_COST = 1.0
        for token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
            self.token_costs[token] = FULL_COST
        

        for char in chars:
            if char not in self.token_costs:
                
                self.token_costs[char] = FULL_COST / 2  
        
        self.is_trained = True
        print(f"Training complete; vocab size: {len(self.vocab)}")
        print(f"  Unique characters: {len(chars)}")
    
    def _build_optimal_vocab(self, texts: List[str]) -> Tuple[Set[str], Counter]:
        """
        tuple returned is (vocab_set, substring_freqs)
        """
        substring_freqs = Counter()
        
        for text in texts:
            words = text.split()
            for word in words:
                for i in range(len(word)):
                    #smallest of max token or word
                    for j in range(i + 1, min(i + self.max_token_length + 1, len(word) + 1)):
                        substring = word[i:j]
                        substring_freqs[substring] += 1
        
        #keep track
        chars = set()
        for text in texts:
            chars.update(text)
        for char in chars:
            if char not in substring_freqs:
                substring_freqs[char] = 1
        
        #filter by min freq
        filtered_freqs = {token: freq for token, freq in substring_freqs.items() 
                         if freq >= self.min_frequency or len(token) == 1}

        #capped
        CAP = 200_000
        filtered_items = sorted(filtered_freqs.items(), key=lambda x: x[1], reverse=True)[:CAP]
        filtered_freqs = dict(filtered_items)
        
        SPECIAL_SIZE = 5
        target_size = self.vocab_size - SPECIAL_SIZE - len(chars)
        #minimum size is 100
        target_size = max(target_size, 100)
        
        #larger breaks ties
        sorted_items = sorted(filtered_freqs.items(), 
                            key=lambda x: (x[1], len(x[0])), 
                            reverse=True)
        top_substrings = [token for token, freq in sorted_items[:target_size]]
        
        #include all characters
        vocab_set = set(top_substrings)
        vocab_set.update(chars)
        
        return vocab_set, substring_freqs
    
    def _ungreedy_encode(self, word: str, dropout: bool = False) -> List[str]:
        if not word:
            return []
        
        if dropout:
            return self._ungreedy_encode_with_dropout(word)
        
        n = len(word)
        #for this dp problem - dp[i] = (best_cost, best_segmentation)
        dp = [(float('inf'), []) for _ in range(n + 1)]
        dp[0] = (0.0, [])
        
        for i in range(1, n + 1):
            #tries all possible segs ending at position i
            for j in range(max(0, i - self.max_token_length), i):
                candidate = word[j:i]
                
                if candidate in self.vocab:
                    cost = self.token_costs.get(candidate, 10.0)
                else:
                    #not in vocab! high cost
                    cost = 10.0
                
                total_cost = dp[j][0] + cost
                if total_cost < dp[i][0]:
                    segmentation = dp[j][1] + [candidate]
                    dp[i] = (total_cost, segmentation)
        
        #char-level if no vocab
        if not dp[n][1] or dp[n][0] == float('inf'):
            return list(word)
        
        return dp[n][1]

    def encode(self, text: str, dropout: bool = False) -> List[int]:
        if not self.is_trained:
            raise ValueError("tokenizer must be trained before encoding!")
        
        text = self.preprocess_text(text)
        
        words = text.split()
        token_ids = []
        
        for word in words:
            tokens = self._ungreedy_encode(word, dropout=dropout)
            for token in tokens:
                token_id = self.vocab.get(token)
                #this really shouldn't happen...
                if token_id is None:
                    token_id = self.vocab.get('[UNK]', 0)
                token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, '[UNK]') for tid in token_ids]
        tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]', '[MASK]']]
        
        return ''.join(tokens)
    
    def _get_save_data(self) -> Dict:
        return {
            'capcode_markers': self.capcode_markers,
            'token_costs': self.token_costs,
            'dropout_rate': self.dropout_rate
        }
    
    def _load_save_data(self, data: Dict):
        self.capcode_markers = data.get('capcode_markers', {})
        self.token_costs = data.get('token_costs', {})
        self.dropout_rate = data.get('dropout_rate', 0.1)


class SRETokenizer(BaseTokenizer):
    """
    Semitic Root Encoding; based on morphology. separates roots from templates, currently using heuristics
    """
    
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 5,
                 analyzer: str = "dicta", dropout_rate: float = 0.0):
        super().__init__(vocab_size, min_frequency)
        self.analyzer = analyzer
        self.dropout_rate = dropout_rate
        self.root_vocab: Dict[str, int] = {}
        self.template_vocab: Dict[str, int] = {}
        #list of Hebrew/Aramaic consonants
        self.consonants = set('בגדהוזחטיכסעפצקרשת')
        #some Hebrew/Aramaic prefixes and suffixes
        self.common_prefixes = ['ו', 'ב', 'כ', 'ל', 'מ', 'ש', 'ה', 'ת', 'ד']
        self.common_suffixes = ['ים', 'ות', 'ה', 'ו', 'י', 'ת', 'ך', 'כ', 'ם', 'ן', 'נא', 'נן', 'תא', 'יה']
       
        self.prefix_vocab: Dict[str, int] = {}
        self.suffix_vocab: Dict[str, int] = {}
        self.subword_vocab: Dict[str, int] = {}
    
    def train(self, texts: List[str], **kwargs):
        print("Training SRE tokenizer...")
        print("  Using heuristic-based morphological analysis with subword support")
        
        texts = [self.preprocess_text(text) for text in texts]
        
        roots = Counter()
        templates = Counter()
        prefixes = Counter()
        suffixes = Counter()
        root_template_pairs = []
        morphological_segments = []  
        
        substring_freqs = Counter()
        
        #DECOMPOSE THE WORD
        for text in texts:
            for word in text.split():
                prefix, root, suffix, template_stem = self._decompose_word_advanced(word)
                morphological_segments.append((prefix, root, suffix, template_stem))
                
                if root:
                    roots[root] += 1
                if template_stem:
                    templates[template_stem] += 1
                if prefix:
                    prefixes[prefix] += 1
                if suffix:
                    suffixes[suffix] += 1
                if root and template_stem:
                    root_template_pairs.append((root, template_stem))
                
                # Collect substrings for subword fallback (up to length 6 for better coverage)
                for i in range(len(word)):
                    for j in range(i + 1, min(i + 7, len(word) + 1)):
                        substring = word[i:j]
                        substring_freqs[substring] += 1
        
        print(f"  Extracted {len(roots)} unique roots, {len(templates)} unique templates")
        print(f"  Extracted {len(prefixes)} unique prefixes, {len(suffixes)} unique suffixes")
        
        #learn prefix/suffix lists from corpus frequencies, update common_prefixes and common_suffixes based on frequencies
        learned_prefixes = [p for p, freq in prefixes.most_common(20) if freq >= self.min_frequency]
        learned_suffixes = [s for s, freq in suffixes.most_common(25) if freq >= self.min_frequency]
        
        #merge w/original lists
        self.common_prefixes = list(dict.fromkeys(learned_prefixes + self.common_prefixes))[:30]
        self.common_suffixes = list(dict.fromkeys(learned_suffixes + self.common_suffixes))[:40]
        
        print(f"  Learned {len(learned_prefixes)} prefixes, {len(learned_suffixes)} suffixes from corpus")

        #space allocated: 35% roots, 20% templates, 5% prefixes, 5% suffixes, 20% subwords, about 15% special/other
        root_size = min(int(self.vocab_size * 0.35), len(roots))
        template_size = min(int(self.vocab_size * 0.20), len(templates))
        prefix_size = min(int(self.vocab_size * 0.05), len(prefixes))
        suffix_size = min(int(self.vocab_size * 0.05), len(suffixes))
        subword_size = min(int(self.vocab_size * 0.20), len(substring_freqs))
        
        top_roots = [r for r, _ in roots.most_common(root_size)]
        self.root_vocab = {root: idx for idx, root in enumerate(top_roots)}
        
        top_templates = [t for t, _ in templates.most_common(template_size)]
        self.template_vocab = {template: idx for idx, template in enumerate(top_templates)}
        
        top_prefixes = [p for p, _ in prefixes.most_common(prefix_size)]
        self.prefix_vocab = {prefix: idx for idx, prefix in enumerate(top_prefixes)}
        
        top_suffixes = [s for s, _ in suffixes.most_common(suffix_size)]
        self.suffix_vocab = {suffix: idx for idx, suffix in enumerate(top_suffixes)}
        
        #shouldn't need fallback
        top_subwords = [sw for sw, _ in substring_freqs.most_common(subword_size) 
                       if sw not in self.root_vocab and sw not in self.template_vocab]
        self.subword_vocab = {sw: idx for idx, sw in enumerate(top_subwords)}
        
        #combined vocab
        vocab_set = set()

        for prefix in top_prefixes[:50]:
            vocab_set.add(f"PREFIX:{prefix}")
        for root in top_roots[:200]:
            vocab_set.add(f"ROOT:{root}")
        for suffix in top_suffixes[:50]:
            vocab_set.add(f"SUFFIX:{suffix}")
        for template in top_templates[:100]:
            vocab_set.add(f"TEMPLATE:{template}")
        
        for root, template in root_template_pairs[:1500]:
            if root in self.root_vocab and template in self.template_vocab:
                vocab_set.add(f"{root}:{template}")

        for subword in top_subwords[:subword_size]:
            vocab_set.add(f"SUB:{subword}")
        
        #add all chars just in case
        chars = set()
        for text in texts:
            chars.update(text)
        for char in chars:
            vocab_set.add(char)
        
        #speciasl
        vocab_list = sorted(vocab_set)
        vocab_list.extend(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        self.is_trained = True
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
        print(f"  Root vocab: {len(self.root_vocab)}, Template vocab: {len(self.template_vocab)}")
        print(f"  Prefix vocab: {len(self.prefix_vocab)}, Suffix vocab: {len(self.suffix_vocab)}")
        print(f"  Subword vocab: {len(self.subword_vocab)}")
    
    def _extract_consonants(self, word: str) -> str:
        consonants_only = ''.join(c for c in word if c in self.consonants)
        return consonants_only
    
    def _extract_template_stem(self, word_stem: str, root: str) -> str:
        """
        Extract triconsonantal stem with placeholder notation (1,2,3 for radicals)
        return emplate stem string with placeholders (such as "12u3", "y123", if were in english)
        """
        if not root or not word_stem:
            return ''
        
        #position of root consonants (yes, yes there are some roots with vav and hey in them TODO)
        root_positions = []
        root_idx = 0
        cons_stem = self._extract_consonants(word_stem)
        
        if len(cons_stem) >= len(root):
            #greedy matching to find root consonants (in order) in the stem
            for i, char in enumerate(word_stem):
                if char in self.consonants and root_idx < len(root):
                    if cons_stem.find(root[root_idx], root_positions[-1] + 1 if root_positions else 0) >= 0:
                        #check if matches next
                        if char == root[root_idx]:
                            root_positions.append(i)
                            root_idx += 1
                            if root_idx >= len(root):
                                break
        
        #if not all radicals found...
        if len(root_positions) < len(root):
            #assume radicals are first N consonants in stem [works for past, some futures]
            root_positions = []
            cons_count = 0
            for i, char in enumerate(word_stem):
                if char in self.consonants:
                    if cons_count < len(root):
                        root_positions.append(i)
                        cons_count += 1
        
        #placeholder time
        template_parts = list(word_stem)
        for i, pos in enumerate(root_positions):
            if pos < len(template_parts):
                #these are 1-indexed by convention
                template_parts[pos] = str(i + 1)
        
        return ''.join(template_parts)
    
    def _decompose_word_advanced(self, word: str) -> Tuple[str, str, str, str]:
        if not word:
            return '', '', '', ''
        
        prefix = ''
        suffix = ''
        root = ''
        template_stem = ''
        
        cons = self._extract_consonants(word)
        
        #find prefixes and suffixes
        for pref in self.common_prefixes:
            if word.startswith(pref) and len(word) > len(pref):
                prefix = pref
                word_stem = word[len(pref):]
                break
        else:
            word_stem = word
        
        for suf in self.common_suffixes:
            if word_stem.endswith(suf) and len(word_stem) > len(suf):
                suffix = suf
                word_stem = word_stem[:-len(suf)]
                break
        
        #get root
        if len(cons) >= 3:
            #we'll let it be 4...usually triconsontal
            if prefix or suffix:
                cons_stem = self._extract_consonants(word_stem)
                if len(cons_stem) >= 3:
                    root = cons_stem[:min(4, len(cons_stem))]
                else:
                    #worst comes to worst use original consonants
                    root = cons[:min(4, len(cons))]
            else:
                root = cons[:min(4, len(cons))]

        if root and word_stem:
            template_stem = self._extract_template_stem(word_stem, root)
        elif word_stem:
            template_stem = word_stem
        
        return prefix, root, suffix, template_stem

    def _encode_word_morphological(self, word: str) -> List[str]:
        """
        prefix tokens + root token + template_stem token + suffix tokens
        """
        prefix, root, suffix, template_stem = self._decompose_word_advanced(word)
        tokens = []

        #try to get as much detail as possible, progressively step down

        if root and template_stem:
            pair_token = f"{root}:{template_stem}"
            if pair_token in self.vocab:
                # Add prefix and suffix if present
                if prefix and f"PREFIX:{prefix}" in self.vocab:
                    tokens.append(f"PREFIX:{prefix}")
                tokens.append(pair_token)
                if suffix and f"SUFFIX:{suffix}" in self.vocab:
                    tokens.append(f"SUFFIX:{suffix}")
                if tokens:
                    return tokens
                
        if prefix and f"PREFIX:{prefix}" in self.vocab:
            tokens.append(f"PREFIX:{prefix}")
        
        if root and f"ROOT:{root}" in self.vocab:
            tokens.append(f"ROOT:{root}")
        elif root and root in self.vocab:
            tokens.append(root)
        
        if template_stem and f"TEMPLATE:{template_stem}" in self.vocab:
            tokens.append(f"TEMPLATE:{template_stem}")
        elif template_stem and template_stem in self.vocab:
            tokens.append(template_stem)
        
        if suffix and f"SUFFIX:{suffix}" in self.vocab:
            tokens.append(f"SUFFIX:{suffix}")
        
        if tokens:
            return tokens
        
        if root and template_stem:
            if f"ROOT:{root}" in self.vocab:
                tokens.append(f"ROOT:{root}")
            if f"TEMPLATE:{template_stem}" in self.vocab:
                tokens.append(f"TEMPLATE:{template_stem}")
            if tokens:
                return tokens

        if root and f"ROOT:{root}" in self.vocab:
            return [f"ROOT:{root}"]
        
        if template_stem and f"TEMPLATE:{template_stem}" in self.vocab:
            return [f"TEMPLATE:{template_stem}"]
        
        return []

    def encode(self, text: str, dropout: bool = False) -> List[int]:
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        text = self.preprocess_text(text)
        
        words = text.split()
        token_ids = []
        
        for word in words:
            tokens = self._encode_word_morphological(word)
            
            #plan B
            if not tokens:
                tokens = self._encode_word_subword_fallback(word, dropout=dropout)
            
            for token in tokens:
                token_id = self.vocab.get(token)
                if token_id is None:
                    token_id = self.vocab.get('[UNK]', 0)
                token_ids.append(token_id)
        
        return token_ids
    
    def _reconstruct_word_from_root_template(self, root: str, template_stem: str) -> str:
        if not root or not template_stem:
            return template_stem if template_stem else ''
        
        result = list(template_stem)
        
        for i, char in enumerate(template_stem):
            if char.isdigit():
                #get back to 0-indexed
                placeholder_idx = int(char) - 1  
                if 0 <= placeholder_idx < len(root):
                    result[i] = root[placeholder_idx]
        
        return ''.join(result)
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(tid, '[UNK]') for tid in token_ids]
        
        tokens = [t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']]
        
        if not tokens:
            return ''
        
        words = []
        current_prefix = ''
        current_root = ''
        current_template = ''
        current_suffix = ''
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if ':' in token and not token.startswith(('PREFIX:', 'ROOT:', 'TEMPLATE:', 'SUFFIX:', 'SUB:')):
                parts = token.split(':', 1)
                if len(parts) == 2:
                    root, template_stem = parts
                    stem = self._reconstruct_word_from_root_template(root, template_stem)
                    word = current_prefix + stem + current_suffix if (current_prefix or current_suffix) else stem
                    words.append(word)
                    current_prefix = ''
                    current_root = ''
                    current_template = ''
                    current_suffix = ''
                    i += 1
                    continue
            
            if token.startswith('PREFIX:'):
                prefix = token[len('PREFIX:'):] 
                if current_prefix or current_root or current_template:
                    stem = self._reconstruct_word_from_root_template(current_root, current_template)
                    word = current_prefix + stem + current_suffix
                    words.append(word)
                    current_prefix = ''
                    current_root = ''
                    current_template = ''
                    current_suffix = ''
                current_prefix = prefix
            
            #get rid of prefix
            elif token.startswith('ROOT:'):
                current_root = token[len('ROOT:'):]
            elif token.startswith('TEMPLATE:'):
                current_template = token[len('TEMPLATE:'):]
            elif token.startswith('SUFFIX:'):
                current_suffix = token[len('SUFFIX:'):]
            elif token.startswith('SUB:'):
                subword = token[len('SUB:'):]  # Remove "SUB:" prefix
                if current_prefix or current_root or current_template or current_suffix:
                    # Previous word is complete
                    stem = self._reconstruct_word_from_root_template(current_root, current_template)
                    word = current_prefix + stem + current_suffix
                    words.append(word)
                    current_prefix = ''
                    current_root = ''
                    current_template = ''
                    current_suffix = ''
                words.append(subword)
            else:
                #just a plain token
                if current_root or current_template:
                    #then we have a word! finish the morph
                    stem = self._reconstruct_word_from_root_template(current_root, current_template)
                    word = current_prefix + stem + current_suffix
                    words.append(word)
                    current_prefix = ''
                    current_root = ''
                    current_template = ''
                    current_suffix = ''
                words.append(token)
            
            i += 1
        
        #last word, if exists
        if current_root or current_template or current_prefix or current_suffix:
            stem = self._reconstruct_word_from_root_template(current_root, current_template)
            word = current_prefix + stem + current_suffix
            words.append(word)
        
        return ' '.join(words)
    
    def _get_save_data(self) -> Dict:
        return {
            'root_vocab': self.root_vocab,
            'template_vocab': self.template_vocab,
            'analyzer': self.analyzer,
            'dropout_rate': self.dropout_rate,
        }
    
    def _load_save_data(self, data: Dict):
        self.root_vocab = data.get('root_vocab', {})
        self.template_vocab = data.get('template_vocab', {})
        self.analyzer = data.get('analyzer', 'dicta')
        self.dropout_rate = data.get('dropout_rate', 0.1)

class TokenMonsterSREHybridTokenizer(TokenMonsterTokenizer):
    """
    TokenMonster-SRE Hybrid
    """
    
    def _identify_morphological_tokens_parallel(self, vocab_set: Set[str], texts: List[str]) -> Dict[str, str]:
        #implemented with parallelization from the start
        try:
            if mp.current_process().daemon:
                return self._identify_morphological_tokens_sequential(vocab_set, texts)
        except Exception:
            return self._identify_morphological_tokens_sequential(vocab_set, texts)
        
        if len(texts) < 5000:
            return self._identify_morphological_tokens_sequential(vocab_set, texts)
        
        def worker(text_chunk: List[str]) -> Dict[str, Set[str]]:
            local_roots = set()
            local_templates = set()
            local_prefixes = set()
            local_suffixes = set()
            local_pairs = set()
            
            for text in text_chunk:
                for word in text.split():
                    prefix, root, suffix, template_stem = self._decompose_word_advanced(word)
                    
                    if root:
                        local_roots.add(root)
                    if template_stem:
                        local_templates.add(template_stem)
                    if prefix:
                        local_prefixes.add(prefix)
                    if suffix:
                        local_suffixes.add(suffix)
                    if root and template_stem:
                        local_pairs.add(f"{root}:{template_stem}")
            
            return {
                'roots': local_roots,
                'templates': local_templates,
                'prefixes': local_prefixes,
                'suffixes': local_suffixes,
                'pairs': local_pairs
            }
        
        cpu = max(1, mp.cpu_count() - 1)
        chunks = _chunk_items(texts, cpu)
        
        with mp.Pool(processes=cpu) as pool:
            results = pool.map(worker, chunks)
        
        all_roots = set()
        all_templates = set()
        all_prefixes = set()
        all_suffixes = set()
        all_pairs = set()
        
        for r in results:
            all_roots.update(r['roots'])
            all_templates.update(r['templates'])
            all_prefixes.update(r['prefixes'])
            all_suffixes.update(r['suffixes'])
            all_pairs.update(r['pairs'])
        
        token_types = {}
        for token in vocab_set:
            if token in all_pairs:
                token_types[token] = "pair"
            elif token in all_roots:
                token_types[token] = "root"
            elif token in all_templates:
                token_types[token] = "template"
            elif token in all_prefixes:
                token_types[token] = "prefix"
            elif token in all_suffixes:
                token_types[token] = "suffix"
            else:
                token_types[token] = "subword"
        
        return token_types
    
    def _identify_morphological_tokens_sequential(self, vocab_set: Set[str], texts: List[str]) -> Dict[str, str]:
        roots = set()
        templates = set()
        prefixes = set()
        suffixes = set()
        pairs = set()
        
        for text in texts:
            for word in text.split():
                prefix, root, suffix, template_stem = self._decompose_word_advanced(word)
                
                if root:
                    roots.add(root)
                if template_stem:
                    templates.add(template_stem)
                if prefix:
                    prefixes.add(prefix)
                if suffix:
                    suffixes.add(suffix)
                if root and template_stem:
                    pairs.add(f"{root}:{template_stem}")
        
        token_types = {}
        for token in vocab_set:
            if token in pairs:
                token_types[token] = "pair"
            elif token in roots:
                token_types[token] = "root"
            elif token in templates:
                token_types[token] = "template"
            elif token in prefixes:
                token_types[token] = "prefix"
            elif token in suffixes:
                token_types[token] = "suffix"
            else:
                token_types[token] = "subword"
        
        return token_types
    #see above for code for SRE
    def _decompose_word_advanced(self, word: str) -> Tuple[str, str, str, str]:
        #Hebrew/Aramaic consonants
        consonants = set('בגדהוזחטסעפצקרשת')
        common_prefixes = ['ו', 'ב', 'כ', 'ל', 'מ', 'ש', 'ה', 'ת', 'ד']
        common_suffixes = ['ים', 'ות', 'ה', 'ו', 'י', 'ת', 'ך', 'כ', 'ם', 'ן', 'נא', 'נן', 'תא', 'יה']
        
        if not word:
            return '', '', '', ''
        
        prefix = ''
        suffix = ''
        root = ''
        template_stem = ''
        
        cons = ''.join(c for c in word if c in consonants)
        
        #find prefix and suffix
        for pref in common_prefixes:
            if word.startswith(pref) and len(word) > len(pref):
                prefix = pref
                word_stem = word[len(pref):]
                break
        else:
            word_stem = word
        
        for suf in common_suffixes:
            if word_stem.endswith(suf) and len(word_stem) > len(suf):
                suffix = suf
                word_stem = word_stem[:-len(suf)]
                break
        
        #root extraction
        if len(cons) >= 3:
            if prefix or suffix:
                cons_stem = ''.join(c for c in word_stem if c in consonants)
                if len(cons_stem) >= 3:
                    root = cons_stem[:min(4, len(cons_stem))]
                else:
                    root = cons[:min(4, len(cons))]
            else:
                root = cons[:min(4, len(cons))]
        
        #template holder
        template_stem = word_stem if word_stem else ''
        
        return prefix, root, suffix, template_stem
    
    def train(self, texts: List[str], **kwargs):
        print("Training TokenMonster-SRE hybrid tokenizer...")
        
        texts = [self.preprocess_text(text) for text in texts]
        
        vocab, substring_freqs = self._build_optimal_vocab(texts)
        print(f"  Built vocabulary with {len(vocab)} tokens")
        
        print("  Identifying morphological components...")
        token_types = self._identify_morphological_tokens_parallel(vocab, texts)
        
        chars = set()
        for text in texts:
            chars.update(text)
        vocab.update(chars)
        
        #token costs with 'morphological discount'; 0.7x cost multiplier (where lower cost is preferred)
        self.token_costs = {}
        for token in vocab:
            freq = substring_freqs.get(token, 1)
            base_cost = 1.0 / (freq + 1)
            
            #morphological discount (tweak!)
            comp_type = token_types.get(token, "subword")
            if comp_type == "pair":
                self.token_costs[token] = base_cost * 0.7
            elif comp_type == "root":
                self.token_costs[token] = base_cost * 0.75 
            elif comp_type == "template":
                self.token_costs[token] = base_cost * 0.8
            elif comp_type in ["prefix", "suffix"]:
                self.token_costs[token] = base_cost * 0.85
            else:
                self.token_costs[token] = base_cost
        
        vocab_list = sorted(vocab)
        vocab_list.extend(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        for token in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
            self.token_costs[token] = 1.0
        
        for char in chars:
            if char not in self.token_costs:
                self.token_costs[char] = 0.5
        
        self.is_trained = True
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
        print(f"  Morphological tokens identified: {sum(1 for t in token_types.values() if t != 'subword')}")


def _chunk_items(items: List, num_chunks: int) -> List[List]:
    '''for parallel processing; even chunker'''
    chunk_size = max(1, len(items) // num_chunks)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _merge_counters(counters: List[Counter]) -> Counter:
    merged = Counter()
    for c in counters:
        merged.update(c)
    return merged


#export all of our tokenizers
__all__ = [
    'UnigramTokenizer',
    'SaGeTokenizer', 
    'TokenMonsterTokenizer',
    'SRETokenizer',
    'SREHybridTokenizer',
    'SRERegularizedTokenizer',
    'TokenMonsterSREHybridTokenizer'
]