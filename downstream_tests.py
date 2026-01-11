"""
Downstream evaluation for comparing Talmud tokenizers. We run this only AFTER tokenizers have been trained and
pickled by the main pipeline; the pipeline loads the full Babylonian Talmud, evaluates
semantic embedding quality, classification tasks, and a tiny GPT-2 LM. It finishes by
produces a comparison table/plot across tokenizers.
see the thesis itself for the explanation of the benchmark ground truth we created for the semantic tests.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import math
import time
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformers import GPT2LMHeadModel, GPT2Config
import torch
from config import Language
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm  
except ImportError:
    from tqdm import tqdm

from data import FullTalmudCorpus, TalmudCorpus, TalmudSentence
from RESEARCH_FINAL.talmud_tokenizers.tokenizer_base import BaseTokenizer
from prepare_corpus_splits import load_corpus_splits
from embedding_eval import EmbeddingEvaluator


# Phase 1: intrinsic evaluation (requires embeddings)


def evaluate_semantic_similarity(evaluator: EmbeddingEvaluator) -> Dict:
    """
    Test if semantically related words cluster together.
    Uses a pre-trained evaluator to avoid redundant training.
    """
    test_words = [
        "הלכה", 
        "תניא", 
        "איפוך",
        "קאי",
        "ריפתא",
        "רחמנא",
        "שכיחא",
        "מודי"
    ]

    results: Dict[str, List[Tuple[str, float]]] = {}
    for word in test_words:
        if not evaluator.model:
            similar = []
        elif hasattr(evaluator, "_surface_to_vocab_token"):
            token = evaluator._surface_to_vocab_token(word)  
            similar = (
                evaluator.model.most_similar(token, top_k=10)
                if token is not None
                else []
            )
        else:
            similar = evaluator.model.most_similar(word, top_k=10)

        results[word] = similar

    #avg top‑5 similarity per query and then avg across queries
    per_word_scores: List[float] = []
    for word in test_words:
        sims = [s for _, s in results.get(word, [])[:5]]
        if sims:
            per_word_scores.append(float(np.mean(sims)))
    avg_similarity = float(np.mean(per_word_scores)) if per_word_scores else 0.0

    return {
        "avg_top5_similarity": avg_similarity,
        "examples": results,
    }


def evaluate_morphological_coherence(evaluator: EmbeddingEvaluator) -> Dict:
    """
    Test if morphological variants cluster together.
    Uses a pre-trained evaluator to avoid redundant training.
    """
    root_families = {
        "SV'": ["נשבע", "משתבע", "ישבע"],         
        "AMR": ["אימא", "נימא", "תימא", "לימא"],   
        "LMD": ["למד", "מלמד", "ללמד", "למדת", "ללמוד"],        
        "QRA": ["קרא", "קורא", "מקרא", "קראה"]      
    }

    coherence_scores = evaluator.analyze_morphological_coherence(root_families)

    return {
        "avg_coherence": float(np.mean(list(coherence_scores.values())))
        if coherence_scores else 0.0,
        "per_root": coherence_scores,
    }


def evaluate_cross_lingual_alignment(evaluator: EmbeddingEvaluator) -> Dict:
    translation_pairs = [
        ("איש", "גברא"), 
        ("אשה", "איתתא"),    
        ("יום", "יומא"),     
        ("שעתא", "זמן"),
        ("גברא", "איש"),
        ("איסור", "איסורא"),
        ("סליק", "עלה"),
        ("ארנקי", "כיס")
    ]

    alignment_score = evaluator.compare_cross_lingual_coherence(translation_pairs)

    return {
        "alignment_score": float(alignment_score),
        "num_pairs": len(translation_pairs),
    }


def evaluate_analogies(evaluator: EmbeddingEvaluator) -> Dict:
    """
    Test word analogies (a:b :: c:d).
    Uses a pre-trained evaluator to avoid redundant training.
    
    Note: Using words that are more likely to be in the vocabulary based on testing.
    """
    analogies = [
        ("מקרא", "דקרא", "מהכא", "דהכא"),
        ("למד", "ללמוד", "טמא", "לטמא"),
        ("חמרא", "שתי", "ריפתא", "אכיל"),
        ("גיטא", "גירושין", "יאוש", "הפקר"),
        ("חודש", "זמן", "גניבה", "איסור"),
        ("דיקלא", "ארעא", "עניים", "עיר"),
        ("ברך", "קלל", "נכנס", "יצא")
    ]

    accuracy = evaluator.evaluate_analogy_task(analogies)

    return {
        "analogy_accuracy": float(accuracy),
        "num_analogies": len(analogies),
    }


def compute_embedding_score(tokenizer: BaseTokenizer,
                            corpus_texts: List[str],
                            embedding_epochs: int = 5,
                            embedding_dim: int = 100,
                            max_corpus_samples: Optional[int] = None,
                            tokenizer_id: Optional[str] = None,
                            early_stopping_patience: Optional[int] = None,
                            pre_tokenized_path: Optional[Path] = None,
                            embedding_save_path: Optional[Path] = None,
                            force_retrain: bool = False,
                            use_dropout: bool = False) -> Dict:
    """
    no need to be wasteful; train embeddings once and reuses them across all evaluation tasks.
    """

    evaluator = EmbeddingEvaluator(tokenizer, corpus_texts, embedding_dim=embedding_dim,
                                   tokenizer_id=tokenizer_id, use_cache=True,
                                   pre_tokenized_path=pre_tokenized_path,
                                   use_dropout=use_dropout)
    evaluator.train_embeddings(epochs=embedding_epochs, max_samples=max_corpus_samples,
                               early_stopping_patience=early_stopping_patience,
                               embedding_save_path=embedding_save_path,
                               force_retrain=force_retrain)
    
    sim = evaluate_semantic_similarity(evaluator)
    morph = evaluate_morphological_coherence(evaluator)
    cross = evaluate_cross_lingual_alignment(evaluator)
    analog = evaluate_analogies(evaluator)

    score = float(
        0.25 * sim["avg_top5_similarity"]
        + 0.30 * morph["avg_coherence"]
        + 0.25 * cross["alignment_score"]
        + 0.20 * analog["analogy_accuracy"]
    )

    return {
        "composite_score": score,
        "semantic_similarity": sim["avg_top5_similarity"],
        "morphological_coherence": morph["avg_coherence"],
        "cross_lingual_alignment": cross["alignment_score"],
        "analogy_accuracy": analog["analogy_accuracy"],
    }


# now for some supervised tasks!

def _encode_texts_as_bow(tokenizer: BaseTokenizer,
                         texts: List[str],
                         use_dropout: bool = False) -> np.ndarray:
    """
    encodes texts as histograms of token-ids for sklearn models
    """

    all_ids = set()
    encoded: List[List[int]] = []
    for t in texts:
        ids = tokenizer.encode(t, dropout=use_dropout)
        encoded.append(ids)
        all_ids.update(ids)

    if not all_ids:
        return np.zeros((len(texts), 0), dtype=np.float32)

    id_list = sorted(all_ids)
    id_to_col = {tid: i for i, tid in enumerate(id_list)}

    X = np.zeros((len(texts), len(id_list)), dtype=np.float32)
    for row, ids in enumerate(encoded):
        for tid in ids:
            col = id_to_col.get(tid)
            if col is not None:
                X[row, col] += 1.0
    return X

#TODO: not fully implemented, and I doubt this will work very well for most tractates
def evaluate_tractate_classification(tokenizer: BaseTokenizer,
                                     sentences: List[TalmudSentence],
                                     use_dropout: bool = False) -> Dict:
    """
    tests classification accuracy on task: which tractate a passage comes from
    """

    texts = [s.text for s in sentences if s.mesekhta is not None]
    labels = [s.mesekhta for s in sentences if s.mesekhta is not None]

    if not texts:
        return {"tractate_accuracy": 0.0, "num_samples": 0}

    X = _encode_texts_as_bow(tokenizer, texts, use_dropout=use_dropout)
    if X.shape[1] == 0:
        return {"tractate_accuracy": 0.0, "num_samples": len(texts)}

    clf = LogisticRegression(max_iter=200)
    scores = cross_val_score(clf, X, labels, cv=5)

    return {
        "tractate_accuracy": float(np.mean(scores)),
        "num_samples": len(texts),
    }

#TODO: integrate some examples
def evaluate_layer_classification(tokenizer: BaseTokenizer,
                                  sentences: List[TalmudSentence],
                                  use_dropout: bool = False) -> Dict:
    """
    test how well these classify Tannaic Hebrew (MH) vs Gemara (JBA) using language labels in `TalmudCorpus`.
    """
    

    texts: List[str] = []
    labels: List[int] = []

    for s in sentences:
        if s.language in (Language.MISHNAIC_HEBREW, Language.JEWISH_BABYLONIAN_ARAMAIC):
            texts.append(s.text)
            labels.append(0 if s.language == Language.MISHNAIC_HEBREW else 1)

    if not texts:
        return {"layer_accuracy": 0.0, "num_samples": 0}

    X = _encode_texts_as_bow(tokenizer, texts, use_dropout=use_dropout)
    if X.shape[1] == 0:
        return {"layer_accuracy": 0.0, "num_samples": len(texts)}

    clf = LogisticRegression(max_iter=200)
    scores = cross_val_score(clf, X, labels, cv=5)

    return {
        "layer_accuracy": float(np.mean(scores)),
        "num_samples": len(texts),
    }


#gpt-2 model

def evaluate_generation_perplexity(
    tokenizer: BaseTokenizer,
    train_texts: List[str],
    test_texts: List[str],
    max_train_examples: int = 5000,
    max_seq_len: int = 256,
    use_dropout: bool = False,
) -> Dict:

    #using cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if len(train_texts) > max_train_examples:
        train_texts = train_texts[:max_train_examples]

    def encode_corpus(texts: List[str]) -> torch.Tensor:
        ids: List[int] = []
        for t in texts:
            ids.extend(tokenizer.encode(t, dropout=use_dropout))
        if not ids:
            return torch.empty(0, dtype=torch.long)
        n_full = len(ids) // max_seq_len
        ids = ids[: n_full * max_seq_len]
        return torch.tensor(ids, dtype=torch.long)

    train_ids = encode_corpus(train_texts)
    test_ids = encode_corpus(test_texts)
    if train_ids.numel() == 0 or test_ids.numel() == 0:
        return {"perplexity": float("inf"), "bits_per_byte": float("inf")}

    train_ids = train_ids.view(-1, max_seq_len).to(device)
    test_ids = test_ids.view(-1, max_seq_len).to(device)

    vocab_size = max(tokenizer.vocab.values()) + 1 if tokenizer.vocab else 32000

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_positions=max_seq_len,
    )
    model = GPT2LMHeadModel(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()

    def run_epoch(split_ids: torch.Tensor) -> float:
        losses = []
        for batch in split_ids.split(8, dim=0):
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, labels=batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    #2 epochs, progress bar
    train_losses = []
    epoch_bar = tqdm(range(2), desc="GPT-2 training", unit="epoch")
    for epoch in epoch_bar:
        epoch_loss = run_epoch(train_ids)
        train_losses.append(epoch_loss)
        epoch_bar.set_postfix({"loss": f"{epoch_loss:.4f}"})
    train_loss = float(np.mean(train_losses))

    model.eval()
    with torch.no_grad():
        val_losses = []
        for batch in test_ids.split(8, dim=0):
            out = model(batch, labels=batch)
            val_losses.append(out.loss.item())
        val_loss = float(np.mean(val_losses))

    perplexity = float(math.exp(val_loss))
    bits_per_byte = float(np.log2(perplexity))

    return {
        "perplexity": perplexity,
        "bits_per_byte": bits_per_byte,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "device": str(device),
    }


def _evaluate_tokenizer_worker(args: Tuple) -> Dict:
    """
    Worker function for parallel tokenizer evaluation, where each worker loads its own copy of the corpus.
    """
    if len(args) >= 4:
        tokenizer, config_name, annotated_corpus_path, corpus_splits_dir = args[0], args[1], args[2], args[3]
        embedding_dim = args[4] if len(args) > 4 else 100
        embedding_epochs = args[5] if len(args) > 5 else 5
        max_corpus_samples = args[6] if len(args) > 6 else None
        early_stopping_patience = args[7] if len(args) > 7 else None
        pre_tokenized_path = args[8] if len(args) > 8 else None
        embedding_save_dir = args[9] if len(args) > 9 else None
        force_retrain_embeddings = args[10] if len(args) > 10 else False
    else:
        tokenizer, config_name, annotated_corpus_path, corpus_splits_dir = args
        embedding_dim, embedding_epochs, max_corpus_samples, early_stopping_patience, pre_tokenized_path = 100, 5, None, None, None
        embedding_save_dir, force_retrain_embeddings = None, False
    
    try:
        if corpus_splits_dir:
            train_texts, val_texts, test_texts = load_corpus_splits(Path(corpus_splits_dir))
        else:
            corpus = FullTalmudCorpus(cache_dir=None)
            if not getattr(corpus, "texts", None):
                corpus.load()
            
            train_texts, val_texts, test_texts, _, _, _ = corpus.split()
        
        annotated_corpus = None
        if annotated_corpus_path is not None:
            annotated_corpus = TalmudCorpus(annotated_corpus_path)
            annotated_corpus.load()
        
        results: Dict = {
            "config_name": config_name,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            all_texts = train_texts + val_texts + test_texts
            
            embedding_save_path = None
            if embedding_save_dir:
                embedding_save_dir_path = Path(embedding_save_dir)
                embedding_save_dir_path.mkdir(parents=True, exist_ok=True)
                embedding_save_path = embedding_save_dir_path / f"{config_name}_embeddings.pkl"
            
            emb_results = compute_embedding_score(
                tokenizer, all_texts, 
                embedding_epochs=embedding_epochs,
                embedding_dim=embedding_dim,
                max_corpus_samples=max_corpus_samples,
                tokenizer_id=config_name,
                early_stopping_patience=early_stopping_patience,
                pre_tokenized_path=pre_tokenized_path,
                embedding_save_path=embedding_save_path,
                force_retrain=force_retrain_embeddings,
            )
            results["embeddings"] = emb_results
        except Exception as e:
            results["embeddings"] = {"error": str(e)}
        
        try:
            if annotated_corpus is not None:
                tc = evaluate_tractate_classification(
                    tokenizer, annotated_corpus.sentences
                )
                lc = evaluate_layer_classification(
                    tokenizer, annotated_corpus.sentences
                )
                results["tractate_classification"] = tc
                results["layer_classification"] = lc
            else:
                results["tractate_classification"] = {"warning": "No annotated corpus"}
                results["layer_classification"] = {"warning": "No annotated corpus"}
        except Exception as e:
            results["tractate_classification"] = {"error": str(e)}
            results["layer_classification"] = {"error": str(e)}
        
        try:            
            lm = evaluate_generation_perplexity(
                tokenizer, train_texts, test_texts
            )
            results["language_model"] = lm
        except Exception as e:
            results["language_model"] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        return {
            "config_name": config_name,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }

#----MAIN CLASS---- for evaluating set of trained tokenizers

class DownstreamEvaluator:
    """
    full downstream evaluation pipeline manager for set of trained tokenizer implementation
    """

    def __init__(
        self,
        full_talmud_corpus: Optional[FullTalmudCorpus] = None,
        annotated_corpus_path: Optional[str] = None,
        corpus_splits_dir: Optional[str] = None,
        embedding_dim: int = 100,
        embedding_epochs: int = 5,
        max_corpus_samples: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        pre_tokenized_map: Optional[Dict[str, Path]] = None,
        embedding_save_dir: Optional[str] = None,
        force_retrain_embeddings: bool = False,
    ):
        """
        some notes on args:
            full_talmud_corpus: note that if corpus_splits_dir is provided, this is ignored
            annotated_corpus_path: Path to annotated corpus (for tractate/lay
        """
        if corpus_splits_dir:
            print(f"Loading corpus splits from {corpus_splits_dir}...")
            splits_path = Path(corpus_splits_dir)
            self.train_texts, self.val_texts, self.test_texts = load_corpus_splits(splits_path)
            print(f"  Train: {len(self.train_texts)} segments")
            print(f"  Val: {len(self.val_texts)} segments")
            print(f"  Test: {len(self.test_texts)} segments")
        else:
            self.corpus = full_talmud_corpus or FullTalmudCorpus(cache_dir=None)
            if not getattr(self.corpus, "texts", None):
                self.corpus.load()

            (
                self.train_texts,
                self.val_texts,
                self.test_texts,
                _,
                _,
                _,
            ) = self.corpus.split()

        self.annotated_corpus_path: Optional[str] = annotated_corpus_path
        self.corpus_splits_dir: Optional[str] = corpus_splits_dir
        self.annotated_corpus: Optional[TalmudCorpus] = None
        if annotated_corpus_path is not None:
            self.annotated_corpus = TalmudCorpus(annotated_corpus_path)
            self.annotated_corpus.load()
        
        self.embedding_dim = embedding_dim
        self.embedding_epochs = embedding_epochs
        self.max_corpus_samples = max_corpus_samples
        self.early_stopping_patience = early_stopping_patience
        self.pre_tokenized_map = pre_tokenized_map or {}
        self.embedding_save_dir = Path(embedding_save_dir) if embedding_save_dir else None
        self.force_retrain_embeddings = force_retrain_embeddings
        
        if self.embedding_save_dir:
            self.embedding_save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_tokenizer(self, tokenizer: BaseTokenizer,
                           config_name: str) -> Dict:
        """
        run all downstream evaluations for a SINGLE tokenizer. useful if you have to tweak one 
        algorithm but don't want to run the whole thing again
        """
        print("\n" + "=" * 80)
        print(f"EVALUATING: {config_name}")
        print("=" * 80)

        results: Dict = {
            "config_name": config_name,
            "timestamp": datetime.now().isoformat(),
        }

        #this may be getting repetitive...

        print("\n1. Embedding-based evaluation...")
        try:
            all_texts = self.train_texts + self.val_texts + self.test_texts
            
            # Check for pre-tokenized sequences
            pre_tokenized_path = self.pre_tokenized_map.get(config_name)
            if pre_tokenized_path:
                print(f"   Using pre-tokenized sequences: {pre_tokenized_path.name}")
            
            embedding_save_path = None
            if self.embedding_save_dir:
                embedding_save_path = self.embedding_save_dir / f"{config_name}_embeddings.pkl"
                if embedding_save_path.exists() and not self.force_retrain_embeddings:
                    print(f"   Found saved embeddings: {embedding_save_path.name}")
            
            
            emb_results = compute_embedding_score(
                tokenizer, all_texts, 
                embedding_epochs=self.embedding_epochs,
                embedding_dim=self.embedding_dim,
                max_corpus_samples=self.max_corpus_samples,
                tokenizer_id=config_name,
                early_stopping_patience=self.early_stopping_patience,
                pre_tokenized_path=pre_tokenized_path,
                embedding_save_path=embedding_save_path,
                force_retrain=self.force_retrain_embeddings,
                use_dropout=use_dropout
            )
            results["embeddings"] = emb_results
            print(f"   Composite score: {emb_results['composite_score']:.4f}")
        except Exception as e:
            print(f"   Failed: {e}")
            results["embeddings"] = {"error": str(e)}

        print("\n2. Classification tasks (tractate / layer)...")
        try:
            
            if self.annotated_corpus is not None:
                tc = evaluate_tractate_classification(
                    tokenizer, self.annotated_corpus.sentences, use_dropout=use_dropout
                )
                lc = evaluate_layer_classification(
                    tokenizer, self.annotated_corpus.sentences, use_dropout=use_dropout
                )
                results["tractate_classification"] = tc
                results["layer_classification"] = lc
                print(
                    f"   Tractate accuracy: {tc['tractate_accuracy']:.4f} "
                    f"(n={tc['num_samples']}), "
                    f"Layer accuracy: {lc['layer_accuracy']:.4f} "
                    f"(n={lc['num_samples']})"
                )
            else:
                print("  warning; Skipping: no annotated corpus")
                results["tractate_classification"] = {
                    "warning": "No annotated corpus",
                }
                results["layer_classification"] = {
                    "warning": "No annotated corpus",
                }
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results["tractate_classification"] = {"error": str(e)}
            results["layer_classification"] = {"error": str(e)}


        print("\n3. Language modeling (small GPT‑2)...")
        try:
            lm = evaluate_generation_perplexity(
                tokenizer, self.train_texts, self.test_texts
            )
            results["language_model"] = lm
            print(f"   ✓ Perplexity: {lm['perplexity']:.2f} (device={lm['device']})")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results["language_model"] = {"error": str(e)}

        return results

    def compare_all_tokenizers(
        self, 
        tokenizer_configs: List[Tuple[str, BaseTokenizer]],
        num_workers: Optional[int] = None
    ) -> List[Dict]:
        """
        compare list list of (config_name, tokenizer) pairs.
        """
        if num_workers is None:
            all_results: List[Dict] = []
            tokenizer_bar = tqdm(tokenizer_configs, desc="Evaluating tokenizers", unit="tokenizer")
            for name, tok in tokenizer_bar:
                tokenizer_bar.set_description(f"Evaluating: {name}")
                res = self.evaluate_tokenizer(tok, name)
                all_results.append(res)
                if "embeddings" in res and "composite_score" in res["embeddings"]:
                    score = res["embeddings"]["composite_score"]
                    tokenizer_bar.set_postfix({"score": f"{score:.4f}"})
            
            self.create_comparison_table(all_results)
            return all_results
        
        #initiate parallel execution
        if num_workers == 0:
            num_workers = max(1, mp.cpu_count() - 1)
        
        print(f"\n{'='*80}")
        print(f"PARALLEL EXECUTION: {len(tokenizer_configs)} tokenizers with {num_workers} workers")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"{'='*80}\n")
        
        #just as above, each worker needs corpus info
        corpus_splits_dir = getattr(self, 'corpus_splits_dir', None)
        embedding_dim = getattr(self, 'embedding_dim', 100)
        embedding_epochs = getattr(self, 'embedding_epochs', 5)
        max_corpus_samples = getattr(self, 'max_corpus_samples', None)
        early_stopping_patience = getattr(self, 'early_stopping_patience', None)
        pre_tokenized_map = getattr(self, 'pre_tokenized_map', {})
        embedding_save_dir = str(self.embedding_save_dir) if self.embedding_save_dir else None
        force_retrain_embeddings = getattr(self, 'force_retrain_embeddings', False)
        worker_args = [
            (tok, name, self.annotated_corpus_path, corpus_splits_dir,
             embedding_dim, embedding_epochs, max_corpus_samples, early_stopping_patience,
             pre_tokenized_map.get(name), embedding_save_dir, force_retrain_embeddings)
            for name, tok in tokenizer_configs
        ]
        
        #run evals in parallel
        start_time = time.time()
        all_results: List[Dict] = []
        
        with Pool(processes=num_workers) as pool:
            parallel_bar = tqdm(total=len(tokenizer_configs), desc="Parallel evaluation", unit="tokenizer")
            
            for result in pool.imap_unordered(_evaluate_tokenizer_worker, worker_args):
                config_name = result.get("config_name", "unknown")
                all_results.append(result)
                
                #update tqdm bar
                if "error" in result:
                    parallel_bar.set_postfix({"status": f"✗ {config_name}"})
                    tqdm.write(f"[{len(all_results)}/{len(tokenizer_configs)}] ✗ {config_name} (error: {result['error'][:50]})")
                else:
                    emb = result.get("embeddings", {})
                    score = emb.get("composite_score", "N/A")
                    parallel_bar.set_postfix({"status": f"✓ {config_name}", "score": score})
                    tqdm.write(f"[{len(all_results)}/{len(tokenizer_configs)}] ✓ {config_name} (score: {score})")
                
                parallel_bar.update(1)
            
            parallel_bar.close()
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"PARALLEL EXECUTION COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"{'='*80}\n")
        
        #sort results to match original order (lexicographic for ease)
        name_to_result = {r["config_name"]: r for r in all_results}
        ordered_results = []
        for name, _ in tokenizer_configs:
            if name in name_to_result:
                ordered_results.append(name_to_result[name])
            else:
                #error result for missing tokenizers
                ordered_results.append({
                    "config_name": name,
                    "error": "Missing result from parallel execution"
                })
        
        self.create_comparison_table(ordered_results)
        return ordered_results

    def create_comparison_table(self, results: List[Dict]) -> pd.DataFrame:
        rows = []
        for r in results:
            if "error" in r or "config_name" not in r:
                continue
                
            emb = r.get("embeddings", {})
            tc = r.get("tractate_classification", {})
            lm = r.get("language_model", {})

            row = {
                "Tokenizer": r["config_name"],
                "Embedding Score": emb.get("composite_score", np.nan),
                "Semantic Sim": emb.get("semantic_similarity", np.nan),
                "Morph Coherence": emb.get("morphological_coherence", np.nan),
                "Cross-Lingual": emb.get("cross_lingual_alignment", np.nan),
                "Analogy Acc": emb.get("analogy_accuracy", np.nan),
            }

            if tc and "error" not in tc and "warning" not in tc:
                row["Tractate Acc"] = tc["tractate_accuracy"]

            if lm and "error" not in lm:
                row["Perplexity"] = lm["perplexity"]

            rows.append(row)

        df = pd.DataFrame(rows)
        
        if df.empty:
            print("\n" + "=" * 80)
            print("COMPARISON TABLE")
            print("=" * 80 + "\n")
            print("uh oh!! No results to display. could be that all tokenizers have failed or produced errors.")
            return df
        
        if "Embedding Score" in df.columns:
            df = df.sort_values("Embedding Score", ascending=False)

        df.to_csv("downstream_comparison.csv", index=False)
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80 + "\n")
        print(df.to_string(index=False))

        self.plot_comparison(df)
        return df

    def plot_comparison(self, df: pd.DataFrame) -> None:
        """
        creates a multi-panel comparison plot
        """
        # Handle empty DataFrame
        if df.empty or "Tokenizer" not in df.columns:
            print("can't create the plots: DataFrame is empty or missing necessary columns")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        if "Embedding Score" in df.columns:
            df.plot(x="Tokenizer", y="Embedding Score", kind="barh", ax=ax1)
            ax1.set_title("Overall Embedding Score")
        else:
            ax1.text(0.5, 0.5, "No embedding scores", 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Overall Embedding Score")

        ax2 = axes[0, 1]
        comps = ["Semantic Sim", "Morph Coherence", "Cross-Lingual", "Analogy Acc"]
        available_comps = [c for c in comps if c in df.columns]
        if available_comps:
            df.plot(x="Tokenizer", y=available_comps, kind="barh", ax=ax2)
            ax2.set_title("Embedding Components")
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, "No embedding components", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Embedding Components")

        ax3 = axes[1, 0]
        if "Tractate Acc" in df.columns:
            df.plot(x="Tokenizer", y="Tractate Acc", kind="barh", ax=ax3)
            ax3.set_title("Tractate Classification Accuracy")
        else:
            ax3.text(0.5, 0.5, "No tractate classification data", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Tractate Classification Accuracy")

        ax4 = axes[1, 1]
        if "Perplexity" in df.columns:
            df.plot(x="Tokenizer", y="Perplexity", kind="barh", ax=ax4)
            ax4.set_title("Language Model Perplexity (lower = better)")
            ax4.invert_xaxis()
        else:
            ax4.text(0.5, 0.5, "No perplexity data", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Language Model Perplexity (lower = better)")

        plt.tight_layout()
        plt.savefig("downstream_comparison.png", dpi=300, bbox_inches="tight")
        print("\n downstream comparison plot saved to downstream_comparison.png")