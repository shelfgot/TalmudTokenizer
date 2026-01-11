"""
some utilities for parallel execution of tokenization training
"""
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict, Callable, Any, Tuple
import time
from pathlib import Path
import json
import traceback


def _run_configuration_worker(args: Tuple) -> Dict:
    """
    needs to be at module level for pickling...
    """
    if len(args) == 3:
        config_dict, experiment_config_dict, config_name = args
    else:
        if len(args) == 4:
            config_dict, _, experiment_config_dict, config_name = args
        else:
            config_dict, experiment_config_dict = args[:2]
            config_name = None
    
    try:
        #import inside so can pickle
        from config import TokenizerConfig, TokenizerAlgorithm, VocabularyStrategy
        from data import TalmudCorpus, FullTalmudCorpus
        from talmud_tokenizers.tokenizer_base import TokenizerWrapper
        from talmud_tokenizers.canonical import BPETokenizer, WordPieceTokenizer
        from talmud_tokenizers.advanced import (UnigramTokenizer, TokenMonsterTokenizer, 
                                                SRETokenizer, TokenMonsterSREHybridTokenizer)
        from evaluation import TokenizerEvaluator, MLMEvaluator
        
        #need required config_dict keys
        required_keys = ['algorithm', 'vocabulary_strategy', 'use_bpe_dropout']
        missing_keys = [key for key in required_keys if key not in config_dict]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        #reconstruct config from dict
        config = TokenizerConfig(
            algorithm=TokenizerAlgorithm(config_dict['algorithm']),
            vocabulary_strategy=VocabularyStrategy(config_dict['vocabulary_strategy']),
            use_bpe_dropout=config_dict['use_bpe_dropout'],
            vocab_size=config_dict.get('vocab_size', 32000),
            min_frequency=config_dict.get('min_frequency', 2),
            alpha_renyi=config_dict.get('alpha_renyi', 2.5),
            aramaic_upsampling_factor=config_dict.get('aramaic_upsampling_factor', 2.0),
            morphological_analyzer=config_dict.get('morphological_analyzer', 'dicta'),
        )
        
        #load corpus data
        corpus_source = experiment_config_dict.get('corpus_source', 'file')
        if corpus_source == 'sefaria_full':
            corpus = FullTalmudCorpus(
                tractates=experiment_config_dict.get('tractates'),
                cache_dir=None,
                max_words_per_segment=experiment_config_dict.get('max_words_per_segment', 40),
                min_words_per_segment=experiment_config_dict.get('min_words_per_segment', 3)
            )
        else:
            corpus_path = experiment_config_dict.get('corpus_path', 'sample_talmud_corpus.txt')
            corpus = TalmudCorpus(corpus_path)
        
        corpus.load()
        
        if not hasattr(corpus, 'texts') or not corpus.texts or len(corpus.texts) == 0:
            raise ValueError(f"Corpus failed to load or is empty (corpus_source={corpus_source})")

        train_texts, val_texts, test_texts, train_indices, val_indices, test_indices = corpus.split(
            train_ratio=experiment_config_dict.get('train_split', 0.8),
            val_ratio=experiment_config_dict.get('val_split', 0.1),
            test_ratio=experiment_config_dict.get('test_split', 0.1)
        )

        if not train_texts or len(train_texts) == 0:
            raise ValueError("Corpus split resulted in empty training set")
        if not test_texts or len(test_texts) == 0:
            raise ValueError("Corpus split resulted in empty test set")
        
        #check unified 
        if corpus_source == "sefaria_full":
            if config.vocabulary_strategy in [VocabularyStrategy.PARTITIONED, VocabularyStrategy.LANGUAGE_INFORMED]:
                return {
                    'config': config_dict,
                    'status': 'skipped',
                    'reason': 'with unannotated Bavli, we can only do unified tokenizers'
                }
        
        if config.algorithm == TokenizerAlgorithm.BPE:
            base_tokenizer = BPETokenizer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency,
                dropout_rate=config.dropout_rate
            )
        elif config.algorithm == TokenizerAlgorithm.WORDPIECE:
            base_tokenizer = WordPieceTokenizer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency
            )
        elif config.algorithm == TokenizerAlgorithm.UNIGRAM:
            base_tokenizer = UnigramTokenizer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency
            )
        elif config.algorithm == TokenizerAlgorithm.TOKENMONSTER:
            base_tokenizer = TokenMonsterTokenizer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency
            )
        elif config.algorithm == TokenizerAlgorithm.SRE:
            base_tokenizer = SRETokenizer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency,
                analyzer=config_dict.get('morphological_analyzer', 'dicta')
            )
        elif config.algorithm == TokenizerAlgorithm.TOKENMONSTER_SRE_HYBRID:
            from talmud_tokenizers.advanced import TokenMonsterSREHybridTokenizer
            base_tokenizer = TokenMonsterSREHybridTokenizer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
        
        tokenizer = TokenizerWrapper(base_tokenizer, config.vocabulary_strategy.value)
        
        #_____TRAIN TOKENIZER___________
        start_time = time.time()
        has_language_labels = hasattr(corpus, 'sentences') and hasattr(corpus, 'get_balanced_sample')
        
        if config.vocabulary_strategy == VocabularyStrategy.UNIFIED:
            tokenizer.train(train_texts)
        elif config.vocabulary_strategy == VocabularyStrategy.PARTITIONED:
            if not has_language_labels:
                raise ValueError("fatal error. the partitioned strategy needs language labels")
            language_labels = [corpus.sentences[i].language.value for i in train_indices]
            tokenizer.train(train_texts, language_labels=language_labels)
        elif config.vocabulary_strategy == VocabularyStrategy.LANGUAGE_INFORMED:
            if not has_language_labels:
                raise ValueError("fatal error. the language-informed strategy needs language labels")
            balanced_texts = corpus.get_balanced_sample(
                aramaic_factor=config.aramaic_upsampling_factor
            )
            balanced_labels = []
            if not corpus.sentences or len(corpus.sentences) == 0:
                raise ValueError("Your selected corpus doesn't have any sentences for the languaged-informed strategy")
            
            for text in balanced_texts:
                matching_sentence = next((s for s in corpus.sentences if s.text == text), None)
                if matching_sentence:
                    balanced_labels.append(matching_sentence.language.value)
                else:
                    #use first sentence's language
                    balanced_labels.append(corpus.sentences[0].language.value)
            tokenizer.train(balanced_texts, language_labels=balanced_labels)
        
        train_time = time.time() - start_time
        
        #evaluation of tokenizer
        if config.vocabulary_strategy == VocabularyStrategy.PARTITIONED:
            if tokenizer.language_tokenizers:
                eval_tokenizer = list(tokenizer.language_tokenizers.values())[0]
            else:
                eval_tokenizer = tokenizer.base_tokenizer
        else:
            eval_tokenizer = tokenizer.base_tokenizer
        
        if not eval_tokenizer.is_trained:
            raise ValueError(f"Tokenizer must be trained before encoding!!")
        
        #save the tokenizer
        output_dir = Path(experiment_config_dict.get('output_dir', './results'))
        tokenizer_dir = output_dir / 'tokenizers'
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = tokenizer_dir / f"{config.get_name()}.pkl"
        try:
            eval_tokenizer.save(tokenizer_path)
        except Exception as e:
            #if saving fails don't bail out
            print(f"Warning: Could not save tokenizer {config.get_name()}: {e}")
        
        from evaluation_optimized import OptimizedTokenizerEvaluator, EncodingCache
        
        #make cache directory for the worker
        output_dir = Path(experiment_config_dict.get('output_dir', './results'))
        cache_dir = output_dir / 'cache' / f"worker_{mp.current_process().pid}"
        cache = EncodingCache(cache_dir=cache_dir, max_size=50000)
        
        evaluator = OptimizedTokenizerEvaluator(
            eval_tokenizer,
            test_texts,
            cache=cache,
            tokenizer_id=config.get_name(),
            use_dropout=config.use_bpe_dropout,
            random_seed=42
        )
        intrinsic_results = evaluator.evaluate_all()
        
        cache.save()

        mlm_evaluator = MLMEvaluator(eval_tokenizer, train_texts, val_texts)
        mlm_results = mlm_evaluator.train_mlm_model(
            epochs=experiment_config_dict.get('mlm_epochs', 10),
            batch_size=experiment_config_dict.get('mlm_batch_size', 32),
            mask_prob=experiment_config_dict.get('mlm_mask_prob', 0.15)
        )
        
        result = {
            'config': config_dict,
            'intrinsic': intrinsic_results,
            'mlm': mlm_results,
            'train_time': train_time,
            'status': 'success'
        }

        if config_name:
            result['config']['name'] = config_name
        return result
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        if not isinstance(config_dict, dict):
            config_dict = {}
        result = {
            'config': config_dict.copy() if config_dict else {},
            'status': 'failed',
            'error': error_msg
        }
        if config_name:
            result['config']['name'] = config_name
        elif config_dict and isinstance(config_dict, dict):
            try:
                algo = config_dict.get('algorithm', 'unknown')
                strategy = config_dict.get('vocabulary_strategy', 'unknown')
                dropout = 'dropout' if config_dict.get('use_bpe_dropout', False) else 'nodropout'
                result['config']['name'] = f"{algo}_{strategy}_{dropout}"
            except Exception:
                result['config']['name'] = 'unknown'
        return result


class ParallelExperimentExecutor:
    """
    this executor runs the tokenization configurations in parallel
    """
    
    def __init__(self, num_workers: int = None, progress_callback: Callable = None):
        """
        start parallel executor
        """
        if num_workers is None:
            # Use CPU count - 1 to leave one core for system
            num_workers = max(1, mp.cpu_count() - 1)
        self.num_workers = num_workers
        self.progress_callback = progress_callback
        
    def run_configurations(self, 
                          configs: List[Dict],
                          experiment_config: Dict) -> Dict[str, Dict]:
        """
        returns a dict mapping config names to results
        """
        print(f"\n{'='*80}")
        print(f"PARALLEL EXECUTION: {len(configs)} configurations with {self.num_workers} workers")
        print(f"{'='*80}\n")
        
        worker_args = [
            (config['config_dict'], experiment_config, config['name'])
            for config in configs
        ]
        
        start_time = time.time()
        results = {}
        
        with Pool(processes=self.num_workers) as pool:
            completed = 0
            total = len(configs)
            
            for result in pool.imap_unordered(_run_configuration_worker, worker_args):
                completed += 1
                config_name = result.get('config', {}).get('name', 'unknown')
                if config_name == 'unknown' and 'config' in result:
                    config_dict = result['config']
                    algo = config_dict.get('algorithm', 'unknown')
                    strategy = config_dict.get('vocabulary_strategy', 'unknown')
                    dropout = 'dropout' if config_dict.get('use_bpe_dropout', False) else 'nodropout'
                    config_name = f"{algo}_{strategy}_{dropout}"
                
                results[config_name] = result
                
                if self.progress_callback:
                    self.progress_callback(completed, total)
                
                status = result.get('status', 'unknown')
                train_time = result.get('train_time', 0)
                if status == 'success':
                    print(f"[{completed}/{total}] ✓ {config_name} ({train_time:.1f}s)")
                elif status == 'skipped':
                    reason = result.get('reason', 'unknown')
                    print(f"[{completed}/{total}] ⊘ {config_name} (skipped: {reason})")
                else:
                    error = result.get('error', 'unknown error')[:50]  # Truncate long errors
                    print(f"[{completed}/{total}] ✗ {config_name} (failed: {error})")
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"PARALLEL EXECUTION COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"{'='*80}\n")
        
        return results