"""
Main experiment orchestration for Talmud tokenization research.
"""

from typing import List, Dict
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import sys

from config import ExperimentConfig, TokenizerConfig, TokenizerAlgorithm, VocabularyStrategy
from data import TalmudCorpus
from RESEARCH_FINAL.talmud_tokenizers.tokenizer_base import TokenizerWrapper
from talmud_tokenizers.canonical import BPETokenizer, WordPieceTokenizer
from talmud_tokenizers.advanced import (UnigramTokenizer, TokenMonsterTokenizer, 
                                        SRETokenizer, TokenMonsterSREHybridTokenizer)
from evaluation import MLMEvaluator, VisualizationTools


class TalmudTokenizationExperiment:
    
    def __init__(self, config: ExperimentConfig):

        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.corpus = None
        self.train_texts = None
        self.val_texts = None
        self.test_texts = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        self.results = {}
        
    def load_corpus(self):
        print("=" * 80)
        print("loading the Talmudic Corpus")
        print("=" * 80)
        
        if self.config.corpus_source == "sefaria_full":
            from data import FullTalmudCorpus
            self.corpus = FullTalmudCorpus(
                tractates=self.config.tractates,
                #none i.e. default
                cache_dir=None, 
                max_words_per_segment=self.config.max_words_per_segment,
                min_words_per_segment=self.config.min_words_per_segment
            )
        else:
            self.corpus = TalmudCorpus(self.config.corpus_path)
        
        self.corpus.load()
        
        stats = self.corpus.compute_statistics()
        stats_path = self.output_dir / 'corpus_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nCorpus statistics saved to {stats_path}")
        
        print("\nSplitting corpus...")
        self.train_texts, self.val_texts, self.test_texts, self.train_indices, self.val_indices, self.test_indices = self.corpus.split(
            train_ratio=self.config.train_split,
            val_ratio=self.config.val_split,
            test_ratio=self.config.test_split
        )
        
        print(f"Train: {len(self.train_texts)} sentences")
        print(f"Val: {len(self.val_texts)} sentences")
        print(f"Test: {len(self.test_texts)} sentences")
    
    def create_tokenizer(self, config: TokenizerConfig):
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
        elif config.algorithm == TokenizerAlgorithm.TOKENMONSTER_SRE_HYBRID:
            base_tokenizer = TokenMonsterSREHybridTokenizer(
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
                analyzer=config.morphological_analyzer
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
        
        return TokenizerWrapper(base_tokenizer, config.vocabulary_strategy.value)
    
    def train_tokenizer(self, config: TokenizerConfig, tokenizer, pbar=None):

        msg = f"\nTraining {config.get_name()}..."
        if pbar:
            tqdm.write(msg, file=sys.stdout)
        else:
            print(msg)
        start_time = time.time()
        
        has_language_labels = hasattr(self.corpus, 'sentences') and hasattr(self.corpus, 'get_balanced_sample')
        
        if config.vocabulary_strategy == VocabularyStrategy.UNIFIED:
            tokenizer.train(self.train_texts)
        
        elif config.vocabulary_strategy == VocabularyStrategy.PARTITIONED:
            if not has_language_labels:
                raise ValueError(
                    f"partitioned tokenizers need tagged data but corpus {type(self.corpus).__name__}"
                )
            #use indices to map back to original sentences
            language_labels = [self.corpus.sentences[i].language.value for i in self.train_indices]
            tokenizer.train(self.train_texts, language_labels=language_labels)
        
        elif config.vocabulary_strategy == VocabularyStrategy.LANGUAGE_INFORMED:
            if not has_language_labels:
                raise ValueError(
                    f"partitioned tokenizers need tagged data but corpus {type(self.corpus).__name__}"
                )
            #upsample!
            balanced_texts = self.corpus.get_balanced_sample(
                aramaic_factor=config.aramaic_upsampling_factor
            )
            
            balanced_labels = []
            if not self.corpus.sentences or len(self.corpus.sentences) == 0:
                raise ValueError("Corpus has no sentences for LANGUAGE_INFORMED strategy")
            
            for text in balanced_texts:
                matching_sentence = next((s for s in self.corpus.sentences if s.text == text), None)
                if matching_sentence:
                    balanced_labels.append(matching_sentence.language.value)
                else:
                    balanced_labels.append(self.corpus.sentences[0].language.value)
            
            tokenizer.train(balanced_texts, language_labels=balanced_labels)
        
        train_time = time.time() - start_time
        msg = f"Training completed in {train_time:.2f}s"
        if pbar:
            tqdm.write(msg, file=sys.stdout)
        else:
            print(msg)
        
        tokenizer_dir = self.output_dir / 'tokenizers'
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = tokenizer_dir / f"{config.get_name()}.pkl"
        tokenizer.base_tokenizer.save(tokenizer_path)
        msg = f"Tokenizer saved to {tokenizer_path}"
        if pbar:
            tqdm.write(msg, file=sys.stdout)
        else:
            print(msg)
        
        return train_time
    
    def evaluate_tokenizer(self, config: TokenizerConfig, tokenizer, pbar=None) -> Dict:

        msg = f"\nEvaluating {config.get_name()}..."
        if pbar:
            tqdm.write(msg, file=sys.stdout)
        else:
            print(msg)
        
       
        if config.vocabulary_strategy == VocabularyStrategy.PARTITIONED:
            if tokenizer.language_tokenizers:

                eval_tokenizer = list(tokenizer.language_tokenizers.values())[0]
            else:
                eval_tokenizer = tokenizer.base_tokenizer
        else:
            eval_tokenizer = tokenizer.base_tokenizer
        
        if not eval_tokenizer.is_trained:
            raise ValueError(f"Tokenizer must be trained before encoding. Strategy: {config.vocabulary_strategy.value}")
        
        from evaluation_optimized import OptimizedTokenizerEvaluator, EncodingCache
        
        cache_dir = self.output_dir / 'cache'
        cache = EncodingCache(cache_dir=cache_dir, max_size=50000)
        evaluator = OptimizedTokenizerEvaluator(
            eval_tokenizer, 
            self.test_texts,
            cache=cache,
            tokenizer_id=config.get_name(),
            use_dropout=config.use_bpe_dropout,
            #Fixed seed so it's reproducibility
            random_seed=42  
        )
        intrinsic_results = evaluator.evaluate_all()
        
        cache_stats = cache.stats()
        if cache_stats['hits'] > 0 or cache_stats['misses'] > 0:
            msg = f"  Encoding cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate']:.1%} hit rate)"
            if pbar:
                tqdm.write(msg, file=sys.stdout)
            else:
                print(msg)
        
        cache.save()
        
        mlm_evaluator = MLMEvaluator(eval_tokenizer, self.train_texts, self.val_texts)
        mlm_results = mlm_evaluator.train_mlm_model(
            epochs=self.config.mlm_epochs,
            batch_size=self.config.mlm_batch_size,
            mask_prob=self.config.mlm_mask_prob
        )
        
        results = {
            'config': config.to_dict(),
            'intrinsic': intrinsic_results,
            'mlm': mlm_results,
        }
        
        return results
    
    def run_single_configuration(self, config: TokenizerConfig, pbar=None) -> Dict:

        if self.config.corpus_source == "sefaria_full":
            from config import VocabularyStrategy
            if config.vocabulary_strategy in [VocabularyStrategy.PARTITIONED, VocabularyStrategy.LANGUAGE_INFORMED]:
                msg = f"Skipping {config.get_name()} - requires language labels (full Talmud is unannotated)"
                if pbar:
                    tqdm.write(msg, file=sys.stdout)
                else:
                    print(msg)
                results = {
                    'config': config.to_dict(),
                    'status': 'skipped',
                    'reason': 'Full unannotated Talmud only supports UNIFIED strategy'
                }
                return results
        
        try:
            tokenizer = self.create_tokenizer(config)
            train_time = self.train_tokenizer(config, tokenizer, pbar=pbar)
            
            results = self.evaluate_tokenizer(config, tokenizer, pbar=pbar)
            results['train_time'] = train_time
            results['status'] = 'success'
            
        except Exception as e:
            msg = f"ERROR in configuration {config.get_name()}: {str(e)}"
            if pbar:
                tqdm.write(msg, file=sys.stdout)
            else:
                print(msg)
            results = {
                'config': config.to_dict(),
                'status': 'failed',
                'error': str(e)
            }
        
        return results
    
    def run_all_configurations(self):
        """Run all configurations (sequentially or in parallel)."""
        print("\n" + "=" * 80)
        print("RUNNING ALL CONFIGURATIONS")
        print("=" * 80)
        
        configs = self.config.generate_all_configs()
        print(f"\nTotal configurations to run: {len(configs)}")
        
        if self.config.use_parallel and len(configs) > 1:
            self._run_all_configurations_parallel(configs)
        else:
            self._run_all_configurations_sequential(configs)
        
        print("\n" + "=" * 80)
        print("ALL CONFIGURATIONS COMPLETED")
        print("=" * 80)
    
    def _run_all_configurations_sequential(self, configs):
        """Run configurations sequentially (original method)."""
        tokenizer_dir = self.output_dir / 'tokenizers'
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        results_dir = self.output_dir / 'individual_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        pbar = tqdm(configs, desc="Running configurations", unit="config", 
                   total=len(configs), dynamic_ncols=True, file=sys.stderr,
                   leave=True, mininterval=0.5)
        for config in pbar:
            pbar.set_postfix_str(config.get_name())
            tqdm.write(f"\n{'=' * 80}", file=sys.stdout)
            tqdm.write(f"CONFIGURATION: {config.get_name()}", file=sys.stdout)
            tqdm.write(f"{'=' * 80}", file=sys.stdout)
            
            results = self.run_single_configuration(config, pbar=pbar)
            self.results[config.get_name()] = results
            
            individual_result_path = results_dir / f"{config.get_name()}.json"
            with open(individual_result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self._save_results()
            tqdm.write(f"✓ Tokenizer and results saved for {config.get_name()}", file=sys.stdout)
            pbar.close()
    
    def _run_all_configurations_parallel(self, configs):
        from parallel_executor import ParallelExperimentExecutor
        
        config_dicts = []
        for config in configs:
            config_dicts.append({
                'name': config.get_name(),
                'config_dict': config.to_dict()
            })
        
        experiment_config_dict = {
            'corpus_source': self.config.corpus_source,
            'corpus_path': self.config.corpus_path,
            'output_dir': str(self.output_dir),  
            'tractates': self.config.tractates,
            'max_words_per_segment': self.config.max_words_per_segment,
            'min_words_per_segment': self.config.min_words_per_segment,
            'train_split': self.config.train_split,
            'val_split': self.config.val_split,
            'test_split': self.config.test_split,
            'mlm_epochs': self.config.mlm_epochs,
            'mlm_batch_size': self.config.mlm_batch_size,
            'mlm_mask_prob': self.config.mlm_mask_prob,
        }
        
        executor = ParallelExperimentExecutor(
            num_workers=self.config.num_workers,
            progress_callback=lambda current, total: self._save_results()
        )
        
        self.results = executor.run_configurations(
            config_dicts,
            experiment_config_dict
        )
        
        self._save_results()
    
    def _save_results(self):
        """Save results to disk."""
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        successful_results = {
            name: res for name, res in self.results.items() 
            if res.get('status') == 'success'
        }
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        flat_results = {}
        for name, res in successful_results.items():
            dist_stats = res['intrinsic']['distribution_stats']
            flat_results[name] = {
                'renyi_entropy': res['intrinsic']['renyi_entropy'],
                'nsl': res['intrinsic']['nsl'],
                'fertility': res['intrinsic']['fertility'],
                'zipfian_alignment': res['intrinsic']['zipfian_alignment'],
                'unique_tokens': dist_stats.get('unique_tokens_used', 0),
                'total_tokens': dist_stats.get('total_tokens', 0),
                'vocab_size': dist_stats.get('vocab_size', 0),
                'vocab_usage_percentage': dist_stats.get('vocab_usage_percentage', 0.0),
                'gini': dist_stats.get('gini_coefficient', 0.0),
            }
        
        viz_tools = VisualizationTools()
        
        for metric in ['renyi_entropy', 'nsl', 'fertility', 'zipfian_alignment']:
            viz_tools.plot_comparative_metrics(
                flat_results,
                viz_dir / f'{metric}_comparison.png',
                metric_name=metric
            )
        
        viz_tools.plot_combined_metrics(
            flat_results,
            viz_dir / 'combined_metrics.png'
        )
        
        metrics = ['renyi_entropy', 'nsl', 'fertility', 'zipfian_alignment', 'gini', 
                  'vocab_usage_percentage', 'total_tokens']
        viz_tools.plot_metrics_heatmap(
            flat_results,
            metrics,
            viz_dir / 'metrics_heatmap.png'
        )
        
        print(f"\nVisualizations saved to {viz_dir}")
    
    def generate_report(self):
        """Generate final research report."""
        print("\n" + "=" * 80)
        print("GENERATING REPORT")
        print("=" * 80)
        
        rows = []
        for name, res in self.results.items():
            if res.get('status') != 'success':
                continue
            
            row = {
                'Configuration': name,
                'Algorithm': res['config']['algorithm'],
                'Strategy': res['config']['vocabulary_strategy'],
                'Dropout': res['config']['use_bpe_dropout'],
                'Renyi Entropy': res['intrinsic']['renyi_entropy'],
                'NSL': res['intrinsic']['nsl'],
                'Fertility': res['intrinsic']['fertility'],
                'Zipfian Alignment': res['intrinsic']['zipfian_alignment'],
                'Train Time (s)': res['train_time'],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        csv_path = self.output_dir / 'results_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results summary saved to {csv_path}")
        
        report_path = self.output_dir / 'REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Talmud Tokenization Research Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary Stats\n\n")
            f.write(f"- Total configurations: {len(self.results)}\n")
            f.write(f"- Successful: {sum(1 for r in self.results.values() if r.get('status') == 'success')}\n")
            f.write(f"- Failed: {sum(1 for r in self.results.values() if r.get('status') != 'success')}\n\n")
            
            f.write("## Top 5 Configurations by Renyi Entropy\n\n")
            df_sorted = df.sort_values('Renyi Entropy', ascending=False).head()
            try:
                f.write(df_sorted.to_markdown(index=False))
            except ImportError:
                f.write(df_sorted.to_string(index=False))
            f.write("\n\n")
            
            f.write("## Top 5 Configurations by NSL\n\n")
            df_sorted = df.sort_values('NSL').head()
            try:
                f.write(df_sorted.to_markdown(index=False))
            except ImportError:
                f.write(df_sorted.to_string(index=False))
            f.write("\n\n")
        
        print(f"Report saved to {report_path}")
    
    def run(self):
        '''RUN THE FULL PIPELINE'''
        print("=" * 80)
        print("TALMUD TOKENIZATION EXPERIMENT")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_corpus()
        self.run_all_configurations()
        self.generate_visualizations()
        self.generate_report()
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED")
        print("=" * 80)
        print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {self.output_dir}")

    def main():
        """Main entry point."""
        config = ExperimentConfig()
        
        experiment = TalmudTokenizationExperiment(config)
        experiment.run()


if __name__ == "__main__":
    main()
