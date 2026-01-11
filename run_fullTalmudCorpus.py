from pathlib import Path
import json
import argparse
from datetime import datetime
from config import ExperimentConfig, TokenizerAlgorithm
from main import TalmudTokenizationExperiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full Talmud corpus tokenization experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=None
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='output directory for results (default: timestamped: ./results_full_talmud_<YYYY-MM-DD_HH-MM-SS>)'
    )
    parser.add_argument(
        '--use-timestamp',
        action='store_true',
        help='output timestamp-based output directory (e.g., results_full_talmud_2025-01-30_14-30-45). Default is True unless --output-dir is provided.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='retrain all tokenizers (ignore existing .pkl / results).'
    )
    
    args = parser.parse_args()
    
    if args.use_timestamp or args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"./results_full_talmud_{timestamp}"
    elif args.output_dir:
        output_dir = args.output_dir
    
    algorithms = list(TokenizerAlgorithm)
    
    config = ExperimentConfig(
        corpus_source="sefaria_full",
        tractates=None,              
        max_words_per_segment=40,
        min_words_per_segment=3,
        vocab_size=15000,
        algorithms=algorithms,
        output_dir=output_dir
    )

    print(f"\n{'='*80}")
    print(f"TALMUD TOKENIZATION EXPERIMENT")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    experiment = TalmudTokenizationExperiment(config)
    
    output_dir = Path(config.output_dir)
    tokenizer_dir = output_dir / 'tokenizers'
    results_dir = output_dir / 'individual_results'
    results_path = output_dir / 'results.json'
    
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            experiment.results = json.load(f)
        print(f"Loaded {len(experiment.results)} existing results")
    else:
        experiment.results = {}
    
    all_configs = config.generate_all_configs()
    
    #remove completed configurations unless --force
    completed_configs = set()
    if not args.force:
        if tokenizer_dir.exists():
            existing_tokenizers = {f.stem for f in tokenizer_dir.glob("*.pkl")}
            #tokenizer is then trained
            completed_configs = existing_tokenizers
    else:
        print("Forcing retrain: ignoring existing tokenizers/results.")

    remaining_configs = [cfg for cfg in all_configs if args.force or cfg.get_name() not in completed_configs]
    
    print(f"\nTotal configurations: {len(all_configs)}")
    print(f"Already completed: {len(completed_configs)}")
    print(f"Remaining to run: {len(remaining_configs)}")
    
    if completed_configs and not args.force:
        print(f"\nCompleted configurations:")
        for name in sorted(completed_configs):
            print(f"  good. {name}")
    
    if remaining_configs:
        print(f"\nRemaining configurations:")
        for cfg in remaining_configs:
            print(f"  - {cfg.get_name()}")
        
        experiment.load_corpus()
        
        if config.use_parallel and len(remaining_configs) > 1:
            experiment._run_all_configurations_parallel(remaining_configs)
        else:
            experiment._run_all_configurations_sequential(remaining_configs)
        
        experiment.generate_visualizations()
        experiment.generate_report()
        
        print("\n" + "=" * 80)
        print("evaluation completed.")
        print("=" * 80)
        print(f"results saved to: {config.output_dir}")
    else:
        print("\nall configurations are already complete!")
        experiment.generate_visualizations()
        experiment.generate_report()