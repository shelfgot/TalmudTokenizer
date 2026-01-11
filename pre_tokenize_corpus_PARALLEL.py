"""
pretokenize sequences with parallelization (which won't help for training LLM's downstream)
Usage: run, for example, python3 pre_tokenize_corpus_PARALLEL.py --tokenizer-dir tokenizers --corpus-splits-dir corpus_splits
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import sys
import multiprocessing as mp
from multiprocessing import Pool

from talmud_tokenizers.tokenizer_base import BaseTokenizer
from talmud_tokenizers.canonical import BPETokenizer, WordPieceTokenizer
from talmud_tokenizers.advanced import UnigramTokenizer, TokenMonsterSREHybridTokenizer, TokenMonsterTokenizer, SRETokenizer
from prepare_corpus_splits import load_corpus_splits
from embedding_eval import EmbeddingEvaluator

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

def load_tokenizer_class(class_name: str) -> type:
    tokenizer_classes = {
        'BPETokenizer': BPETokenizer,
        'WordPieceTokenizer': WordPieceTokenizer,
        'UnigramTokenizer': UnigramTokenizer,
        'TokenMonsterTokenizer': TokenMonsterTokenizer,
        'SRETokenizer': SRETokenizer,
        'TokenMonsterSREHybridTokenizer': TokenMonsterSREHybridTokenizer,
    }
    
    if class_name not in tokenizer_classes:
        raise ValueError(f"Unknown tokenizer class: {class_name}")
    
    return tokenizer_classes[class_name]


def load_saved_tokenizer(tokenizer_path: Path) -> Optional[BaseTokenizer]:
    try:
        #see if valid pickle
        with open(tokenizer_path, 'rb') as f:
            data = pickle.load(f)
        
       
        class_name = data.get('class_name')

        tokenizer_class = load_tokenizer_class(class_name)
        
        #reconstruct the tokenizer using the load method
        tokenizer = tokenizer_class.load(tokenizer_path)
        
        #check if tokenizer actually trained
        if not hasattr(tokenizer, 'is_trained') or not tokenizer.is_trained:
            print(f" tokenizer loaded but not trained")
        
        return tokenizer
        
    except Exception as e:
        print(f"error loading tokenizer from {tokenizer_path}: {e}")
        traceback.print_exc()
        return None


def load_tokenizer(tokenizer_path: Path) -> BaseTokenizer:    
    tokenizer = load_saved_tokenizer(tokenizer_path)
    if tokenizer is None:
        raise ValueError(f"Couldn't load tokenizer from {tokenizer_path}")
    
    if not hasattr(tokenizer, 'is_trained') or not tokenizer.is_trained:
        raise ValueError(f"Tokenizer at {tokenizer_path} is not trained!")
    
    return tokenizer


def _pre_tokenize_worker(args: Tuple) -> Tuple[Optional[Path], str]:
    """
    Worker function for parallel pre-tokenization; hads to be at module level for multiprocessing.
    """
    tokenizer_path, corpus_splits_dir, output_dir, max_samples = args
    tokenizer_name = tokenizer_path.stem
    
    try:
        #load splits in workers
        train_texts, val_texts, test_texts = load_corpus_splits(corpus_splits_dir)
        all_texts = train_texts + val_texts + test_texts
        
        if max_samples and len(all_texts) > max_samples:
            import random
            all_texts = random.sample(all_texts, max_samples)
        
        if max_samples:
            output_filename = f"{tokenizer_name}_max{max_samples}.pkl"
        else:
            output_filename = f"{tokenizer_name}_full.pkl"
        
        output_path = output_dir / output_filename
        
        tokenizer = load_tokenizer(tokenizer_path)
        
        evaluator = EmbeddingEvaluator(
            tokenizer=tokenizer,
            texts=all_texts,
            tokenizer_id=tokenizer_name,
            use_cache=False
        )
        
        token_sequences = evaluator._get_tokenized_sequences(max_samples=None)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'token_sequences': token_sequences,
            'tokenizer_id': tokenizer_name,
            'num_texts': len(all_texts),
            'num_sequences': len(token_sequences),
            'max_samples': max_samples
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        return (output_path, tokenizer_name)
        
    except Exception as e:
        import traceback
        error_msg = f"{tokenizer_name}: {str(e)}\n{traceback.format_exc()}"
        return (None, error_msg)


def batch_pre_tokenize_all_parallel(
    tokenizer_dir: Path,
    corpus_splits_dir: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    skip_existing: bool = True,
    num_workers: Optional[int] = None
) -> List[Path]:

    #find all tokenizer files
    tokenizer_files = sorted(tokenizer_dir.glob("*.pkl"))
    
    if not tokenizer_files:
        print(f"warning - no tokenizer .pkl files found in {tokenizer_dir}")
        return []
    
    print(f"\n{'='*80}")
    print(f"BATCH PRE-TOKENIZATION (PARALLEL): Found {len(tokenizer_files)} tokenizers")
    print(f"{'='*80}")
    
    #filter tokenizers; prepare tasks
    tokenizer_tasks = []
    skipped_count = 0
    
    for tokenizer_path in tokenizer_files:
        tokenizer_name = tokenizer_path.stem
        
        """ #skip partitioned tokenizers; we need to implement
        if 'partitioned' in tokenizer_name.lower():
            print(f" skipping {tokenizer_name} (because it is partitioned)")
            skipped_count += 1
            continue """
        #TODO 
        
        if max_samples:
            output_filename = f"{tokenizer_name}_max{max_samples}.pkl"
        else:
            output_filename = f"{tokenizer_name}_full.pkl"
        
        output_path = output_dir / output_filename
        
        #if exists, skip it
        if skip_existing and output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f" skipping {tokenizer_name} (exists: {file_size_mb:.2f} MB)")
            skipped_count += 1
            continue
        
        tokenizer_tasks.append((tokenizer_path, corpus_splits_dir, output_dir, max_samples))
    
    if not tokenizer_tasks:
        print(f"\n warning - no tokenizers to process (all skipped or already exist)")
        return []
    
    print(f"\nProcessing {len(tokenizer_tasks)} tokenizers in parallel...")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} tokenizers")
    
    #0 = autodetect
    if num_workers is None or num_workers == 0:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {num_workers} parallel workers")
    print(f"{'='*80}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    failed_count = 0
    
    with Pool(processes=num_workers) as pool:
        #TODO: fix pooled progress bar
        progress_bar = tqdm(total=len(tokenizer_tasks), desc="Pre-tokenizing", unit="tokenizer")
        
        for result in pool.imap_unordered(_pre_tokenize_worker, tokenizer_tasks):
            output_path, result_info = result
            
            if output_path is None:
                failed_count += 1
                tqdm.write(f"failed; {result_info.split(chr(10))[0]}")  
            else:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                created_files.append(output_path)
                progress_bar.set_postfix({
                    "completed": len(created_files), 
                    "failed": failed_count,
                    "size": f"{file_size_mb:.1f}MB"
                })
                tqdm.write(f"{result_info} ({file_size_mb:.2f} MB)")
            
            progress_bar.update(1)
        
        progress_bar.close()
    
    print(f"\n{'='*80}")
    print(f"Done! batch pre-tokenization complete!")
    print(f"   Created: {len(created_files)} files")
    print(f"   Skipped: {skipped_count} files (they already exist)")
    if failed_count > 0:
        print(f"   Failed: {failed_count} files")
    print(f"{'='*80}\n")
    
    return created_files


def main():
    parser = argparse.ArgumentParser(description='Pre-tokenize corpus in PARALLEL')
    parser.add_argument('--tokenizer-dir', type=Path, required=True,
                       help='dir with tokenizer .pkl files')
    parser.add_argument('--corpus-splits-dir', type=Path, required=True,
                       help='dir with train.txt, val.txt, test.txt')
    parser.add_argument('--output-dir', type=Path, default=Path('tokenized_sequences'),
                       help='dir to save tokenized sequences')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='optional: maximum number of texts to tokenize')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                       help='if enabled, skip tokenizers that already have pre-tokenized files')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                       help='overwrite the existing pre-tokenized files (=default)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of parallel workers (0 or None gives you auto-detect, default: 0)')
    
    args = parser.parse_args()
    
    batch_pre_tokenize_all_parallel(
        tokenizer_dir=args.tokenizer_dir,
        corpus_splits_dir=args.corpus_splits_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        skip_existing=args.skip_existing,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
