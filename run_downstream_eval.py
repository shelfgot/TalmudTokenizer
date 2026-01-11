from pathlib import Path
from downstream_tests import DownstreamEvaluator
from test_embeddings_for_all_tokenizers import load_saved_tokenizer, infer_config_from_filename
import torch
import argparse
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run downstream evaluation on trained tokenizers')
parser.add_argument('--tokenizer-dir', type=str, 
                    default='results_full_talmud_2025-12-08_09-42-33/tokenizers',
                    help='Directory containing tokenizer .pkl files')
parser.add_argument('--skip-sage', action='store_true', default=True,
                    help='Skip SaGe tokenizers (default: True, skip if still training)')
parser.add_argument('--include-sage', dest='skip_sage', action='store_false',
                    help='Include SaGe tokenizers in evaluation')

args = parser.parse_args()

#if there's CUDA (like on the Zoo) we'll use it!
print("=" * 80)
print("SYSTEM INFO")
print("=" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print("=" * 80 + "\n")

#load all tokenizers
tokenizer_dir = Path(args.tokenizer_dir)

if not tokenizer_dir.exists():
    print(f"Fatal error: tokenizer directory does not exist: {tokenizer_dir}")
    print(f"   you need to train tokenizers first or update the path in this script.")
    exit(1)

paths = sorted(tokenizer_dir.glob("*.pkl"))

if not paths:
    print(f"fatal error: No tokenizer .pkl files found in {tokenizer_dir}")
    print(f"   the directory is supposed to contain trained tokenizer files.")
    print(f"    train tokenizers first.")
    exit(1)

print(f"Loading tokenizers from {tokenizer_dir}...")
tokenizers = []
for p in paths:
    tok = load_saved_tokenizer(p)
    if tok is not None:
        tokenizers.append((p.stem, tok))
        print(f"Loaded {p.stem}")
    else:
        print(f" Failed to load {p.stem}")

print(f"\nLoaded {len(tokenizers)} tokenizers\n")

if len(tokenizers) == 0:
    print("fatal error: no tokenizers were successfully loaded!")
    print("   check that tokenizer files are valid and can be loaded.")
    exit(1)

filtered_tokenizers = tokenizers

print("Testing all tokenizers:")
for name, _ in filtered_tokenizers:
    print(f"  → {name}")
print(f"Total tokenizers: {len(filtered_tokenizers)}\n")

from prepare_corpus_splits import load_corpus_splits
corpus_splits_path = Path("corpus_splits")
expected_corpus_size = None
if corpus_splits_path.exists():
    train_texts, val_texts, test_texts = load_corpus_splits(corpus_splits_path)
    expected_corpus_size = len(train_texts) + len(val_texts) + len(test_texts)
    print(f"approximate corpus size from splits: {expected_corpus_size} segments")
    print(f"  (Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)})\n")

#if pretoked sequences exist...
tokenized_sequences_dir = Path("tokenized_sequences")
pre_tokenized_map = {}  

if tokenized_sequences_dir.exists():
    print(f"Checking for pre-tokenized sequences in {tokenized_sequences_dir}...")
    import pickle
    import random

    if expected_corpus_size is None:
        print(" corpus_splits not found, so no access to pre-tokenized sequences")
    else:
        all_texts = train_texts + val_texts + test_texts

        for name, tok in tokenizers:
            config = infer_config_from_filename(f"{name}.pkl")
            use_dropout = config.get("use_bpe_dropout", False)


            pre_tokenized_file = tokenized_sequences_dir / f"{name}_full.pkl"
            if pre_tokenized_file.exists():
                try:
                    with open(pre_tokenized_file, 'rb') as f:
                        save_data = pickle.load(f)
                    pre_tokenized_size = save_data.get('num_texts', len(save_data.get('token_sequences', [])))
                    
                    if pre_tokenized_size != expected_corpus_size:
                        print(f" Skipping {name}_full.pkl: corpus size mismatch")
                        print(f"      Pre-tokenized: {pre_tokenized_size} segments, expected size: {expected_corpus_size} segments")
                        continue
                    
                    pre_tokenized_map[name] = pre_tokenized_file
                    print(f"  ✓ Using pre-tokenized corpus: {pre_tokenized_file.name} ({pre_tokenized_size} segments)")
                except Exception as e:
                    print(f"  Error validating {pre_tokenized_file.name}: {e}, and thus skipping")
            else:
                print(f" No pre-tokenized file for {name} (expected {pre_tokenized_file.name})")
            
else:
    print(f"  directory {tokenized_sequences_dir} not found, will tokenize on-the-fly\n")

evaluator = DownstreamEvaluator(
    annotated_corpus_path=None,  
    corpus_splits_dir="corpus_splits",  
    pre_tokenized_map=pre_tokenized_map if pre_tokenized_map else None,  
    embedding_save_dir="saved_embeddings",  
    force_retrain_embeddings=False,
)

#num_workers=None is sequential, while num_workers=0: auto-detect (CPU count - 1). the system will autodetect
evaluator.compare_all_tokenizers(filtered_tokenizers, num_workers=0)  