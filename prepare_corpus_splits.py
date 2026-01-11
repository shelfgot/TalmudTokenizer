import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data import FullTalmudCorpus
import argparse
import numpy as np


def prepare_corpus_splits(
    output_dir: Path = Path("corpus_splits"),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    tractates: list = None,
    sample_ratio: float = 1.0
):

    print("=" * 80)
    print("NOW BEGINNING CORPUS SPLITS")
    print("=" * 80)
    
    # Load corpus
    print("Loading the full Talmud corpus...")
    corpus = FullTalmudCorpus(
        tractates=tractates,
        max_words_per_segment=40,
        min_words_per_segment=3
    )
    corpus.load()
    
    original_total = len(corpus.texts)
    print(f"Loaded {original_total} text segments")
    
    if sample_ratio < 1.0:
        print(f"\nSampling {sample_ratio*100:.0f}% of corpus (seed={seed})...")
        np.random.seed(seed)
        sample_size = int(original_total * sample_ratio)
        sampled_indices = np.random.choice(original_total, size=sample_size, replace=False)
        sampled_indices = np.sort(sampled_indices) 
        corpus.texts = [corpus.texts[i] for i in sampled_indices]
        print(f"Sampled a total of {len(corpus.texts)} segments ({sample_ratio*100:.0f}% of {original_total})")
    else:
        sampled_indices = None
        print(f"Using full corpus (no sampling whatsoever1)")
    
    print()
    
    print(f"Splitting corpus ({train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%)...")
    train_texts, val_texts, test_texts, train_indices, val_indices, test_indices = corpus.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    print(f"Train: {len(train_texts)} segments")
    print(f"Val: {len(val_texts)} segments")
    print(f"Test: {len(test_texts)} segments")
    print(f"Total: {len(train_texts) + len(val_texts) + len(test_texts)} segments\n")
    
    #save splits and indices
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving splits to {output_dir}/...")
    
    for name, texts in [('train', train_texts), ('val', val_texts), ('test', test_texts)]:
        path = output_dir / f'{name}.txt'
        with open(path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        print(f"  ✓ Saved {len(texts)} segments to {path}")
    
    import json
    indices_file = output_dir / 'split_indices.json'
    metadata = {
        'train_indices': train_indices.tolist(),
        'val_indices': val_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'seed': seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'total_segments': len(corpus.texts)
    }
    
    #save sampling stats
    if sample_ratio < 1.0:
        metadata['sample_ratio'] = sample_ratio
        metadata['original_total_segments'] = original_total
        metadata['sampled_total_segments'] = len(corpus.texts)
    
    with open(indices_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f" split indices saved to {indices_file}")
    
    print("\n" + "=" * 80)
    print("CORPUS SPLITS HAVE BEEN PREPARED!")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nFiles created:")
    print(f"  - train.txt ({len(train_texts)} segments)")
    print(f"  - val.txt ({len(val_texts)} segments)")
    print(f"  - test.txt ({len(test_texts)} segments)")
    print(f"  - split_indices.json (for reference)")


def load_corpus_splits(splits_dir: Path) -> tuple:
    train_path = splits_dir / 'train.txt'
    val_path = splits_dir / 'val.txt'
    test_path = splits_dir / 'test.txt'
    
    def load_file(path: Path) -> list:
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts
    
    train_texts = load_file(train_path)
    val_texts = load_file(val_path)
    test_texts = load_file(test_path)
    
    return train_texts, val_texts, test_texts


def main():
    parser = argparse.ArgumentParser(
        description="split the Talmud corpus into train/val/test and save to files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="corpus_splits",
        help="Output directory (default: corpus_splits)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--tractates",
        type=str,
        nargs="+",
        default=None,
        help="Specific tractates to use (default: all)"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Ratio of corpus to sample before splitting (1.0 = no sampling, 0.2 = 20%%, default: 1.0)"
    )
    
    args = parser.parse_args()
    
    prepare_corpus_splits(
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        tractates=args.tractates,
        sample_ratio=args.sample_ratio
    )


if __name__ == "__main__":
    main()
