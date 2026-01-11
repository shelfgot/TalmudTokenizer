import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from collections import Counter
import argparse


def extract_top_words_from_file(
    input_file: Path,
    output_file: Path,
    top_n: int = 50000
):

    print("=" * 80)
    print("EXTRACTING TOP WORDS FROM CORPUS FILE")
    print("=" * 80)
    
    print(f"Loading corpus from {input_file}...")
    texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    
    print(f"Loaded {len(texts)} text segments\n")
    
    print("Counting word frequencies...")
    word_counts = Counter()
    
    for text in texts:
        words = text.split()
        words = [w for w in words if len(w) >= 2 and not all(c in '.,;:!?()[]{}' for c in w)]
        word_counts.update(words)
    
    print(f"Found {len(word_counts)} unique words\n")
    
    top_words = [word for word, count in word_counts.most_common(top_n)]
    
    print(f"Top {len(top_words)} words:")
    print(f"  Most frequent: '{top_words[0]}' ({word_counts[top_words[0]]} occurrences)")
    print(f"  Least frequent in top {top_n}: '{top_words[-1]}' ({word_counts[top_words[-1]]} occurrences)\n")
    
    print(f"Formatting output...")
    lines = []
    for i in range(0, len(top_words), 10):
        chunk = top_words[i:i+10]
        line = ','.join(chunk)
        lines.append(line)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n Saved top {len(top_words)} words to: {output_file}")
    print(f"  Format: {len(lines)} lines, 10 words per line (comma-separated)")
    
    #save with frequencies as well for reference
    with open(freq_file, 'w', encoding='utf-8') as f:
        for i, word in enumerate(top_words, 1):
            f.write(f"{i:4d}. {word:20s} ({word_counts[word]:,} occurrences)\n")
    
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="extract top words from a corpus file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="input corpus file (one sentence per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output file path"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50000,
        help="number of top words to extract (the default is 50000)"
    )
    
    args = parser.parse_args()
    
    extract_top_words_from_file(
        input_file=Path(args.input),
        output_file=Path(args.output),
        top_n=args.top_n
    )


if __name__ == "__main__":
    main()
