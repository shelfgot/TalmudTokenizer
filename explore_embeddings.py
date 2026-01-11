"""for testing embeddings manually"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from embedding_eval import SimpleWord2Vec
from pathlib import Path
import argparse


def list_available_embeddings(embeddings_dir: Path):
    if not embeddings_dir.exists():
        return []
    
    embedding_files = sorted(embeddings_dir.glob("*_embeddings.pkl"))
    return embedding_files


def load_embedding(filepath: Path) -> SimpleWord2Vec:
    return SimpleWord2Vec.load(filepath)


def interactive_explorer(model: SimpleWord2Vec, model_name: str):
    """"designed as interactive loop"""
    print(f"\n{'='*80}")
    print(f"Embedding Explorer: {model_name}")
    print(f"{'='*80}")
    print(f"Vocabulary size: {len(model.vocab)}")
    print(f"Embedding dimension: {model.embedding_dim}")
    print(f"\nCommands:")
    print(f"  <word>           - Find most similar words")
    print(f"  analogy a b c    - Solve analogy: a is to b as c is to ?")
    print(f"  vocab            - Show sample vocabulary")
    print(f"  stats            - Show statistics")
    print(f"  help             - Show this help")
    print(f"  quit/exit        - Exit")
    print(f"{'='*80}\n")
    
    while True:
        try:
            query = input("> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("bye bye!")
                break
            
            if query.lower() == 'help':
                print("\nCommands:")
                print("  <word>           - Find most similar words")
                print("  analogy a b c    - Solve analogy: a is to b as c is to ?")
                print("  vocab            - Show sample vocabulary")
                print("  stats            - Show statistics")
                print("  quit/exit        - Exit\n")
                continue
            
            if query.lower() == 'vocab':
                vocab_list = sorted(list(model.vocab.keys()))[:50]
                print(f"\nSample vocabulary (showing 50 of {len(model.vocab)}):")
                for i in range(0, len(vocab_list), 10):
                    print("  " + "  ".join(vocab_list[i:i+10]))
                print()
                continue
            
            if query.lower() == 'stats':
                print(f"\nStatistics:")
                print(f"  Vocabulary size: {len(model.vocab)}")
                print(f"  Embedding dimension: {model.embedding_dim}")
                print(f"  Window size: {model.window_size}")
                print(f"  Negative samples: {model.negative_samples}")
                print()
                continue
            
            parts = query.split()
            if len(parts) == 4 and parts[0].lower() == 'analogy':
                a, b, c = parts[1], parts[2], parts[3]
                print(f"\nAnalogy: {a} is to {b} as {c} is to WHAT")
                results = model.analogy(a, b, c, top_k=10)
                if results:
                    print("Top results:")
                    for i, (token, score) in enumerate(results, 1):
                        print(f"  {i}. {token} (score: {score:.4f})")
                else:
                    print("  No results found. Make sure all three words are in the vocabulary1")
                print()
                continue
            
            print(f"\nMost similar to your query '{query}':")
            results = model.most_similar(query, top_k=15)
            
            if results:
                print("Top results:")
                for i, (token, similarity) in enumerate(results, 1):
                    print(f"  {i:2d}. {token:20s} (similarity: {similarity:.4f})")
            else:
                suggestions = model.get_suggestions(query, max_suggestions=5)
                if suggestions:
                    print(f"  Word not found. Is it possible you meant:")
                    for sug in suggestions:
                        print(f"    - {sug}")
                else:
                    print(f"  Word '{query}' not found in vocabulary.")
            print()
            
        except KeyboardInterrupt:
            print("\n\nbye bye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive embedding explorer")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("saved_embeddings"),
        help="Directory containing saved embeddings (default: saved_embeddings)"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Specific embedding file to load (e.g., 'bpe_unified_nodropout_embeddings.pkl')"
    )
    
    args = parser.parse_args()
    
    # List available embeddings
    available = list_available_embeddings(args.embeddings_dir)
    
    if not available:
        print(f" no embeddings found in {args.embeddings_dir}")
        print(f"   Make sure you've run downstream evaluation with embedding_save_dir set.")
        return
    
    # Select embedding
    if args.embedding:
        # User specified a file
        if args.embedding.endswith('.pkl'):
            filepath = args.embeddings_dir / args.embedding
        else:
            filepath = args.embeddings_dir / f"{args.embedding}_embeddings.pkl"
        
        if not filepath.exists():
            print(f" embedding file not found at {filepath}")
            return
    else:
        print(f"\nAvailable embeddings for selection ({len(available)}):")
        for i, emb_file in enumerate(available, 1):
            model_name = emb_file.stem.replace('_embeddings', '')
            print(f"  {i}. {model_name}")
        
        while True:
            try:
                choice = input(f"\nSelect embedding (1-{len(available)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    filepath = available[idx]
                    break
                else:
                    print(f"enter a number between 1 and {len(available)}")
            except ValueError:
                print("enter a valid number!")
            except KeyboardInterrupt:
                return
    try:
        print(f"\nLoading {filepath.name}...")
        model = load_embedding(filepath)
        model_name = filepath.stem.replace('_embeddings', '')
        interactive_explorer(model, model_name)
    except Exception as e:
        print(f"error loading embedding: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()