"""
this utility checks which words used in downstream tests are in the perfect single token words list;
it loads downstream_tests.py and extracts all Hebrew words placed in there for testing
"""

from pathlib import Path
from typing import Set
import re
import importlib.util
import sys


def load_perfect_words(perfect_file: Path) -> Set[str]:
    perfect_words = set()
    with open(perfect_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                words = [w.strip() for w in line.split(',') if w.strip()]
                perfect_words.update(words)
    return perfect_words


def extract_hebrew_words_from_file(file_path: Path) -> Set[str]:
    hebrew_words = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    #finds sequences of Hebrew/Aramaic characters)
    matches = re.findall(r'[\u0590-\u05FF]+', content)
    
    for word in matches:
        word = word.strip('"\'(),;[]{}')
        #arbitrary bound
        if len(word) >= 2:
            hebrew_words.add(word)
    
    return hebrew_words


def extract_hebrew_words_from_module(module_path: Path) -> Set[str]:
    """
    load the module and extract Hebrew words from its source
    """
    #loads dynamically
    spec = importlib.util.spec_from_file_location("downstream_tests", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Couldnt load module from {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["downstream_tests"] = module
    
    #read as string literals
    return extract_hebrew_words_from_file(module_path)


def check_test_words():
    """check these words against perfect words list"""
    script_dir = Path(__file__).parent
    perfect_file = script_dir / "perfect_single_token_words.txt"
    downstream_tests_file = script_dir / "downstream_tests.py"
    
    #the perfect file doesn't exi....
    if not perfect_file.exists():
        print(f"Error: {perfect_file} not found")
        print("make sure you're running this from the right directory!!")
        return
    
    if not downstream_tests_file.exists():
        print(f"Error: {downstream_tests_file} not found")
        return
    
    perfect_words = load_perfect_words(perfect_file)
    print(f"loaded {len(perfect_words)} perfect words from {perfect_file.name}\n")
    
    print(f"extracting Hebrew words from {downstream_tests_file.name}...")
    try:
        test_words = extract_hebrew_words_from_module(downstream_tests_file)
        print(f"found {len(test_words)} unique Hebrew words in {downstream_tests_file.name}\n")
    except Exception as e:
        print(f"Error extracting words: {e}")
        test_words = extract_hebrew_words_from_file(downstream_tests_file)
        print(f"Found {len(test_words)} unique Hebrew words (using file reading)\n")
    
    if not test_words:
        print("No Hebrew words found in downstream_tests.py")
        return
    
    perfect_in_tests = test_words & perfect_words
    not_perfect = test_words - perfect_words
    
    perfect_in_tests_sorted = sorted(perfect_in_tests)
    not_perfect_sorted = sorted(not_perfect)
    
    print("=" * 80)
    print("checking test words...")
    print("=" * 80)
    
    if perfect_in_tests_sorted:
        print(f"\n words in the good list: ({len(perfect_in_tests_sorted)}/{len(test_words)}):")
        print("-" * 80)
        words_per_line = 5
        for i in range(0, len(perfect_in_tests_sorted), words_per_line):
            words_line = perfect_in_tests_sorted[i:i+words_per_line]
            print("  " + "  ".join(f"{word:15s}" for word in words_line))
    
    if not_perfect_sorted:
        print(f"\n words in the naughty list ({len(not_perfect_sorted)}/{len(test_words)}):")
        print("-" * 80)
        words_per_line = 5
        for i in range(0, len(not_perfect_sorted), words_per_line):
            words_line = not_perfect_sorted[i:i+words_per_line]
            print("  " + "  ".join(f"{word:15s}" for word in words_line))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique test words: {len(test_words)}")
    print(f"Perfect words: {len(perfect_in_tests_sorted)} ({100*len(perfect_in_tests_sorted)/len(test_words):.1f}%)")
    print(f"Not perfect: {len(not_perfect_sorted)} ({100*len(not_perfect_sorted)/len(test_words):.1f}%)")

if __name__ == "__main__":
    check_test_words()
