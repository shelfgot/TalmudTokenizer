"""
loads Biblical Hebrew text to enrich training data for partitioned tokenizers
(for future!! not implemented fully)
"""
from typing import List, Optional
from pathlib import Path
import re


def load_biblical_text_from_file(filepath: Optional[str] = None) -> List[str]:
    """
    loads Biblical Hebrew text from a file; returns as segments (Sefaria-style)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    line = re.sub(r'^BH\|[^|]+\|[^|]+\|', '', line)
                    line = re.sub(r'^<LID:BH>\s*', '', line)
                    if line:
                        lines.append(line)
            
            print(f"loaded {len(lines)} Biblical Hebrew segments from {filepath}")
            return lines
    except FileNotFoundError:
        print(f"text file not found: {filepath}")
        return []
    except Exception as e:
        print(f"some other error occurred while loading Biblical text: {e}")
        return []


def _extract_text_from_nested_structure(obj) -> List[str]:
    """
    extract text strings from nested JSON structure; very hackery but whatever
    """
    texts = []
    
    if isinstance(obj, str):
        if obj.strip():
            texts.append(obj.strip())
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(_extract_text_from_nested_structure(item))
    elif isinstance(obj, dict):
        if 'he' in obj:
            texts.extend(_extract_text_from_nested_structure(obj['he']))
        else:
            for value in obj.values():
                texts.extend(_extract_text_from_nested_structure(value))
    
    return texts


def fetch_biblical_text_from_sefaria(books: Optional[List[str]] = None) -> List[str]:
    try:
        import requests
        #if none givem all!
        if books is None:
            books = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Songs', 'Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi']
        
        all_texts = []
        
        for book in books:
            try:
                #Sefaria API endpoint
                url = f"https://www.sefaria.org/api/texts/{book}?lang=he"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'he' in data:
                        extracted_texts = _extract_text_from_nested_structure(data['he'])
                        
                        #if too long split
                        segments = []
                        for text in extracted_texts:
                            if isinstance(text, str):
                                parts = re.split(r'[\.\?\!]\s+', text)
                                parts = [p.strip() for p in parts if p.strip()]
                                segments.extend(parts)
                        
                        if segments:
                            all_texts.extend(segments)
                            print(f"recieved {len(segments)} segments from {book}")
            except Exception as e:
                print(f"error getting {book}: {e}")
                continue
        
        if all_texts:
            print(f"got {len(all_texts)} total Biblical Hebrew segments from Sefaria")
        else:
            print("no Biblical text recieved from Sefaria")
        
        return all_texts
    except Exception as e:
        print(f"bad luck, fetching from Sefaria is broken for some other reason {e}")
        return []


def get_biblical_text_enrichment(
    source: str = "file",
    filepath: Optional[str] = None,
    sefaria_books: Optional[List[str]] = None
) -> List[str]:

    texts = []
    
    if source in ['file', 'both']:
        file_texts = load_biblical_text_from_file(filepath)
        texts.extend(file_texts)
    
    if source in ['sefaria', 'both']:
        sefaria_texts = fetch_biblical_text_from_sefaria(sefaria_books)
        texts.extend(sefaria_texts)
    
    #dedup!
    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    
    return unique_texts