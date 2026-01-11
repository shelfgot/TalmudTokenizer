"""
Loads the Talmud Bavli from Sefaria
"""

from typing import List, Dict, Optional
from pathlib import Path
import json
import re
import hashlib
import time

#this is all 37 Babylonian Talmud tractates!
BAVLI_TRACTATES = [
    "Berakhot", "Shabbat", "Eruvin", "Pesachim", "Rosh Hashanah",
    "Yoma", "Sukkah", "Beitzah", "Taanit", "Megillah",
    "Moed Katan", "Chagigah", "Yevamot", "Ketubot", "Nedarim",
    "Nazir", "Sotah", "Gittin", "Kiddushin", "Bava Kamma",
    "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot", "Shevuot",
    "Avodah Zarah", "Horayot", "Zevachim", "Menachot", "Chullin",
    "Bekhorot", "Arakhin", "Temurah", "Keritot", "Meilah",
    "Tamid", "Niddah"
]


def get_all_bavli_tractates() -> List[str]:
    return BAVLI_TRACTATES.copy()


def _extract_text_from_nested_structure(obj) -> List[str]:
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
        elif 'text' in obj:
            texts.extend(_extract_text_from_nested_structure(obj['text']))
        else:
            for value in obj.values():
                texts.extend(_extract_text_from_nested_structure(value))
    
    return texts


def get_talmud_from_sefaria(tractate: str, lang: str = "he") -> List[Dict[str, str]]:
    """
    gets a single tractate from Sefaria API.
    """
    segments = []
    
    try:
        url = f"https://www.sefaria.org/api/v3/texts/{tractate}?lang={lang}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()

            he_data = None
            
            if 'versions' in data and len(data['versions']) > 0:
                version = data['versions'][0]
                if 'text' in version:
                    he_data = version['text']
            
            if he_data is None and 'he' in data:
                he_data = data['he']
            
            if he_data is not None:
                if isinstance(he_data, list) and len(he_data) > 0:
                    if isinstance(he_data[0], list):
                        for daf_index, daf_segments in enumerate(he_data):
                            if not daf_segments or (isinstance(daf_segments, list) and len(daf_segments) == 0):
                                continue
                            
                            #get daf reference (2a, 2b, etc.); recall that dapim start at 2
                            daf_number = (daf_index // 2) + 2  
                            daf_side = 'a' if daf_index % 2 == 0 else 'b'
                            daf_ref = f"{daf_number}{daf_side}"
                            
                            for segment in daf_segments:
                                if isinstance(segment, str) and segment.strip():
                                    segments.append({
                                        'tractate': tractate,
                                        'daf': daf_ref,
                                        'text': segment.strip()
                                    })
                    else:
                        extracted_texts = _extract_text_from_nested_structure(he_data)
                        for i, text in enumerate(extracted_texts):
                            if isinstance(text, str) and text.strip():
                                segments.append({
                                    'tractate': tractate,
                                    'daf': None,
                                    'text': text.strip()
                                })
                else:
                    extracted_texts = _extract_text_from_nested_structure(he_data)
                    for i, text in enumerate(extracted_texts):
                        if isinstance(text, str) and text.strip():
                            segments.append({
                                'tractate': tractate,
                                'daf': None,
                                'text': text.strip()
                            })
            elif 'text' in data:
                extracted_texts = _extract_text_from_nested_structure(data['text'])
                for i, text in enumerate(extracted_texts):
                    if isinstance(text, str) and text.strip():
                        segments.append({
                            'tractate': tractate,
                            'daf': None,
                            'text': text.strip()
                        })
            else:
                extracted_texts = _extract_text_from_nested_structure(data)
                for i, text in enumerate(extracted_texts):
                    if isinstance(text, str) and text.strip():
                        segments.append({
                            'tractate': tractate,
                            'daf': None,
                            'text': text.strip()
                        })
        
        elif response.status_code == 404:
            print(f"tractate '{tractate}' not found in Sefaria")
        else:
            print(f"drror fetching {tractate}: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"error fetching {tractate}: {e}")
        import traceback
        traceback.print_exc()
    
    return segments


def split_long_segments(text: str, max_words: int = 40) -> List[str]:
    words = text.split()
    
    if len(words) <= max_words:
        return [text]
    
    segments = []
    current_segment = []
    current_word_count = 0
    
   
    sentence_endings = ['.', '?', '!', '׃', '׀', '׆']
    
    for i, word in enumerate(words):
        current_segment.append(word)
        current_word_count += 1
        
        ends_sentence = any(word.endswith(ending) for ending in sentence_endings)
        
        if current_word_count >= max_words or (ends_sentence and current_word_count >= max_words * 0.7):
            segment_text = ' '.join(current_segment)
            segments.append(segment_text)
            current_segment = []
            current_word_count = 0
        
        if current_word_count >= max_words * 1.5:
            segment_text = ' '.join(current_segment)
            segments.append(segment_text)
            current_segment = []
            current_word_count = 0
    
    if current_segment:
        segment_text = ' '.join(current_segment)
        segments.append(segment_text)
    
    return segments


def prepare_talmud_corpus(
    tractates: Optional[List[str]] = None,
    max_words_per_segment: int = 40,
    min_words_per_segment: int = 3,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    force_refresh: bool = False
) -> List[str]:
    
    from text_normalization import clean_corpus_formatting, remove_nekudot, normalize_whitespace
    
    if tractates is None:
        tractates = get_all_bavli_tractates()
    
    if cache_dir is None:
        cache_dir = Path(__file__).parent / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = get_cached_talmud_path(tractates, cache_dir)
    
    if use_cache and not force_refresh:
        cached_texts = load_cached_talmud(cache_path)
        if cached_texts is not None:
            print(f"loaded {len(cached_texts)} segments from cache: {cache_path}")
            return cached_texts
    
    print(f"downloading {len(tractates)} tractates from Sefaria API...")
    print("(This may take several minutes, and we'll be sure to cache for use on subsequent runs!)")
    
    all_texts = []
    
    for i, tractate in enumerate(tractates, 1):
        print(f"[{i}/{len(tractates)}] Fetching {tractate}...", end=' ', flush=True)
        
        segments = get_talmud_from_sefaria(tractate)
        
        if not segments:
            print("no segments found")
            continue
        
        processed_count = 0
        for seg in segments:
            text = seg['text']
            text = re.sub(r'<[^>]+>', '', text)
            text = clean_corpus_formatting(text)
            text = remove_nekudot(text)
            text = normalize_whitespace(text)
            
            sub_segments = split_long_segments(text, max_words=max_words_per_segment)
            
            #minimum wordcount filter
            for sub_seg in sub_segments:
                word_count = len(sub_seg.split())
                if word_count >= min_words_per_segment:
                    all_texts.append(sub_seg)
                    processed_count += 1
        
        print(f"✓ {processed_count} segments")
        
        #small delay between requests to sefaria API so they don't block us
        if i < len(tractates):
            time.sleep(0.5)
    
    print(f"\n Processed {len(all_texts)} total segments from {len(tractates)} tractates")
    
    #cache results
    if use_cache:
        cache_talmud_texts(all_texts, cache_path)
        print(f"cached results to path: {cache_path}")
    
    return all_texts


def get_cached_talmud_path(tractates: List[str], cache_dir: Path) -> Path:
    tractate_str = ','.join(sorted(tractates))
    tractate_hash = hashlib.md5(tractate_str.encode('utf-8')).hexdigest()[:8]
    
    cache_path = cache_dir / f"talmud_{tractate_hash}.json"
    return cache_path


def cache_talmud_texts(texts: List[str], cache_path: Path):

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        'texts': texts,
        'count': len(texts),
        'cached_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    print(f"cached {len(texts)} texts to path: {cache_path}")


def load_cached_talmud(cache_path: Path) -> Optional[List[str]]:
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        if 'texts' in cache_data and isinstance(cache_data['texts'], list):
            return cache_data['texts']
        else:
            print(f"invalid cache file format: {cache_path}")
            return None
            
    except Exception as e:
        print(f"error loading cache from {cache_path}: {e}")
        return None
