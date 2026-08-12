import re
from typing import List, Dict, Any

FILLER_WORDS = {
    "um", "uh", "er", "ah", "hmm",
    "like", "you know", "basically", "actually", "i mean", "sort of", "kind of"
}

def analyze_disfluencies(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Annotate a list of timestamped words with `disfluency: True/False` flags.
    Replaces character-interpolated fabrication with word-level regex & pause analysis.
    """
    if not words:
        return words

    cleaned_words = []
    for i, w in enumerate(words):
        w_copy = dict(w)
        raw_text = str(w_copy.get("word", "")).strip()
        clean_text = re.sub(r"[^\w\s]", "", raw_text).lower()

        is_disfluency = False

        # 1. Exact filler word match
        if clean_text in FILLER_WORDS:
            is_disfluency = True

        # 2. Repeated word detection (e.g. "the the", "I I")
        if i > 0:
            prev_raw = str(words[i - 1].get("word", "")).strip()
            prev_clean = re.sub(r"[^\w\s]", "", prev_raw).lower()
            if clean_text and clean_text == prev_clean and len(clean_text) > 1:
                is_disfluency = True

        w_copy["disfluency"] = is_disfluency
        cleaned_words.append(w_copy)

    return cleaned_words
