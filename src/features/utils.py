"""
Helper functions for feature extraction
"""
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Optional


# ============================================================
# Segment splitting utilities
# ============================================================

def split_segments_safe(model_inputs, tokenizer, debug=False, max_print=2):
    """
    Split question and answer segments from tokenized input
    
    Returns:
        question_mask, answer_mask (both shape: (B, T), dtype=bool)
    
    Strategy:
        - If token_type_ids exists: use it (0=question, 1=answer)
        - Else: split by first [SEP]
        - Remove [CLS]/[SEP]/[PAD]
        - If attention_mask exists: exclude padding positions
    """
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)
    batch_size, sequence_length = input_ids.shape

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    # Build mask for special tokens
    is_special_token = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_token_id in (cls_token_id, sep_token_id, pad_token_id):
        if special_token_id is not None:
            is_special_token |= (input_ids == special_token_id)

    # Prefer token_type_ids if available
    has_token_type_ids = ("token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None)
    if has_token_type_ids:
        token_type_ids = model_inputs["token_type_ids"]
        question_mask = (token_type_ids == 0)
        answer_mask = (token_type_ids == 1)
    else:
        # Fallback: split by first [SEP]
        if sep_token_id is None:
            raise ValueError("Tokenizer has no sep_token_id; cannot split by [SEP].")

        is_sep = (input_ids == sep_token_id)
        has_sep = is_sep.any(dim=1)

        first_sep_pos = torch.where(
            has_sep,
            is_sep.float().argmax(dim=1),
            torch.full((batch_size,), -1, device=input_ids.device)
        )

        position_indices = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, sequence_length)

        question_mask = (position_indices > 0) & (position_indices < first_sep_pos.unsqueeze(1))
        answer_mask = position_indices > first_sep_pos.unsqueeze(1)

        question_mask &= has_sep.unsqueeze(1)
        answer_mask &= has_sep.unsqueeze(1)

    # Remove special tokens
    question_mask &= ~is_special_token
    answer_mask &= ~is_special_token

    # Exclude padding positions
    if attention_mask is not None:
        valid_token_mask = attention_mask.bool()
        question_mask &= valid_token_mask
        answer_mask &= valid_token_mask

    return question_mask, answer_mask


# ============================================================
# TF-IDF utilities
# ============================================================

def compute_tfidf_cosine_similarity_per_pair(
    question_text_list: List[str],
    answer_text_list: List[str],
    fitted_vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Compute TF-IDF cosine similarity between question and answer pairs
    
    Returns:
        cosine_similarities: (N,) array of similarities
        vectorizer: Fitted TF-IDF vectorizer
    """
    if fitted_vectorizer is None:
        fitted_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
        # Fit on combined Q+A
        fitted_vectorizer.fit(question_text_list + answer_text_list)
    
    # Transform Q and A separately
    q_vectors = fitted_vectorizer.transform(question_text_list)
    a_vectors = fitted_vectorizer.transform(answer_text_list)
    
    # Compute cosine similarity per pair
    similarities = np.array([
        cosine_similarity(q_vectors[i:i+1], a_vectors[i:i+1])[0, 0]
        for i in range(len(question_text_list))
    ])
    
    return similarities, fitted_vectorizer


# ============================================================
# Content word utilities
# ============================================================

STOP_WORD_SET = set(ENGLISH_STOP_WORDS)

def content_word_ratio(text: str) -> float:
    """Compute ratio of content words (non-stopwords) in text"""
    normalized_text = (text or "").lower()
    all_words = re.findall(r"[A-Za-z']+", normalized_text)
    if len(all_words) == 0:
        return 0.0
    content_words = [w for w in all_words if w not in STOP_WORD_SET]
    return len(content_words) / len(all_words)


def compute_content_word_set_metrics(question_text: str, answer_text: str) -> Tuple[float, float]:
    """
    Compute content word Jaccard similarity and question coverage in answer
    
    Returns:
        jaccard: Jaccard similarity of content words
        coverage: Fraction of question content words found in answer
    """
    def get_content_words(text: str) -> set:
        normalized = (text or "").lower()
        words = re.findall(r"[A-Za-z']+", normalized)
        return {w for w in words if w not in STOP_WORD_SET}
    
    q_content = get_content_words(question_text)
    a_content = get_content_words(answer_text)
    
    if len(q_content) == 0 or len(a_content) == 0:
        return 0.0, 0.0
    
    intersection = q_content & a_content
    union = q_content | a_content
    
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
    coverage = len(intersection) / len(q_content) if len(q_content) > 0 else 0.0
    
    return float(jaccard), float(coverage)


# ============================================================
# Pattern-based utilities
# ============================================================

REFUSAL_REGEX_PATTERNS = [
    r"\bI (can't|cannot|won't) (comment|answer|say|discuss)\b",
    r"\bI (don't|do not) (know|have information)\b",
    r"\bI'?m not (sure|aware)\b",
    r"\bno comment\b",
    r"\bI (decline|refuse)\b",
    r"\bI (can't|cannot) (confirm|deny)\b",
    r"\bI (can't|cannot) (talk|speak) about\b",
    r"\bI (won't|will not) speculate\b",
]

CLARIFICATION_REGEX_PATTERNS = [
    r"\b(can|could|would) you clarify\b",
    r"\bcan you (please )?clarify\b",
    r"\bcould you (please )?clarify\b",
    r"\bplease clarify\b",
    r"\bclarify (that|this)\b",
    r"\bwhat do you mean\b",
    r"\bwhat (exactly )?do you mean\b",
    r"\bwhat does (that|this) mean\b",
    r"\bwhat is meant by\b",
    r"\bdefine\b",
    r"\bwhat do you mean by\b",
    r"\b(can|could|would) you (please )?specify\b",
    r"\bcan you be more specific\b",
    r"\bplease be specific\b",
    r"\bwhich (one|part|aspect|point)\b",
    r"\bwhich (exactly )?(one|part)\b",
    r"\bwhat (part|aspect)\b",
    r"\b(i )?(need|would like) more (context|details|information)\b",
    r"\bcan you provide more (context|details)\b",
    r"\bmore (context|details),? please\b",
    r"\b(i )?(don'?t|do not) understand\b",
    r"\b(i )?am confused\b",
    r"\bthat('?s| is) (unclear|not clear)\b",
    r"\bunclear\b",
    r"\b(can|could|would) you (please )?(rephrase|elaborate|explain)\b",
    r"\bcan you say that differently\b",
    r"\bcould you expand on that\b",
    r"\bwhat are you referring to\b",
    r"\bwhich (statement|claim|issue) are you referring to\b",
]


def extract_pattern_based_features_case_insensitive(answer_text: str) -> Tuple[int, int, int, int]:
    """
    Extract pattern-based features from answer (case-insensitive)
    
    Returns:
        refusal_pattern_match_count, clarification_pattern_match_count,
        question_mark_count, word_token_count
    """
    normalized = (answer_text or "").lower()
    
    refusal_count = sum(bool(re.search(p, normalized)) for p in REFUSAL_REGEX_PATTERNS)
    clarification_count = sum(bool(re.search(p, normalized)) for p in CLARIFICATION_REGEX_PATTERNS)
    question_mark_count = normalized.count("?")
    word_tokens = re.findall(r"[A-Za-z']+", normalized)
    word_count = len(word_tokens)
    
    return refusal_count, clarification_count, question_mark_count, word_count


def digit_group_count(text: str) -> int:
    """Count digit groups in text"""
    return len(re.findall(r"\d+", text or ""))


# ============================================================
# Lexicon utilities
# ============================================================

NEGATION_WORD_SET = {
    "no", "not", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor",
    "cannot", "cant", "can't",
    "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",
    "werent", "weren't",
    "won't", "wouldnt", "wouldn't",
    "shouldnt", "shouldn't", "couldnt", "couldn't",
}

HEDGE_WORD_SET = {
    "maybe", "perhaps", "probably", "possibly",
    "apparently",
    "seem", "seems", "seemed",
    "appear", "appears", "appeared",
    "roughly", "around", "about",
    "sort", "kinda", "kind", "somewhat",
    "guess", "think", "believe",
    "suggest", "suggests",
    "likely", "unlikely",
}


def tokenize_lowercase_words(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens"""
    normalized = (text or "").lower()
    return re.findall(r"[a-z']+", normalized)


def extract_answer_lexicon_ratios(answer_text: str) -> Tuple[float, float, float]:
    """
    Compute lexicon ratios for answer only
    
    Returns:
        (stopword_ratio, negation_ratio, hedge_ratio)
    """
    tokens = tokenize_lowercase_words(answer_text)
    if not tokens:
        return 0.0, 0.0, 0.0
    
    stopword_ratio = sum(t in STOP_WORD_SET for t in tokens) / len(tokens)
    negation_ratio = sum(t in NEGATION_WORD_SET for t in tokens) / len(tokens)
    hedge_ratio = sum(t in HEDGE_WORD_SET for t in tokens) / len(tokens)
    
    return float(stopword_ratio), float(negation_ratio), float(hedge_ratio)

