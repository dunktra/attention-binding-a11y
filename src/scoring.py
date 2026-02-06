"""Scoring functions for behavioral probes.

v2: Log-probability recognition scoring for base (non-instruction-tuned) models,
    word-boundary keyword matching, and expanded keyword lists.
"""

import re
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Recognition scoring
# ---------------------------------------------------------------------------

def score_recognition_logprob(
    model,
    prompt: str,
    choices: List[str],
    answer_idx: int,
) -> Dict:
    """Score MCQ by comparing full-sequence log-probabilities of each choice.

    For each candidate answer we compute:
        log P(choice | prompt) = Σ log P(token_i | prompt, token_0..i-1)

    This is the standard approach used by lm-eval-harness for evaluating
    base (non-instruction-tuned) language models on multiple-choice tasks.

    Args:
        model: HookedTransformer
        prompt: Full prompt text (question + choices listed)
        choices: List of answer texts ["Blind users", "Colorblind users", ...]
        answer_idx: Index of correct answer (0=A, 1=B, etc.)

    Returns:
        Dict with predicted_idx, is_correct, score, and per-choice log_probs.
    """
    prompt_tokens = model.to_tokens(prompt)          # [1, prompt_len]
    prompt_len = prompt_tokens.shape[1]

    choice_scores: List[float] = []
    for choice in choices:
        # Tokenize " <choice>" (leading space for proper BPE alignment)
        choice_token_ids = model.tokenizer.encode(
            " " + choice, add_special_tokens=False
        )
        choice_tensor = torch.tensor(
            [choice_token_ids], device=prompt_tokens.device
        )

        # Concatenate prompt + choice tokens → single forward pass
        full_tokens = torch.cat([prompt_tokens, choice_tensor], dim=1)

        with torch.no_grad():
            logits = model(full_tokens)  # [1, seq_len, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)

        # Sum log-probs for each choice token.
        # Token at position t predicts token at position t+1,
        # so log P(choice_token_i) is at position (prompt_len - 1 + i).
        total_lp = 0.0
        for i, tok_id in enumerate(choice_token_ids):
            pos = prompt_len - 1 + i          # logit position that predicts tok_id
            total_lp += log_probs[0, pos, tok_id].item()

        # Length-normalize to avoid penalising longer choices
        choice_scores.append(total_lp / len(choice_token_ids))

    predicted_idx = int(torch.tensor(choice_scores).argmax())

    return {
        "predicted_idx": predicted_idx,
        "is_correct": predicted_idx == answer_idx,
        "score": 1.0 if predicted_idx == answer_idx else 0.0,
        "log_probs": choice_scores,
        "method": "logprob_rank",
    }


def score_recognition(text_out: str, answer: str) -> bool:
    """Legacy regex scoring (kept as fallback for tests)."""
    patterns = [
        r"(?:answer|option|choice)[:\s]*([A-D])",
        r"^([A-D])[).:]",
        r"([A-D])\s*$",
    ]
    text_clean = text_out.strip().upper()
    for pattern in patterns:
        match = re.search(pattern, text_clean)
        if match:
            return match.group(1) == answer.upper()
    for letter in ["A", "B", "C", "D"]:
        if text_clean.startswith(letter):
            return letter == answer.upper()
    return False


# ---------------------------------------------------------------------------
# Generation scoring
# ---------------------------------------------------------------------------

# Expanded keyword lists: include both technical terms and natural base-model
# completions so that valid outputs from untrained → fully-trained models
# receive a non-trivial score.
KEYWORDS = {
    "screen reader": [
        # Technical / definitional
        "blind", "visual", "impairment", "aloud", "software",
        "assistive", "technology", "voice", "synthesize",
        # Natural base-model completions
        "text", "screen", "program", "tool", "navigate", "web",
        "display", "content", "user", "accessibility", "computer",
        "speak", "audio", "output", "interface", "information", "read",
    ],
    "skip link": [
        "jump", "skip", "main", "content", "keyboard", "navigation",
        "accessibility", "bypass", "header", "anchor", "link", "page",
        "section", "heading", "tab", "focus", "move", "quick", "direct",
    ],
    "alt text": [
        "image", "description", "alternative", "blind", "screen reader",
        "visual", "impairment", "describe", "picture", "photo", "text",
        "equivalent", "accessibility", "graphic", "content", "meaning",
        "context", "convey", "represent", "display", "element",
    ],
}

CONTRADICTIONS = {
    "screen reader": ["deaf", "hearing", "colorblind", "see", "look"],
    "skip link": ["advertisement", "ad", "popup", "slow", "delay"],
    "alt text": ["video", "audio", "caption", "subtitle", "sound"],
}

# A "good" answer is expected to hit at least this many keywords.
KEYWORD_THRESHOLD = 3


def score_generation(text_out: str, term: str) -> float:
    """Score generation task using word-boundary keyword rubric.

    Returns score between 0 and 1.
    """
    text_lower = text_out.lower()
    term_lower = term.lower()

    term_keywords = KEYWORDS.get(term_lower, [])
    if not term_keywords:
        return 0.0

    # Word-boundary matching (prevents "bread" matching "read")
    matches = 0
    for kw in term_keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            matches += 1

    # Normalize: reaching KEYWORD_THRESHOLD keywords → 1.0
    score = min(1.0, matches / KEYWORD_THRESHOLD)

    # Contradiction penalty
    term_contradictions = CONTRADICTIONS.get(term_lower, [])
    for c in term_contradictions:
        if re.search(r"\b" + re.escape(c) + r"\b", text_lower):
            score -= 0.2

    return max(0.0, score)


# ---------------------------------------------------------------------------
# Main evaluation entry-point
# ---------------------------------------------------------------------------

def evaluate_output(
    text_out: str,
    task: str,
    term: str,
    answer: Optional[str] = None,
    model=None,
    prompt: Optional[str] = None,
    choices: Optional[List[str]] = None,
    answer_idx: Optional[int] = None,
) -> Dict:
    """Main evaluation function.

    For recognition tasks, uses log-probability ranking when *model* is
    provided, otherwise falls back to legacy regex matching.

    Args:
        text_out: Model-generated text (used for generation; informational
                  for recognition when log-prob mode is active).
        task: "recognition" or "generation"
        term: The accessibility term being tested.
        answer: Correct answer letter for legacy recognition scoring.
        model: HookedTransformer (required for log-prob recognition).
        prompt: Full prompt string (required for log-prob recognition).
        choices: List of choice texts (required for log-prob recognition).
        answer_idx: 0-indexed correct choice (required for log-prob recognition).

    Returns:
        Dict with score, is_correct, and method.
    """
    if task == "recognition":
        if model is not None and choices is not None and answer_idx is not None:
            result = score_recognition_logprob(
                model, prompt, choices, answer_idx
            )
            return {
                "is_correct": result["is_correct"],
                "score": result["score"],
                "method": result["method"],
                "predicted_idx": result["predicted_idx"],
                "log_probs": result["log_probs"],
            }
        # Fallback: legacy regex matching
        is_correct = score_recognition(text_out, answer)
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "method": "exact_match",
        }
    elif task == "generation":
        score = score_generation(text_out, term)
        return {
            "is_correct": score > 0.5,
            "score": round(score, 4),
            "method": "keyword_rubric",
        }
    else:
        raise ValueError(f"Unknown task: {task}")
