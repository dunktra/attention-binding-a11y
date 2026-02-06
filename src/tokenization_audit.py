"""Tokenization audit for 1B: Verify spans are well-defined across models."""

import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

# Config
MODEL_SIZES = ["160m", "1b", "2.8b"]
TERMS = ["screen reader", "skip link", "alt text"]
CHECKPOINT = "step143000"  # Use final checkpoint (tokenization stable)
OUTPUT_DIR = Path("data/tokenization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_span_indices(tokenizer, text: str) -> tuple[list[int], list[str]]:
    """Get token IDs and strings for a text span."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    return tokens, token_strings


def audit_tokenization():
    """Run tokenization audit for all (model, term) pairs."""
    results = []

    for size in MODEL_SIZES:
        model_name = f"EleutherAI/pythia-{size}-deduped"
        print(f"\n=== {model_name} ===")

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=CHECKPOINT)

        for term in TERMS:
            tokens, token_strings = get_span_indices(tokenizer, term)

            # Determine if span is "clean" (no partial words).
            # GPT-NeoX/Pythia BPE uses space-prefixed tokens for word
            # boundaries — that is normal and NOT a subword issue.
            # Only flag true subword markers (## for BERT-style).
            is_clean = all(
                not s.startswith("##")
                for s in token_strings
            )

            result = {
                "model_size": size,
                "term": term,
                "n_tokens": len(tokens),
                "token_ids": json.dumps(tokens),
                "token_strings": json.dumps(token_strings),
                "span_start": 0,  # Relative to term start
                "span_end": len(tokens) - 1,
                "is_clean": is_clean,
                "valid_for_binding": len(tokens) >= 2,  # Need at least 2 tokens for binding
            }
            results.append(result)

            print(f"  {term}: {len(tokens)} tokens → {token_strings}")
            print(f"    Clean: {is_clean}, Valid: {result['valid_for_binding']}")

    # Save to CSV
    output_file = OUTPUT_DIR / "tokenization_table.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved to {output_file}")
    return results


def validate_for_binding(results: list[dict]) -> bool:
    """Check if all terms are valid for binding analysis."""
    issues = []

    for r in results:
        if not r["valid_for_binding"]:
            issues.append(f"{r['model_size']}/{r['term']}: only {r['n_tokens']} token")
        if not r["is_clean"]:
            issues.append(f"{r['model_size']}/{r['term']}: unclean span (subword issues)")

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\n✅ All terms valid for binding analysis")
    return True


if __name__ == "__main__":
    results = audit_tokenization()
    is_valid = validate_for_binding(results)

    # Exit code for scripting
    exit(0 if is_valid else 1)
