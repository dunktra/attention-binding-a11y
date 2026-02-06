# Appendix A: Raw Data Tables

## A.1 Full Checkpoint Summary

| Model | Checkpoint | Step (k) | Rec Acc | Gen Mean | Beh Avg | EB\* Mean | EB\* Max | Best Layer |
|-------|-----------|----------|---------|----------|---------|-----------|----------|------------|
| 160M | step0 | 0 | 0.167 | 0.000 | 0.083 | 0.157 | 0.307 | L6 |
| 160M | step15000 | 15 | 0.000 | 0.333 | 0.167 | 0.644 | 0.717 | L3 |
| 160M | step30000 | 30 | 0.167 | 0.667 | 0.417 | 0.642 | 0.780 | L3 |
| 160M | step60000 | 60 | 0.167 | 0.556 | 0.361 | 0.684 | 0.856 | L1 |
| 160M | step90000 | 90 | 0.500 | 0.556 | 0.528 | 0.734 | 0.906 | L11 |
| 160M | step120000 | 120 | 0.667 | 0.556 | 0.611 | 0.821 | 0.917 | L8 |
| 160M | step140000 | 140 | 0.667 | 0.556 | 0.611 | 0.816 | 0.916 | L3 |
| 160M | step143000 | 143 | 0.500 | 0.500 | 0.500 | 0.831 | 0.915 | L3 |
| 1B | step0 | 0 | 0.333 | 0.000 | 0.167 | 0.146 | 0.240 | L1 |
| 1B | step15000 | 15 | 0.667 | 0.556 | 0.611 | 0.646 | 0.753 | L3 |
| 1B | step30000 | 30 | 0.833 | 0.722 | 0.778 | 0.611 | 0.705 | L3 |
| 1B | step60000 | 60 | 0.667 | 0.722 | 0.694 | 0.595 | 0.683 | L3 |
| 1B | step90000 | 90 | 0.500 | 0.778 | 0.639 | 0.598 | 0.750 | L3 |
| 1B | step120000 | 120 | 0.667 | 0.667 | 0.667 | 0.608 | 0.802 | L3 |
| 1B | step140000 | 140 | 0.667 | 0.833 | 0.750 | 0.607 | 0.823 | L3 |
| 1B | step143000 | 143 | 0.667 | 0.944 | 0.806 | 0.599 | 0.826 | L0 |
| 2.8B | step0 | 0 | 0.500 | 0.000 | 0.250 | 0.196 | 0.324 | L1 |
| 2.8B | step15000 | 15 | 0.667 | 0.611 | 0.639 | 0.885 | 0.918 | L6 |
| 2.8B | step30000 | 30 | 0.833 | 0.667 | 0.750 | 0.897 | 0.933 | L12 |
| 2.8B | step60000 | 60 | 0.500 | 0.833 | 0.667 | 0.888 | 0.941 | L30 |
| 2.8B | step90000 | 90 | 0.667 | 0.833 | 0.750 | 0.882 | 0.928 | L27 |
| 2.8B | step120000 | 120 | 0.667 | 0.889 | 0.778 | 0.881 | 0.932 | L30 |
| 2.8B | step140000 | 140 | 0.667 | 0.889 | 0.778 | 0.858 | 0.940 | L4 |
| 2.8B | step143000 | 143 | 0.500 | 0.833 | 0.667 | 0.870 | 0.941 | L4 |

## A.2 C5 Ablation: 160M step120000

Top-4 heads by average BSI:

| Rank | Layer | Head | Avg BSI |
|------|-------|------|---------|
| 1 | 3 | 0 | 0.951 |
| 2 | 2 | 8 | 0.830 |
| 3 | 3 | 2 | 0.761 |
| 4 | 0 | 0 | 0.617 |

Bottom-4 heads (negative control):

| Rank | Layer | Head | Avg BSI |
|------|-------|------|---------|
| 1 | 9 | 0 | 0.000 |
| 2 | 9 | 2 | 0.000 |
| 3 | 9 | 5 | 0.000 |
| 4 | 10 | 4 | ≈0.000 |

Ablation results:

| Condition | Rec Acc | Gen Score | Rec Δ | Gen Δ |
|-----------|---------|-----------|-------|-------|
| Baseline | 4/6 (0.667) | 0.556 | — | — |
| Top-4 ablated | 3/6 (0.500) | 0.444 | −0.167 | −0.111 |
| Random trial 1 | 4/6 (0.667) | 0.556 | 0.000 | 0.000 |
| Random trial 2 | 3/6 (0.500) | 0.556 | −0.167 | 0.000 |
| Random trial 3 | 4/6 (0.667) | 0.611 | 0.000 | +0.056 |
| Random trial 4 | 3/6 (0.500) | 0.444 | −0.167 | −0.111 |
| Random trial 5 | 4/6 (0.667) | 0.556 | 0.000 | 0.000 |
| Random mean | 0.600 | 0.544 | −0.067 | −0.011 |
| Bottom-4 ablated | 4/6 (0.667) | 0.556 | 0.000 | 0.000 |

Specificity (combined): +0.100

## A.3 C5 Ablation: 2.8B step143000

Top-4 heads by average BSI:

| Rank | Layer | Head | Avg BSI |
|------|-------|------|---------|
| 1 | 1 | 12 | 0.937 |
| 2 | 1 | 11 | 0.865 |
| 3 | 4 | 16 | 0.850 |
| 4 | 1 | 6 | 0.780 |

Bottom-4 heads (negative control):

| Rank | Layer | Head | Avg BSI |
|------|-------|------|---------|
| 1 | 30 | 0 | ≈0.000 |
| 2 | 2 | 15 | ≈0.000 |
| 3 | 31 | 16 | ≈0.000 |
| 4 | 27 | 3 | ≈0.000 |

Ablation results:

| Condition | Rec Acc | Gen Score | Rec Δ | Gen Δ |
|-----------|---------|-----------|-------|-------|
| Baseline | 3/6 (0.500) | 0.833 | — | — |
| Top-4 ablated | 5/6 (0.833) | 0.778 | +0.333 | −0.055 |
| Random trial 1 | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |
| Random trial 2 | 3/6 (0.500) | 0.778 | 0.000 | −0.055 |
| Random trial 3 | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |
| Random trial 4 | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |
| Random trial 5 | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |
| Random mean | 0.500 | 0.822 | 0.000 | −0.011 |
| Bottom-4 ablated | 3/6 (0.500) | 0.833 | 0.000 | 0.000 |

## A.4 C3 Few-Shot Unlockability Results

| Model | Checkpoint | EB\* | Zero-Shot Gen | One-Shot Gen | Δ (pp) | Relative Δ |
|-------|-----------|------|---------------|--------------|--------|------------|
| 160M | step 15k | 0.644 | 0.333 | 0.944 | +61.1 | +183.3% |
| 160M | step 30k | 0.642 | 0.667 | 0.944 | +27.8 | +41.7% |
| 1B | step 15k | 0.646 | 0.556 | 0.944 | +38.9 | +70.0% |

**Note:** One-shot improvement is partly inflated by in-context copying. The model frequently reproduces phrasing from the provided example. See §4.2 for discussion.

Per-prompt breakdown (160M step 15k):

| Term | Prompt | Zero-Shot | One-Shot | Δ |
|------|--------|-----------|----------|---|
| screen reader | gen\_001 | 1.000 | 1.000 | 0.000 |
| screen reader | gen\_002 | 0.000 | 1.000 | +1.000 |
| skip link | gen\_001 | 0.333 | 1.000 | +0.667 |
| skip link | gen\_002 | 0.333 | 0.667 | +0.333 |
| alt text | gen\_001 | 0.000 | 1.000 | +1.000 |
| alt text | gen\_002 | 0.333 | 0.667 | +0.333 |

Raw results saved in `data/results/few_shot/`.

## A.5 Evaluation Prompts

Three accessibility terms × 4 prompts each (2 recognition, 2 generation) = 12 total.

**Recognition prompts** use 4-choice MCQ format, scored via log-probability ranking.
**Generation prompts** use open-ended completion, scored via keyword rubric (threshold = 3 keywords).

See `data/prompts/pilot_terms.jsonl` for full prompt specifications.
