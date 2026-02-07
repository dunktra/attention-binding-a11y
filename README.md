# Attention-Head Binding as a Mechanistic Marker of Accessibility Concept Emergence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and data for the paper *"Attention-Head Binding as a Mechanistic Marker of Accessibility Concept Emergence in Language Models"* by Khanh-Dung Tran (2026).

## Overview

This study extends prior work on accessibility knowledge across Pythia model sizes (Salas, 2026) by introducing a mechanistic attention-based metric for concept emergence.

We introduce **EB\*** (effective binding), a mechanistic interpretability metric that tracks how attention heads bind multi-token accessibility terms (e.g., "screen reader," "alt text") during training. Using the Pythia model suite (160M, 1B, 2.8B parameters) across eight training checkpoints, we demonstrate:

- **C1 (Lead-lag emergence):** Binding precedes behavioral competence (Spearman r = 0.33–0.34, p < 0.001)
- **C3 (Unlockability):** Few-shot prompting yields +61 percentage points improvement when EB\* > 0.6
- **C4 (Decoupling):** At 1B scale, binding saturates early while behavior continues improving
- **C5 (Causal regimes):** Cross-scale reversal — binding heads are necessary at 160M but *interfering* at 2.8B

## Repository Structure

```
attention-binding-a11y/
├── src/                            # Source code
│   ├── utils_model.py              # Model loading with checkpoint support
│   ├── scoring.py                  # Recognition and generation scoring
│   ├── eval_behavior.py            # Behavioral probe evaluation
│   ├── extract_attention.py        # Attention extraction, BSI/EB/EB* metrics
│   ├── tokenization_audit.py       # Tokenization span verification
│   ├── analysis_pilot.py           # Correlation and Go/No-Go analysis
│   ├── minimal_causal.py           # C5: 160M head ablation
│   ├── minimal_causal_28b.py       # C5: 2.8B head ablation
│   └── eval_few_shot.py            # C3: Few-shot unlockability testing
├── data/
│   ├── prompts/
│   │   └── pilot_terms.jsonl       # 12 prompts (3 terms × 2 tasks × 2 variants)
│   ├── results/
│   │   ├── behavioral/             # Behavioral probe scores
│   │   ├── binding/                # EB* binding metrics
│   │   ├── causal/                 # C5 ablation results
│   │   └── few_shot/               # C3 unlockability results
│   └── tokenization/               # Tokenization tables
├── config/
│   └── pilot.yaml                  # Experiment configuration
├── notebooks/
│   ├── figure1_emergence_curves.ipynb  # Figures 1 & 4
│   ├── verify_checkpoints_v2.ipynb     # Checkpoint verification
│   └── verify_setup.ipynb              # Environment check
├── figures/                        # Generated figures
├── paper/                          # Paper source (Markdown)
│   ├── main.md
│   ├── sections/
│   └── appendix/
├── tests/
│   └── test_behavioral.py          # Unit tests
├── requirements.txt
├── setup_data.py                   # Environment setup script
├── REPRODUCTION_CHECKLIST.md
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: 16GB+ VRAM for 2.8B model)

### Setup

```bash
git clone https://github.com/dunktra/attention-binding-a11y.git
cd attention-binding-a11y

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Verify environment
python setup_data.py
```

Pythia model checkpoints are downloaded automatically from HuggingFace when running experiments.

## Quick Start

### Reproduce All Main Results

```bash
# 1. Verify tokenization spans
python src/tokenization_audit.py

# 2. Extract binding metrics (repeat for each model/checkpoint)
python src/extract_attention.py 160m step120000

# 3. Run behavioral evaluation
python src/eval_behavior.py 160m step120000

# 4. C3: Few-shot unlockability
python src/eval_few_shot.py

# 5. C5: Causal ablation
python src/minimal_causal.py        # 160M
python src/minimal_causal_28b.py    # 2.8B

# 6. Summary statistics and correlations
python src/analysis_pilot.py 160m
```

### Expected Key Outputs

| Experiment | Output | Key Metric |
|-----------|--------|------------|
| C1 (Lead-lag) | `data/results/binding/*_binding.jsonl` | Spearman r = 0.33–0.34 |
| C3 (Unlockability) | `data/results/few_shot/*_few_shot.json` | +61.1 pp at 160M step15k |
| C4 (Decoupling) | `data/results/pilot_summary.csv` | 1B EB\* plateau |
| C5 (Causal) | `data/results/causal/*_causal.json` | 160M: −16.7%, 2.8B: +33.3% |

### Approximate Runtime

| Task | GPU | CPU |
|------|-----|-----|
| Tokenization audit | 5 min | 10 min |
| Single checkpoint (binding + behavior) | 15 min | 60 min |
| Full pilot (24 checkpoints) | 6 hours | 24 hours |
| C3 unlockability (3 conditions) | 30 min | 2 hours |
| C5 ablation (160M + 2.8B) | 45 min | 3 hours |
| **Total** | **~8 hours** | **~30 hours** |

## Key Results

| Claim | Finding | Section |
|-------|---------|---------|
| **C1** | EB\* precedes behavior at 160M and 2.8B (r = 0.33–0.34, p < 0.001) | §4.1 |
| **C3** | +61 pp few-shot improvement (183% relative) when EB\* > 0.6 | §4.2 |
| **C4** | 1B binding saturates at step 15k; behavior improves through step 143k | §4.3 |
| **C5** | 160M: ablation impairs (−16.7%); 2.8B: ablation helps (+33.3%) | §4.4 |

## Citation

```bibtex
@article{tran2026binding,
  title={Attention-Head Binding as a Mechanistic Marker of Accessibility
         Concept Emergence in Language Models},
  author={Tran, Khanh-Dung},
  year={2026},
  url={https://github.com/dunktra/attention-binding-a11y}
}
```

## Paper Compilation

The paper source is in `paper/` as Markdown. To compile to PDF:

```bash
# Install pandoc
sudo apt-get install pandoc texlive-latex-base texlive-latex-extra

# Compile all sections into a single PDF
cd paper
pandoc main.md sections/introduction.md sections/related_work.md \
       sections/methods.md sections/results.md sections/discussion.md \
       sections/conclusion.md appendix/raw_data.md \
       -o attention_binding_a11y.pdf \
       --pdf-engine=pdflatex \
       -V geometry:margin=1in
```

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgment of Prior Work

This work builds directly on and extends prior analysis by **Trisha Salas** on testing accessibility-related knowledge across Pythia model sizes ([Salas, 2026](https://trishasalas.com/posts/testing-accessibility-knowledge-across-pythia-model-sizes)).

Salas’ work established that accessibility concepts such as *“screen reader”* and *“alt text”* emerge behaviorally at different rates across model scales. The present study extends this line of inquiry by shifting from **behavioral evaluation** to **mechanistic analysis**, introducing **EB\*** as an attention-based binding metric to probe *how* and *when* these concepts emerge internally during training, and how their causal role changes with scale. Visit [@trishasalas' Github repo](https://github.com/trishasalas/mech-interp-research/blob/main/pythia/pythia-a11y-emergence.ipynb) for more details.
