---
title: "Attention-Head Binding as a Mechanistic Marker of Accessibility Concept Emergence in Language Models"
author: "Khanh-Dung Tran"
date: "2026"
---

# Abstract

We introduce *attention-head binding*, a mechanistic interpretability metric that tracks how attention heads bind multi-token accessibility terms (e.g., "screen reader," "alt text") during training. Using the Pythia model suite (160M, 1B, 2.8B parameters) across eight training checkpoints, we show that binding strength (EB\*) consistently precedes behavioral competence on accessibility knowledge tasks (C1: lead-lag emergence; Spearman r = 0.33–0.34, p < 0.001 for 160M and 2.8B). We identify a *decoupling effect* at the 1B scale, where binding structure saturates early while behavioral performance continues improving—suggesting that larger models develop alternative representational strategies that bypass explicit attention binding (C4). Few-shot prompting unlocks latent knowledge when EB\* exceeds 0.6, yielding up to +61 percentage points improvement (183% relative gain) and near-ceiling performance (94.4%) from near-chance baselines (C3). Targeted head ablation reveals cross-scale mechanistic regimes: at 160M, high-binding heads are necessary for task performance (ablation impairs accuracy by −16.7%), while at 2.8B, the same ablation paradoxically *improves* performance by +33.3%, providing mechanistic evidence for the decoupling effect (C5). These findings establish attention binding as a diagnostic tool for tracking concept acquisition and reveal that the relationship between mechanistic structure and behavioral competence undergoes qualitative transformation across model scales.

---

**Structure:**

- §1 Introduction → `sections/introduction.md`
- §2 Related Work → `sections/related_work.md`
- §3 Methods → `sections/methods.md`
- §4 Results → `sections/results.md`
- §5 Discussion → `sections/discussion.md`
- §6 Conclusion → `sections/conclusion.md`
- Appendix → `appendix/`
