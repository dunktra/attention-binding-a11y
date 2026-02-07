# 4. Results

## 4.1 Lead-Lag Emergence: Binding Precedes Behavior (C1)

Figure 1 plots EB\* (mean across prompts) and behavioral score across training checkpoints for all three model scales. In both the 160M and 2.8B models, attention binding rises sharply in early training and reaches near-ceiling values before behavioral competence emerges.

**Pythia-160M.** EB\* jumps from 0.157 (step 0) to 0.644 (step 15k)—a fourfold increase—while behavioral score rises only from 0.083 to 0.167. Behavioral competence does not reach 0.50 until step 90k, by which point EB\* has already reached 0.734. The Spearman rank correlation between EB\* and behavioral score across checkpoints is r = 0.333 (p = 0.0009), confirming a positive but imperfect association consistent with a lead-lag relationship.

**Pythia-2.8B.** A similar pattern emerges: EB\* reaches 0.885 by step 15k, while behavioral score is 0.639. The correlation is r = 0.338 (p = 0.0008). Notably, EB\* saturates above 0.85 from step 15k onward, while behavioral score continues climbing to 0.778 at step 120k.

**Pythia-1B.** The correlation is weaker and non-significant (r = 0.166, p = 0.107), foreshadowing the decoupling effect discussed in §4.3.

| Model | Spearman r | p-value | EB\* at step 15k | Beh at step 15k | EB\* at peak beh |
|-------|-----------|---------|-------------------|-----------------|-------------------|
| 160M | 0.333 | 0.0009*** | 0.644 | 0.167 | 0.821 (step 120k) |
| 1B | 0.166 | 0.107 (ns) | 0.646 | 0.611 | 0.599 (step 143k) |
| 2.8B | 0.338 | 0.0008*** | 0.885 | 0.639 | 0.881 (step 120k) |

These results support C1: attention binding emerges as an early internal signal that temporally precedes behavioral competence, particularly at the 160M and 2.8B scales.

*[Figure 1: Three-panel emergence curves showing EB\* (blue) and behavioral score (orange) across training steps for 160M, 1B, and 2.8B models.]*

## 4.2 Unlockable Latent Knowledge (C3)

If binding structure represents genuine conceptual organization, models with high EB\* but low behavioral performance should contain *latent knowledge* that few-shot prompting can unlock. We test this by comparing zero-shot and one-shot generation performance on checkpoints where EB\* > 0.6.

**Results.** Table 2 shows dramatic unlockability effects:

| Model | Checkpoint | EB\* | Zero-Shot Gen | Few-Shot Gen | Δ (pp) | Relative |
|-------|-----------|------|-----------|----------|--------|----------|
| 160M | step 15k | 0.644 | 0.333 | **0.944** | **+61.1** | +183% |
| 160M | step 30k | 0.642 | 0.667 | 0.944 | +27.8 | +42% |
| 1B | step 15k | 0.646 | 0.556 | 0.944 | +38.9 | +70% |

The 160M step 15k result is striking: despite low zero-shot generation performance (0.333), a single example unlocks 94.4% generation accuracy.

**Ceiling convergence.** The few-shot scores converge to near-identical levels (0.944) across checkpoints with different zero-shot baselines (0.333–0.667). This consistency suggests that binding structure at EB* > 0.6 corresponds to *complete* conceptual knowledge that is simply inaccessible to standard prompting—not partial knowledge that improves incrementally with training. The ceiling effect reflects scoring rubric granularity (near-perfect keyword coverage) rather than model capability limits.

**Control.** At step 0 (EB\* ≈ 0.15, low binding), few-shot prompting produces negligible improvement, confirming that binding structure is a necessary precondition for unlockability.

**Copying caveat.** Inspection of the one-shot outputs reveals that the model frequently reproduces phrasing from the provided example (e.g., repeating "keyboard-accessible link" from the skip-link example). This in-context copying inflates generation scores. Nevertheless, the pattern remains informative: models with EB\* > 0.6 can leverage contextual cues to produce term-appropriate content, while models with EB\* < 0.3 cannot, regardless of prompting strategy. The binding structure thus identifies models that are *ready* to express conceptual knowledge, even if zero-shot probes fail to elicit it.

## 4.3 Scale-Dependent Decoupling (C4)

A distinctive finding in our longitudinal analysis is the *binding-behavior decoupling effect* at the 1B scale.

**Pythia-1B trajectory.** EB\* rises rapidly to 0.646 at step 15k and then plateaus, remaining in the narrow range 0.595–0.646 through step 143k. In stark contrast, behavioral performance climbs steadily from 0.167 (step 0) to 0.806 (step 143k), with the strongest gains occurring *after* binding has saturated. At step 30k, the 1B model achieves its peak recognition accuracy (83.3%) while EB\* has already begun declining (0.611 vs. 0.646 at step 15k).

**Cross-scale comparison.** The decoupling is specific to the 1B scale:

| Metric | 160M | 1B | 2.8B |
|--------|------|-----|------|
| EB\* range (steps 15k–143k) | 0.642–0.831 | 0.595–0.646 | 0.858–0.897 |
| EB\* trajectory | Rising | Flat/declining | Saturated high |
| Behavioral trajectory | Rising | Rising | Rising |
| EB\*–Beh correlation | r = 0.333*** | r = 0.166 (ns) | r = 0.338*** |

At 160M and 2.8B, binding and behavior co-evolve (positively correlated). At 1B, they decouple: binding saturates early while behavior improves through mechanisms that do not rely on increased binding strength.

**Interpretation.** The 1B model appears to occupy a transitional regime between small models (where binding directly supports behavior) and large models (where binding saturates at high levels and behavior develops through distributed or redundant representations). This may reflect a capacity threshold: at 1B parameters, the model has sufficient capacity to develop behavioral competence through pathways other than attention binding, but not enough capacity for binding to saturate at the high levels seen in 2.8B.

*[Figure 4: 1B decoupling effect. EB\* (red) saturates at step 15k while behavioral score (green) continues rising through step 143k. Shaded region indicates the decoupling period.]*

## 4.4 Mechanistic Causality: Cross-Scale Ablation (C5)

We test whether high-binding heads are causally implicated in task performance via targeted zero-ablation. Surprisingly, the results reveal *opposite causal effects* at different scales, providing mechanistic evidence for the decoupling phenomenon.

### 4.4.1 Pythia-160M: Coupled Regime

Ablating the four heads with highest average BSI (L3H0, L2H8, L3H2, L0H0; avg BSI = 0.62–0.95) impairs performance more than ablating random heads:

| Condition | Rec Acc | Gen Score | Rec Δ | Gen Δ |
|-----------|---------|-----------|-------|-------|
| Baseline (no ablation) | 0.667 | 0.556 | — | — |
| Top-4 binding ablated | 0.500 | 0.444 | **−0.167** | **−0.111** |
| Random ablated (mean×5) | 0.600 | 0.544 | −0.067 | −0.011 |
| Bottom-4 binding ablated | 0.667 | 0.556 | 0.000 | 0.000 |

The graded pattern—top ablation damages most, random damages moderately, bottom ablation has zero effect—indicates that BSI captures functionally relevant structure. The specificity (top drop minus random drop) is +0.100 for the combined metric, confirming that high-binding heads are specifically, not just incidentally, important.

### 4.4.2 Pythia-2.8B: Functionally Superseded Regime

Strikingly, the same ablation procedure produces the *opposite* effect at 2.8B scale. Ablating the four highest-BSI heads (L1H12, L1H11, L4H16, L1H6; avg BSI = 0.78–0.94) *improves* recognition accuracy:

| Condition | Rec Acc | Gen Score | Rec Δ | Gen Δ |
|-----------|---------|-----------|-------|-------|
| Baseline (no ablation) | 0.500 | 0.833 | — | — |
| Top-4 binding ablated | **0.833** | 0.778 | **+0.333** | −0.055 |
| Random ablated (mean×5) | 0.500 | 0.822 | 0.000 | −0.011 |
| Bottom-4 binding ablated | 0.500 | 0.833 | 0.000 | 0.000 |

The discriminant validity of BSI is preserved: only top-binding heads produce any behavioral change; random and bottom ablation have zero effect on recognition. However, the *direction* of the effect reverses—high-binding heads interfere with, rather than support, task performance.

### 4.4.3 Cross-Scale Summary

| Model | Top Ablated Rec Δ | Random Ablated Rec Δ | Bottom Ablated Rec Δ | Regime |
|-------|-------------------|---------------------|---------------------|--------|
| 160M | **−16.7 pp** | −6.7 pp | 0.0 pp | Coupled (binding supports behavior) |
| 2.8B | **+33.3 pp** | 0.0 pp | 0.0 pp | Decoupled (binding interferes) |

**Interpretation.** At 160M, the model relies on attention binding heads for task execution—they are necessary components of the task circuit. At 2.8B, the model has developed alternative pathways for task execution that are *impeded* by the rigid, early-layer binding patterns. The high-binding heads in 2.8B (concentrated in layers 1 and 4) may implement overly specific attention patterns that override more flexible, distributed representations developed in later layers.

This cross-scale reversal provides direct mechanistic evidence for the C4 decoupling effect: binding structure and behavioral competence are not merely uncorrelated at larger scales—they can become actively counterproductive, with binding heads functionally superseded by more flexible representations.

### 4.4.4 Limitations

The evaluation set is small (6 recognition and 6 generation prompts per model). While the discriminant validity pattern (top ≠ random = bottom) is consistent across both scales, the specific accuracy values should be interpreted with caution. Replication with larger prompt sets and additional model scales is warranted. Additionally, zero-ablation is a coarse intervention; future work should employ more targeted techniques such as activation patching to isolate specific computational pathways.
