# 3. Methods

## 3.1 Models and Training Checkpoints

We use the Pythia model suite (Biderman et al., 2023), a family of autoregressive language models trained on the Pile dataset (Gao et al., 2020) with publicly available intermediate checkpoints. We study three model scales:

| Model | Parameters | Layers | Heads | Head Dim | Total Heads |
|-------|-----------|--------|-------|----------|-------------|
| Pythia-160M-deduped | 160M | 12 | 12 | 64 | 144 |
| Pythia-1B-deduped | 1B | 16 | 8 | 128 | 128 |
| Pythia-2.8B-deduped | 2.8B | 32 | 32 | 80 | 1,024 |

For each model, we evaluate eight checkpoints spanning the full training trajectory: step 0, 15k, 30k, 60k, 90k, 120k, 140k, and 143k. This provides 24 model-checkpoint combinations. All models are loaded via TransformerLens (Nanda & Bloom, 2022) using `HookedTransformer`, which provides clean access to intermediate activations and attention patterns.

## 3.2 Accessibility Terms and Evaluation Prompts

We select three multi-token web accessibility terms as our evaluation domain: **"screen reader,"** **"skip link,"** and **"alt text."** These terms were chosen because they are: (a) multi-token, requiring the model to bind constituent tokens into a coherent concept; (b) domain-specific, with clear factual ground truth; and (c) practically important for accessibility-aware AI systems.

For each term, we construct two types of evaluation prompts (12 total):

- **Recognition (6 prompts).** Four-choice multiple-choice questions testing factual knowledge (e.g., "A screen reader is primarily used by: A) Blind users B) Colorblind users C) Deaf users D) Mobility impaired users"). Scored via log-probability ranking: for each candidate answer, we compute the length-normalized log-probability $\frac{1}{|c|} \sum_{i} \log P(c_i \mid \text{prompt}, c_{<i})$ and select the highest-scoring choice. This follows the standard approach used by lm-eval-harness for base (non-instruction-tuned) models.

- **Generation (6 prompts).** Open-ended completions testing conceptual understanding (e.g., "In web accessibility, a screen reader is"). Scored via a keyword rubric: we count word-boundary matches against a curated keyword list per term (e.g., "blind," "visual," "assistive," "software" for "screen reader"), normalize to a threshold of 3 keywords, and apply contradiction penalties. This yields a score in [0, 1].

The **behavioral score** for each checkpoint is the average across all 12 prompts: $\text{Beh} = \frac{1}{2}(\text{RecAcc} + \text{GenMean})$.

## 3.3 Attention-Head Binding Metrics

### Binding Strength Index (BSI)

For a given prompt, term span tokens at positions $\{s_1, s_2, \ldots, s_k\}$, layer $l$, and head $h$, the **Binding Strength Index** measures how strongly later span tokens attend to earlier span tokens:

$$\text{BSI}_{l,h} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} A_{l,h}[s_i, s_j]$$

where $\mathcal{P} = \{(i,j) : s_i > s_j\}$ is the set of later-to-earlier token pairs within the term span, and $A_{l,h}$ is the attention pattern matrix for layer $l$, head $h$.

### Excess Binding (EB)

The **Excess Binding** at layer $l$ captures how much the best head exceeds the layer average:

$$\text{EB}_l = \max_h \text{BSI}_{l,h} - \frac{1}{H} \sum_h \text{BSI}_{l,h}$$

This measures whether binding is concentrated in specific heads (high EB) or distributed uniformly (low EB). High EB indicates specialized binding structure.

### EB\* (Aggregate Binding)

The aggregate binding metric is the maximum EB across layers:

$$\text{EB}^* = \max_l \text{EB}_l$$

EB\* serves as the primary binding metric throughout this paper. For each checkpoint, we report the mean EB\* across all 12 prompts.

### Term Span Identification

Term tokens are located in the input via exact subsequence matching of the BPE token IDs, with fallback to character-level search for aliased forms (e.g., "alternative text" for "alt text"). Multiple encoding variants are tried (bare, space-prefixed, capitalized, title-cased) to handle BPE tokenization variability.

### Memory-Efficient Extraction

Attention patterns are extracted layer-by-layer using TransformerLens's `run_with_cache` with `stop_at_layer` to limit computation and memory usage. This enables extraction on consumer GPUs even for the 2.8B model.

## 3.4 Head Ablation for Causal Testing (C5)

To test whether high-binding heads are causally necessary for task performance, we perform targeted zero-ablation: during the forward pass, the attention pattern tensor $A_{l,h}$ is set to zero for selected heads via TransformerLens forward hooks.

For each model, we identify the top-$k$ heads by average BSI across all prompts, then evaluate under four conditions:

1. **Baseline:** No ablation.
2. **Top-$k$ ablation:** Zero the $k$ highest-BSI heads.
3. **Random ablation:** Zero $k$ randomly selected heads (5 trials, averaged).
4. **Bottom-$k$ ablation:** Zero the $k$ lowest-BSI heads (negative control).

We use $k = 4$ for both models. The **specificity** of the causal effect is measured as the difference between top-$k$ and random accuracy drops.

## 3.5 Few-Shot Unlockability Testing (C3)

To test whether binding structure contains latent knowledge that prompting can unlock, we compare zero-shot and few-shot performance on generation prompts. We prepend a single worked example (one-shot) before the evaluation prompt for models where EB\* > 0.6 but baseline generation performance is low. The improvement from zero-shot to few-shot is the **unlockability score**.

## 3.6 Implementation Details

All experiments are conducted on a single NVIDIA GPU (15GB VRAM). Models are loaded in float32 precision via TransformerLens. Behavioral evaluation uses greedy decoding (temperature = 0) for generation tasks. Random baselines for ablation use a fixed seed (42) for reproducibility. Code is available at [repository URL].
