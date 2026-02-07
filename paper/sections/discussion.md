# 5. Discussion

## 5.1 Summary of Findings

This study introduces attention-head binding (EB\*) as a mechanistic interpretability metric and applies it longitudinally across three model scales. Our four principal findings are:

1. **Binding precedes behavior.** EB\* rises sharply in early training—often reaching 60–90% of its final value by step 15k—while behavioral competence on accessibility tasks lags behind. This lead-lag relationship is statistically significant at 160M and 2.8B (Spearman r ≈ 0.33, p < 0.001).

2. **Latent knowledge is unlockable.** When binding is high but behavior is low, few-shot prompting can bridge the gap, improving generation scores by up to +61 percentage points (183% relative gain). This suggests binding creates structural preconditions that standard behavioral probes may fail to detect.

3. **Binding and behavior decouple at scale.** The 1B model exhibits a distinctive pattern: binding saturates early while behavior continues improving for the remaining ~130k steps. This decoupling is absent at 160M and 2.8B.

4. **Causal effects reverse across scales.** At 160M, high-binding heads are necessary for performance; at 2.8B, they are *functionally superseded*. Both scales show discriminant validity of BSI (only top heads matter), but the direction of their contribution inverts.

## 5.2 Mechanistic Interpretation

### The Binding-Behavior Lifecycle

Our results suggest a developmental trajectory for attention binding across model scales:

- **Small models (160M):** Binding heads are directly incorporated into task circuits. The model's limited capacity means attention binding is a necessary computational strategy for representing multi-token concepts. Ablating these heads disrupts the only available pathway.

- **Medium models (1B):** The model develops sufficient capacity to route information through alternative pathways. Binding structure forms early but becomes increasingly redundant as training progresses and distributed representations mature. The flat binding trajectory alongside rising behavior indicates a transition to non-binding-dependent computation.

- **Large models (2.8B):** Binding achieves very high levels (EB* > 0.85) but becomes functionally superseded. High-binding heads—concentrated in early layers (L1, L4)—implement rigid attention patterns that override more flexible representations in later layers. The binary ablation pattern (top ablation improves; random/bottom ablations have no effect) reveals *massive functional redundancy*: the model has developed alternative distributed representations for concept processing, but early-layer binding heads persist as *vestigial interfering structures*. Ablating them removes an attention bottleneck, allowing more flexible late-layer representations to function fully. The larger improvement magnitude (+33.3 pp) compared to the 160M impairment (−16.7 pp) indicates the model was actively suppressed from using its full capability [Frankle & Carbin, 2019]. These heads likely served a scaffolding role during earlier training, helping the model bind multi-token terms before more flexible distributed representations developed. Their persistence at convergence reflects gradient descent's inability to prune structures that are locally optimal early in training but globally suboptimal at convergence [Frankle & Carbin, 2019].

This lifecycle parallels observations in developmental neuroscience, where early structural scaffolding can become inhibitory as more sophisticated processing develops (Huttenlocher, 2002).

### Unlockability as Evidence of Complete Latent Representations

The magnitude of the unlockability effect (+61 pp at 160M step 15k) suggests that binding structure at EB* > 0.6 represents not partial but *complete* conceptual knowledge that is simply inaccessible to standard prompting. All three tested checkpoints converge to near-identical few-shot performance (0.944), regardless of their zero-shot baselines (0.333–0.667). This ceiling convergence implies that the underlying representations are equivalently rich; differences in zero-shot behavior reflect activation failures, not knowledge gaps [Burns et al., 2022]. This parallels findings in "grokking" [Power et al., 2022], where circuits form before behavioral expression, but operates at the representational rather than algorithmic level.

### Why Early-Layer Binding Interferes at Scale

The 2.8B top-binding heads are concentrated in layers 1 and 4, much earlier than the 160M's distributed pattern (layers 0, 2, 3). In deep transformer networks, early layers typically encode local, syntactic features while later layers develop semantic and task-relevant representations [Tenney et al., 2019; Hewitt & Manning, 2019]. At 2.8B, the early-layer binding heads may "lock in" rigid token associations before later layers can contextually modulate them—effectively creating an attention bottleneck that constrains rather than supports flexible inference.

## 5.3 Implications

### For Mechanistic Interpretability

Our findings caution against assuming that high activation of a mechanistic feature implies positive causal contribution. The cross-scale reversal demonstrates that the same internal structure can play opposite functional roles depending on model capacity and training stage. Interpretability methods that rely on correlation between internal features and behavior may miss—or mischaracterize—these scale-dependent dynamics.

### For Model Development

The decoupling effect suggests that monitoring internal mechanistic markers alongside behavioral benchmarks could reveal when models are developing potentially problematic internal strategies. A model that achieves high behavioral performance despite superseded binding structure may be more fragile than one where binding and behavior are aligned.

### For Accessibility AI

The finding that accessibility concepts undergo complex developmental trajectories in language models has practical implications. Models deployed for accessibility-related tasks should be evaluated not just on behavioral accuracy but on the robustness of their internal representations—particularly at scale, where high performance may mask unstable or conflict-laden internal structure.

## 5.4 Limitations

**Evaluation scale.** Our prompt set is small (12 prompts, 3 terms). While sufficient for detecting the qualitative patterns we report, the specific numerical values (correlation coefficients, accuracy drops) should be interpreted as preliminary. Scaling to larger evaluation sets with more accessibility terms is a priority for future work.

**Domain specificity.** We study only web accessibility terms. Whether the binding-behavior dynamics generalize to other multi-token concept domains (e.g., medical terminology, legal phrases) remains to be tested.

**Ablation granularity.** Zero-ablation of attention patterns is a coarse intervention. More targeted techniques—activation patching, path patching, or causal scrubbing—could provide finer-grained understanding of how binding heads contribute to or interfere with computation.

**Model family.** All experiments use the Pythia suite. Replication across architectures (e.g., Llama, GPT-NeoX, Mistral) would strengthen generalizability claims.

**Stability (C2).** We explicitly did not test Claim C2, which posits that binding structure in mid-to-late layers exhibits greater stability across prompt perturbations than early-layer binding. While our results are consistent across multiple prompts per term, formal stability analysis—varying phrasing, word order, or context—remains for future work. This omission limits our ability to assert that EB\* captures robust conceptual representations rather than prompt-specific attention patterns.

## 5.5 Future Directions

1. **Expanded domain coverage.** Apply attention binding analysis to medical, legal, and scientific multi-token terms to test generality of the emergence patterns.

2. **Fine-grained causal analysis.** Use activation patching and circuit-level analysis to map the complete computational pathways involving binding heads at each scale.

3. **Training intervention.** Test whether artificially strengthening or weakening binding heads during training affects behavioral acquisition, enabling true causal claims about the developmental role of binding.

4. **Instruction-tuned models.** Examine whether instruction tuning realigns binding and behavior at scales where they have decoupled, potentially recovering the coupled regime.

5. **Binding as a monitoring tool.** Develop EB\* as a real-time training diagnostic that flags when binding-behavior decoupling begins, potentially signaling representational instability.
