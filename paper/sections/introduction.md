# 1. Introduction

Understanding how language models acquire and represent domain-specific knowledge is a central challenge in mechanistic interpretability. While behavioral evaluations reveal *what* a model knows, they provide limited insight into *how* and *when* internal representations form during training. This gap is particularly consequential for safety-critical domains such as web accessibility, where models are increasingly deployed to generate code, content, and recommendations that affect users with disabilities.

We address this gap by introducing *attention-head binding* (EB\*), a mechanistic metric that quantifies how strongly individual attention heads bind the constituent tokens of multi-token technical terms—such as "screen reader," "skip link," and "alt text"—into coherent conceptual units. Our central hypothesis is that this binding signal serves as an early, internal marker of concept acquisition that precedes externally observable behavioral competence.

We study the Pythia model suite (EleutherAI; 160M, 1B, and 2.8B parameters) across eight training checkpoints spanning the full training trajectory (step 0 through step 143,000). For each checkpoint, we measure both attention binding strength and behavioral performance on accessibility knowledge tasks (multiple-choice recognition and open-ended generation). This longitudinal design enables us to track the co-evolution of mechanistic structure and behavioral capability.

Our contributions are organized around four empirical claims (C1, C3–C5); a fifth claim concerning representational stability to prompt perturbations (C2) remains for future work (see §5.4):

1. **Lead-lag emergence (C1).** Attention binding (EB\*) rises before behavioral competence during training, establishing a temporal precedence relationship (§4.1).

2. **Unlockable latent knowledge (C3).** Models with high binding but low baseline performance contain latent knowledge that a single example can unlock, yielding up to +61 percentage points generation improvement and near-ceiling generation accuracy (94.4%) from low zero-shot baselines when EB\* > 0.6 (§4.2).

3. **Scale-dependent decoupling (C4).** At the 1B scale, binding structure saturates early while behavioral performance continues improving—revealing that larger models can develop alternative representational strategies that bypass explicit attention binding (§4.3).

4. **Cross-scale causal regimes (C5).** Targeted ablation of high-binding heads reveals opposite causal effects across scales: necessary at 160M but interfering at 2.8B, providing mechanistic evidence for the decoupling phenomenon (§4.4).

These findings establish attention binding as a diagnostic tool for concept emergence and reveal that the relationship between internal mechanistic structure and external behavioral competence undergoes qualitative transformation across model scales—a phenomenon we term the *binding-behavior decoupling effect*.

The remainder of this paper is structured as follows. §2 reviews related work in mechanistic interpretability, concept emergence, and accessibility in NLP. §3 describes our metrics, models, and experimental design. §4 presents results for each claim. §5 discusses implications, limitations, and future directions. §6 concludes.
