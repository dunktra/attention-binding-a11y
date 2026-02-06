# 2. Related Work

## 2.1 Mechanistic Interpretability

Mechanistic interpretability seeks to reverse-engineer the computational structure of neural networks into human-understandable components (Olah et al., 2020; Elhage et al., 2021). Within transformer language models, attention heads have been identified as key functional units: induction heads support in-context learning (Olsson et al., 2022), while specialized heads perform syntactic operations such as subject-verb agreement (Clark et al., 2019; Voita et al., 2019). Our work extends this line by identifying attention heads that bind multi-token concepts, using binding strength as a developmental marker rather than a static feature.

## 2.2 Concept Emergence During Training

The study of how knowledge emerges during training has gained traction through training dynamics analyses. Pythia (Biderman et al., 2023) provides a controlled suite of models with public intermediate checkpoints, enabling longitudinal study. Prior work has examined the emergence of factual knowledge (Swayamdipta et al., 2020), syntactic competence (Choshen et al., 2022), and reasoning abilities (Wei et al., 2022) during training. Our contribution is to track a *mechanistic* signal—attention binding—alongside behavioral competence, revealing that internal structure can precede, decouple from, or even antagonize external capability depending on model scale.

## 2.3 Attention Head Ablation and Causal Analysis

Head ablation (zeroing or mean-ablating attention outputs) is a standard technique for assessing the causal importance of individual heads (Voita et al., 2019; Michel et al., 2019). Recent work has refined this approach through activation patching (Wang et al., 2023) and path patching (Goldowsky-Dill et al., 2023). We adopt simple zero-ablation of attention patterns for transparency and reproducibility, finding that even this coarse intervention reveals interpretable cross-scale structure.

## 2.4 Accessibility in NLP

Web accessibility standards (WCAG; W3C, 2018) define requirements for making digital content usable by people with disabilities. While NLP systems are increasingly used to generate web content, accessibility-aware evaluation of language models remains limited. Prior work has examined bias in assistive technology descriptions (Trewin et al., 2019) and accessibility of AI-generated content (Gleason et al., 2020). Our work is, to our knowledge, the first to use accessibility concepts as a domain for studying mechanistic concept emergence in language models—chosen because these terms are multi-token, domain-specific, and have clear ground-truth evaluations.
