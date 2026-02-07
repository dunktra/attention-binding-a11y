# Reproduction Checklist

## Environment Setup
- [ ] Python 3.9+ installed
- [ ] CUDA GPU available (16GB+ recommended)
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Run `python setup_data.py` to verify environment

## Data Preparation
- [ ] Verify `data/prompts/pilot_terms.jsonl` exists (12 prompts)
- [ ] Run `python src/tokenization_audit.py` to verify token spans

## C1: Lead-Lag Emergence
- [ ] Extract binding for 160M all checkpoints (0, 15k, 30k, 60k, 90k, 120k, 140k, 143k)
- [ ] Extract binding for 1B all checkpoints
- [ ] Extract binding for 2.8B all checkpoints
- [ ] Run behavioral evaluation for all 24 model-checkpoint combinations
- [ ] Compute correlations: expected 160M r=0.333, 1B r=0.166, 2.8B r=0.338
- [ ] Generate Figure 1 (emergence curves) via `notebooks/figure1_emergence_curves.ipynb`

## C3: Few-Shot Unlockability
- [ ] Run `python src/eval_few_shot.py`
- [ ] Verify 160M step15k: zero-shot 0.333 → few-shot 0.944 (+61.1 pp)
- [ ] Verify 160M step30k: zero-shot 0.667 → few-shot 0.944 (+27.8 pp)
- [ ] Verify 1B step15k: zero-shot 0.556 → few-shot 0.944 (+38.9 pp)

## C4: Scale-Dependent Decoupling
- [ ] Verify 1B EB* plateau at step 15k (0.646) vs behavior rise to 0.806
- [ ] Generate Figure 2 (decoupling) via `notebooks/figure1_emergence_curves.ipynb`

## C5: Causal Ablation
- [ ] Run `python src/minimal_causal.py` (160M)
- [ ] Verify: top ablated −16.7% recognition, random −6.7%, bottom 0%
- [ ] Run `python src/minimal_causal_28b.py` (2.8B)
- [ ] Verify: top ablated +33.3% recognition, random 0%, bottom 0%

## Final Verification
- [ ] All figures generated and match paper
- [ ] All tables match paper values
- [ ] Run `python src/analysis_pilot.py` for summary statistics
- [ ] (Optional) Run on clean environment to verify full reproducibility
