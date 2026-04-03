# REPAIR: Reranking via Pairwise Residual Correction for Long-Tailed Classification

<p align="center">
  <b>Beyond Logit Adjustment: A Residual Decomposition Framework for Long-Tailed Reranking</b>
</p>

<p align="center">
  <a href="#method">Method</a> &bull;
  <a href="#results">Results</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#reproducing-results">Reproducing Results</a> &bull;
  <a href="#citation">Citation</a>
</p>

---

## Overview

Long-tailed classification methods like logit adjustment apply a **fixed per-class offset** to correct frequency bias. But the correction needed between two classes varies across inputs — a fixed offset that helps one image may hurt another.

**REPAIR** decomposes the Bayes-optimal residual correction into:
- A **classwise** component (learned per-class offset with empirical Bayes shrinkage)
- A **pairwise** component (input-dependent correction based on shortlist competition)

```
r_y(x, S) = g_y(x) + a_y + (1/(k-1)) * sum_{j in S\{y}} theta^T phi(x, y, j)
              base    classwise            pairwise
```

The theory predicts *when* pairwise correction helps (threshold crossings between class pairs) and *when* it doesn't (class-separable settings). Experiments confirm gains arise precisely where the theory predicts.

## Method

<table>
<tr>
<td width="50%">

**Key idea:** When the same label pair (e.g., leopard vs. cat) requires different corrections across inputs, no fixed offset can be optimal for all. REPAIR adds a lightweight pairwise term that adapts to the competition structure on the shortlist.

**Pairwise features** `phi(x, y, j)`:
- Score gap: `g_y(x) - g_j(x)`
- Rank gap: `rank(j) - rank(y)`
- Log-frequency ratio: `log(n_y) - log(n_j)`
- Domain similarity (taxonomic / WordNet / HPO)

**Training:** Conditional log-likelihood on a calibration set, L-BFGS with L2 regularization. The base model is never modified.

</td>
<td width="50%">

**Theoretical contributions:**
- **Proposition 4.2:** When the residual is class-separable, a fixed offset recovers the Bayes ordering exactly
- **Theorem 4.3:** When contradictory pairs exist (same pair requires opposite corrections in different contexts), no fixed offset can recover the Bayes ordering
- **Threshold dispersion** `D_y`: continuous measure of how prone a class is to pairwise failure

</td>
</tr>
</table>

## Results

### Main Results (Table 1) — Hit@1 (%) conditioned on coverage, k=10

| Dataset | Base | LogitAdj | Classwise | **REPAIR** | Rare Base | **Rare REPAIR** |
|:--------|:----:|:--------:|:---------:|:----------:|:---------:|:---------------:|
| iNaturalist (8142 cls) | 47.0 | 49.0 | 48.8 | **49.0** | 37.5 | **42.2** |
| ImageNet-LT (1000 cls) | 54.4 | 61.5 | 61.8 | **62.0** | 47.4 | **60.8** |
| Places-LT (365 cls) | 44.7 | 44.5 | 45.1 | **45.5** | 46.3 | **48.2** |
| GMDB (508 diseases) | 64.0 | 70.6 | 67.4 | **72.3** | 62.9 | **79.6** |
| RareBench (508, OOD) | 59.4 | 62.5 | 66.2 | **85.6** | 41.2 | **81.2** |

### Key Findings

- **Near-class-separable regime** (vision benchmarks): REPAIR provides small but consistent gains (+0.2–0.4 Hit@1) — the classwise correction already captures most of the recoverable gap.

- **Non-class-separable regime** (rare disease): REPAIR provides large gains. On GMDB, +4.9 Hit@1 and +12.7 Rare Hit@1 over Classwise. On RareBench (OOD text-only), +19.4 Hit@1 over Classwise (bootstrap 95% CI: [+9.4, +37.5], P=1.000).

- **Quintile analysis**: Gains of REPAIR over Classwise increase monotonically from Q1 (low threshold dispersion) to Q5 (high), confirming the theory's predictions.

### Sensitivity to Shortlist Size k

| Dataset | Method | k=5 | k=10 | k=20 | k=50 |
|:--------|:-------|:---:|:----:|:----:|:----:|
| GMDB | Classwise | 77.4 | 67.4 | 56.3 | 46.0 |
| | **REPAIR** | **81.5** | **72.3** | **60.5** | **49.3** |
| RareBench | Classwise | 71.7 | 66.2 | 67.7 | 58.7 |
| | **REPAIR** | **79.3** | **85.6** | **79.0** | **73.5** |

REPAIR consistently outperforms Classwise across all shortlist sizes on all datasets.

## Installation

```bash
pip install -r requirements.txt
# numpy, scipy, matplotlib
# PyTorch required for GMDB/RareBench (classifier weight loading)
```

## Repository Structure

```
REPAIR/
├── repair/                     # Core algorithm
│   ├── core.py                 # fit(), compute_phi(), apply_scores(), evaluate()
│   └── shrinkage.py            # Empirical Bayes shrinkage for classwise offsets
├── experiments/                # Experiment scripts (one per table/figure)
│   ├── run_main_table.py       # Table 1: Main results (5 datasets, 5-seed)
│   ├── ablation.py             # Component ablation (CW-only, PW-only, REPAIR)
│   ├── k_sensitivity.py        # Table 6: Sensitivity to shortlist size k
│   ├── unconditional.py        # Table 7: Unconditional results
│   ├── bootstrap_ci.py         # Table 9: RareBench bootstrap 95% CI
│   ├── synthetic.py            # Figure 2: Synthetic validation
│   └── contradictory.py        # Contradictory pair analysis
├── figures/                    # Figure generation
│   ├── plot_style.py           # Shared matplotlib style
│   ├── gen_validation.py       # Figure 2 (synthetic)
│   ├── gen_quintile.py         # Figure 3 (quintile analysis)
│   ├── gen_ablation.py         # Figure 7 (ablation bars)
│   └── gen_k_sensitivity.py    # k-sensitivity plots
└── requirements.txt
```

## Data Setup

Set `REPAIR_DATA_ROOT` to the directory containing pre-extracted logits:

```bash
export REPAIR_DATA_ROOT=/path/to/data
```

Expected layout:
```
$REPAIR_DATA_ROOT/
├── results/imagenet_lt/imagenet_lt_logits.npz
├── results/inat/logits.npz
├── results/backbone/resnet152_places_lt_logits.npz
├── results/rarebench/rarebench_qwen_logits.npz
├── results/gmdb_classifier_norms.npy
├── qwen-vl-inference/results/inference_output/v2_kway_08b_final_{cal,test}.logits.npz
├── qwen-vl-finetune/qwenvl/data/v2_manifest.csv
├── qwen-vl-finetune/qwenvl/data/v2_disease_cards.json
└── data/imagenet_lt/ImageNet_LT_train.txt
```

## Reproducing Results

All commands run from the repository root:

```bash
# Table 1: Main results
python -m experiments.run_main_table

# Component ablation
python -m experiments.ablation

# Table 6: k-sensitivity
python -m experiments.k_sensitivity

# Table 7: Unconditional results (requires k_sensitivity output)
python -m experiments.unconditional

# Table 9: RareBench bootstrap CI
python -m experiments.bootstrap_ci

# Synthetic experiments + figures
python -m experiments.synthetic
python -m figures.gen_validation

# Quintile analysis figures
python -m figures.gen_quintile

# Ablation bar charts (requires ablation output)
python -m figures.gen_ablation

# Contradictory pair analysis
python -m experiments.contradictory inat
python -m experiments.contradictory imagenet
python -m experiments.contradictory places
python -m experiments.contradictory gmdb
python -m experiments.contradictory rarebench
```

## Citation

If you find this work useful, please cite our paper.
```bibtex
@misc{wang2026logitadjustmentresidualdecomposition,
      title={Beyond Logit Adjustment: A Residual Decomposition Framework for Long-Tailed Reranking}, 
      author={Zhanliang Wang and Hongzhuo Chen and Quan Minh Nguyen and Mian Umair Ahsan and Kai Wang},
      year={2026},
      eprint={2604.01506},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.01506}, 
}
```

## License

This project is released under the MIT License.
