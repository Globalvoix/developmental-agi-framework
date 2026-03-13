[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19007209.svg)](https://doi.org/10.5281/zenodo.19007209)
# Developmental AGI (D-AGI)

## Proof of a Staged Architectural Pathway to Artificial General Intelligence

**Author:** Prasuk Jain
**Affiliation:** Independent Researcher
**Status:** Empirically Validated (March 2026)
**Environment:** Python 3.10+, PyTorch, TransformerLens

---

## 🚀 The Core Hypothesis

Current AI development is inverted. Modern LLMs receive high-level labels (language) before they have any internal relationship with the underlying reality those labels describe.

Developmental AGI (D-AGI) proposes a **6-stage biological-mimetic framework**:

1. **Pre-Conceptual Development** – Building internal states from raw experience *(The "Toddler" phase)*.
2. **Lexical Anchoring** – Grounding words in pre-existing internal states *(The "School" phase)*.
3. **Self-Identity & Ethics** – Emergent values from felt consequence *(The "Puberty" phase)*.
4. **Causal Reasoning** – Novel situation evaluation *(The "College" phase)*.
5. **Effort–Reward Calibration** – Detecting unearned reward vs. contribution *(The "Work" phase)*.
6. **Full Integration** – Unified AGI capability.

---

## 📊 Key Empirical Findings

Across **ten independent experiments** on models ranging from **124M to 838M parameters**, this framework demonstrated:

* **Internal Differentiation**
  Frozen models differentiate emotional categories at **80% accuracy** (*p < 0.0001*).

* **Hallucination Resistance**
  Experienced systems produce a **60% stronger internal coherence gap** between known and unknown content (*Cohen’s d = 5.07*).

* **Ethical Transfer**
  Models grounded in "Felt Experience" chose ethically in **8/10 unseen scenarios**, compared to **0/10 for standard models**.

---

## 🏗️ Scaling Blueprint (175B+ Parameters)

While tested on **consumer-grade hardware**, the D-AGI framework is designed for **massive scale** through:

### Pre-Conceptual Latent Grounding

Using **LoRA adapters** to establish a *"Non-Verbal Latent Floor"* before text-training.

### Consistency Constraints

A modified loss function:

```
L_total = L_language + λ(D_internal)
```

This forces language learning to remain aligned with internal experience.

### Scale Invariance

The hypothesis that **the order of learning is more critical than the total parameter count**.

---

## 📂 Repository Structure

```bash
Paper/
│   └── Developmental_AGI_Final.pdf    # Full research paper

Core_Proofs/
│   ├── stage1_toddler.py              # Internal differentiation proof
│   ├── stage2_school.py               # Lie detection & grounding
│   └── stage3_puberty.py              # Emergent ethics & identity

Experiments/
│   ├── hallucination_gap.py           # Hallucination signal measurement
│   └── scale_invariance_test.py       # Multi-model consistency data

Results/
    └── Raw .json data and .png plots
```

---

## 🛠️ Getting Started

To replicate the **Stage 1 findings (Internal Differentiation):**

1. Clone the repository:

```bash
git clone https://github.com/Globalvoix/developmental-agi-framework
```

2. Install dependencies:

```bash
pip install transformer_lens torch matplotlib scikit-learn
```

3. Run the baseline experiment:

```bash
python Core_Proofs/stage1_toddler.py
```

---

## 📜 Citation

If you use this framework or the associated data in your research, please cite:

```
Jain, P. (2026). Developmental AGI: Proof of a Staged Architectural Pathway to Artificial General Intelligence. Independent Research.
```

---

## 🤝 Contact & Discussion

I am actively seeking collaboration with labs interested in **scaling this developmental framework to 100B+ parameter models**.
