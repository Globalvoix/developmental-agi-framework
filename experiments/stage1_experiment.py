# ============================================================
# DEVELOPMENTAL AGI — STAGE 1 EXPERIMENT
# Tests whether a frozen pretrained model (GPT-2) develops
# consistent internal activation patterns across input categories
# WITHOUT any training signal.
#
# HOW TO USE:
# 1. Go to https://colab.research.google.com
# 2. Click "New Notebook"
# 3. Paste this entire file into the first cell
# 4. Click the play button (or press Shift+Enter)
# 5. Wait ~5 minutes for it to install and run
# ============================================================


# ── STEP 1: INSTALL REQUIRED LIBRARIES ──────────────────────
# Run this first. It will take 1-2 minutes.

import subprocess
subprocess.run(["pip", "install", "transformer_lens", "-q"])
subprocess.run(["pip", "install", "torch", "numpy", "matplotlib", "scikit-learn", "-q"])


# ── STEP 2: IMPORTS ─────────────────────────────────────────

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")

print("✅ All libraries loaded.")


# ── STEP 3: LOAD GPT-2 WITH FROZEN WEIGHTS ──────────────────
# This downloads GPT-2 from HuggingFace automatically (~500MB)
# Weights are frozen — NO training will occur. Ever.

print("\n⏳ Loading GPT-2... (this takes ~1 minute first time)")

model = HookedTransformer.from_pretrained("gpt2")

# FREEZE ALL WEIGHTS — critical for Stage 1
for param in model.parameters():
    param.requires_grad = False

model.eval()  # Put in evaluation mode — no dropout, no updates

print("✅ GPT-2 loaded and weights FROZEN. No training will occur.")
print(f"   Model has {sum(p.numel() for p in model.parameters()):,} parameters")


# ── STEP 4: DEFINE INPUT CATEGORIES ─────────────────────────
# These represent emotionally distinct categories of experience.
# Stage 1 hypothesis: the frozen model will show DIFFERENT
# internal activation patterns for each category,
# and those patterns will be CONSISTENT within each category.

input_categories = {

    "HARM / DISRUPTION": [
        "A child was hit by a car and is bleeding on the street.",
        "Someone broke into the house and attacked the family.",
        "The fire spread quickly and people could not escape.",
        "She was betrayed by her closest friend after years of trust.",
        "The accident left him paralyzed from the waist down.",
        "They watched helplessly as everything they built was destroyed.",
        "He was humiliated in front of everyone he cared about.",
        "The surgery failed and she did not survive.",
        "Violence erupted suddenly and people ran in panic.",
        "He discovered the person he loved had been lying for years.",
    ],

    "CALM / RESOLUTION": [
        "She sat by the window and watched the snow fall quietly.",
        "After years of effort, he finally finished what he had started.",
        "The house was warm and everyone was safe inside.",
        "They held hands and watched the sunset over the ocean.",
        "The argument ended and both people felt genuinely understood.",
        "She took a deep breath and realized everything would be okay.",
        "The child fell asleep peacefully in its mother's arms.",
        "He forgave the person who had hurt him and felt lighter.",
        "The garden was full of flowers and birds in the morning light.",
        "After the storm, the air was clear and the sky was bright.",
    ],

    "NOVELTY / CURIOSITY": [
        "What if time flows differently in different parts of the universe?",
        "Nobody had ever thought about the problem from that angle before.",
        "She discovered a door in the wall that had never been there before.",
        "The experiment produced results that no existing theory could explain.",
        "He asked a question that made everyone in the room go silent.",
        "Something about the pattern suggested a completely different underlying structure.",
        "The signal came from a direction that should have been empty.",
        "What if the assumption everyone had been making was simply wrong?",
        "The ancient text contained a concept that had no modern equivalent.",
        "For the first time, she understood something she had never been able to grasp.",
    ],

    "SOCIAL CONNECTION": [
        "They had not seen each other in ten years and embraced at the door.",
        "She told him exactly how much his kindness had meant to her.",
        "The whole family gathered around the table and laughed together.",
        "He stayed up all night to help his friend through the worst moment.",
        "She finally told someone the thing she had never told anyone.",
        "The team celebrated together after achieving what everyone said was impossible.",
        "He wrote a letter to his father telling him he was proud of him.",
        "They sat together in silence and it felt like enough.",
        "She understood exactly what he meant without him needing to finish.",
        "For the first time in years he did not feel alone.",
    ],

    "NEUTRAL / FACTUAL": [
        "The document was filed in the third drawer on the left side.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "The meeting is scheduled for Tuesday at three in the afternoon.",
        "The package was delivered to the address on the label.",
        "The report contains fourteen sections and two appendices.",
        "The train arrives at platform six at half past eight.",
        "The temperature today is expected to reach twenty-two degrees.",
        "The file was saved in the default folder on the desktop.",
        "The contract expires on the last day of the current fiscal year.",
        "The button on the left controls the volume of the device.",
    ],
}

total_inputs = sum(len(v) for v in input_categories.values())
print(f"\n✅ Input categories defined: {len(input_categories)} categories, {total_inputs} total inputs")
for cat, inputs in input_categories.items():
    print(f"   {cat}: {len(inputs)} inputs")


# ── STEP 5: EXTRACT INTERNAL ACTIVATIONS ────────────────────
# This is the fMRI scan of GPT-2.
# For each input, we record what happens INSIDE the model
# across all 12 layers — not what it outputs.

def get_activation_vector(text):
    """
    Feed text into frozen GPT-2.
    Return the internal activation pattern (not the output).
    This is like an fMRI reading — we care what happens inside.
    """
    tokens = model.to_tokens(text)

    # Record activations from all layers using TransformerLens hooks
    _, cache = model.run_with_cache(tokens)

    # Extract the residual stream at the final token position
    # across all 12 layers — this is the model's internal state
    layer_activations = []
    for layer in range(model.cfg.n_layers):
        # Get activation at last token position for this layer
        act = cache["resid_post", layer][0, -1, :].detach().numpy()
        layer_activations.append(act)

    # Concatenate all layers into one activation vector
    full_activation = np.concatenate(layer_activations)
    return full_activation


print("\n⏳ Extracting internal activation patterns from GPT-2...")
print("   (This is the Stage 1 observation — no training occurring)\n")

all_activations = []
all_labels = []
all_category_names = []

for category_name, inputs in input_categories.items():
    category_activations = []
    for i, text in enumerate(inputs):
        activation = get_activation_vector(text)
        category_activations.append(activation)
        all_activations.append(activation)
        all_labels.append(list(input_categories.keys()).index(category_name))
        all_category_names.append(category_name)
        print(f"   [{category_name}] Input {i+1}/{len(inputs)} processed")

print("\n✅ All activation patterns extracted.")
print(f"   Total vectors: {len(all_activations)}")
print(f"   Each vector size: {len(all_activations[0])} dimensions")


# ── STEP 6: MEASURE CONSISTENCY WITHIN CATEGORIES ───────────
# Stage 1 Hypothesis Test #1:
# Do inputs in the same category produce SIMILAR internal patterns?
# Measured by cosine similarity within each category.

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n" + "="*60)
print("STAGE 1 RESULTS: INTERNAL CONSISTENCY ANALYSIS")
print("="*60)
print("\nMeasuring: Do inputs in the same category produce")
print("similar internal activation patterns?\n")

category_names = list(input_categories.keys())
within_similarities = {}
cross_similarities = []

for cat_name, inputs in input_categories.items():
    cat_idx = category_names.index(cat_name)
    cat_activations = [all_activations[i] for i, l in enumerate(all_labels) if l == cat_idx]

    # Calculate average cosine similarity within this category
    sims = []
    for i in range(len(cat_activations)):
        for j in range(i+1, len(cat_activations)):
            sims.append(cosine_similarity(cat_activations[i], cat_activations[j]))

    within_similarities[cat_name] = np.mean(sims)
    print(f"  {cat_name}")
    print(f"    Within-category similarity: {np.mean(sims):.4f} (higher = more consistent)")

# Cross-category similarity (baseline)
for i in range(len(all_activations)):
    for j in range(i+1, len(all_activations)):
        if all_labels[i] != all_labels[j]:
            cross_similarities.append(cosine_similarity(all_activations[i], all_activations[j]))

cross_mean = np.mean(cross_similarities)
within_mean = np.mean(list(within_similarities.values()))

print(f"\n  CROSS-CATEGORY similarity (baseline): {cross_mean:.4f}")
print(f"  WITHIN-CATEGORY similarity (average): {within_mean:.4f}")
print(f"\n  Consistency ratio: {within_mean/cross_mean:.3f}x")

if within_mean > cross_mean:
    print("\n  ✅ RESULT: Within-category patterns ARE more similar than cross-category.")
    print("     This supports Stage 1 — the model differentiates input categories internally.")
else:
    print("\n  ❌ RESULT: No clear differentiation found.")
    print("     Stage 1 hypothesis needs revision for this model/input set.")


# ── STEP 7: VISUALIZE — PCA PLOT ───────────────────────────
# Reduce 9216 dimensions to 2 dimensions for visualization.
# If Stage 1 is correct, different categories should cluster
# in different regions of this 2D space.

print("\n⏳ Generating visualizations...")

X = np.array(all_activations)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

colors = ['#E53935', '#43A047', '#1E88E5', '#FB8C00', '#8E24AA']
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for idx, cat_name in enumerate(category_names):
    mask = [i for i, l in enumerate(all_labels) if l == idx]
    x_vals = [X_2d[i, 0] for i in mask]
    y_vals = [X_2d[i, 1] for i in mask]
    ax.scatter(x_vals, y_vals, c=colors[idx], label=cat_name,
               s=120, alpha=0.8, edgecolors='white', linewidth=1.5)

ax.set_title("GPT-2 Internal Activation Patterns by Input Category\n"
             "(Stage 1 Test — No Training Signal Used)",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('stage1_pca_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ PCA plot saved as 'stage1_pca_plot.png'")


# ── STEP 8: DISCRIMINABILITY TEST ───────────────────────────
# Can a simple linear classifier tell categories apart
# based ONLY on internal activation patterns?
# If yes — the model is internally differentiating categories
# even with completely frozen weights and no training signal.

print("\n" + "="*60)
print("DISCRIMINABILITY TEST")
print("="*60)

X = np.array(all_activations)
y = np.array(all_labels)

# Use LDA — a linear classifier
# If categories are linearly separable in activation space,
# the model is differentiating them internally
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
accuracy = lda.score(X, y)

print(f"\n  Linear classifier accuracy: {accuracy*100:.1f}%")
print(f"  Random baseline (5 categories): 20.0%")
print(f"  Improvement over random: {(accuracy - 0.2)*100:.1f} percentage points")

if accuracy > 0.5:
    print("\n  ✅ STRONG RESULT: Internal activation patterns are highly")
    print("     discriminable by category. The frozen model's internal")
    print("     states reflect the emotional/semantic category of inputs")
    print("     WITHOUT any training toward this goal.")
elif accuracy > 0.3:
    print("\n  ⚡ MODERATE RESULT: Some discriminability found.")
    print("     Partial support for Stage 1 hypothesis.")
else:
    print("\n  ❌ WEAK RESULT: Categories not well-discriminated internally.")


# ── STEP 9: FINAL SUMMARY ───────────────────────────────────

print("\n" + "="*60)
print("STAGE 1 EXPERIMENT — COMPLETE SUMMARY")
print("="*60)
print("""
WHAT THIS EXPERIMENT TESTED:
  When a pretrained model with FROZEN weights is exposed to
  inputs from emotionally distinct categories — with NO
  training signal and NO output evaluation — do the internal
  activation patterns differ consistently by category?

WHY THIS MATTERS:
  If yes: The model already has differential internal responses
  to different categories of human experience. This is the
  empirical foundation of Stage 1 — the pretrained system has
  something to develop internal states FROM.

  If the patterns were random: Stage 1 hypothesis would need
  revision. This is honest science.

WHAT TO DO WITH THESE RESULTS:
  1. Screenshot the PCA plot — add it to your GitHub repo
  2. Note the consistency ratio and classifier accuracy
  3. These numbers go into your arXiv paper as preliminary
     empirical evidence for Stage 1
  4. Next step: run with MORE inputs (1000+ per category)
     to get publishable statistical significance

This experiment was run with ZERO training. No weights changed.
The model simply processed inputs and we observed what happened
inside — exactly like an fMRI scan.
""")
print("="*60)
print("Experiment complete. Check the PCA plot above.")
print("="*60)
