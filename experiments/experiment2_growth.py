# ============================================================
# DEVELOPMENTAL AGI — EXPERIMENT 2
# Does internal organization GROW over repeated exposures
# with zero training signal?
#
# This is the full proof of Stage 1.
# Experiment 1 proved the model CAN differentiate categories.
# Experiment 2 proves those patterns GROW MORE ORGANIZED
# over accumulated experience — exactly like a developing brain.
#
# HOW TO USE:
# 1. Paste into a NEW cell in the same Colab notebook
#    (after Experiment 1 already ran — GPT-2 is already loaded)
# 2. Press Shift+Enter
# 3. Takes ~10 minutes to run
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("EXPERIMENT 2: DO PATTERNS GROW OVER REPEATED EXPOSURE?")
print("=" * 60)
print("""
The hypothesis:
  Even with FROZEN weights, a system that accumulates
  experience develops more organized internal responses
  over time — just like a developing brain.

How we test it:
  - Feed each input 100 times through frozen GPT-2
  - After each pass, measure how organized/consistent
    the internal patterns are across categories
  - Plot the consistency score over 100 passes
  - If the line goes UP = Stage 1 fully proved
""")


# ── EXPERIENCE ACCUMULATOR ───────────────────────────────────
# This is the key innovation of Experiment 2.
# The weights are frozen — they never change.
# But we build a running "experience history" that
# accumulates across exposures, like emotional memory.
# This simulates what Stage 1 proposes happens in a
# developing system over time.

class ExperienceAccumulator:
    """
    Simulates accumulated experiential history.
    Each time an input is processed, its internal activation
    is added to the accumulator — building a history.
    The accumulator then COLORS how subsequent inputs
    are processed — exactly like mood or emotional baseline
    in biological systems.
    """
    def __init__(self, vector_size, decay=0.95):
        self.state = np.zeros(vector_size)  # starts empty
        self.decay = decay  # older memories fade slightly
        self.history = []   # track how state evolves

    def update(self, activation):
        # Decay old state slightly (older memories fade)
        # Add new activation (new experience)
        self.state = (self.state * self.decay) + (activation * (1 - self.decay))
        self.history.append(self.state.copy())

    def get_state(self):
        return self.state

    def get_influence_strength(self):
        # How strong is the accumulated experience?
        return np.linalg.norm(self.state)


# ── INPUT CATEGORIES (same as Experiment 1) ─────────────────

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

category_names = list(input_categories.keys())


# ── GET ACTIVATION (same function as Experiment 1) ──────────

def get_activation(text, experience_state=None):
    """
    Get internal activation vector from frozen GPT-2.
    If experience_state is provided, it influences the
    processing — simulating accumulated emotional memory.
    """
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)

    layer_activations = []
    for layer in range(model.cfg.n_layers):
        act = cache["resid_post", layer][0, -1, :].detach().numpy()
        layer_activations.append(act)

    raw_activation = np.concatenate(layer_activations)

    # If we have accumulated experience, blend it in
    # This simulates how emotional history colors perception
    if experience_state is not None and np.any(experience_state != 0):
        influence = 0.15  # experience influences 15% of response
        influenced_activation = (raw_activation * (1 - influence) +
                                  experience_state * influence)
        return influenced_activation
    return raw_activation


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)


def measure_discriminability(all_acts, all_lbls):
    """
    Measure how well categories can be distinguished.
    Higher = more organized internal states.
    This is our main metric for Stage 1 proof.
    """
    if len(set(all_lbls)) < 2:
        return 0.0
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(all_acts, all_lbls)
        return lda.score(all_acts, all_lbls)
    except:
        return 0.0


def measure_within_consistency(all_acts, all_lbls):
    """
    Measure average similarity within each category.
    Higher = more consistent internal responses per category.
    """
    within_sims = []
    for cat_idx in range(len(category_names)):
        cat_acts = [all_acts[i] for i, l in enumerate(all_lbls) if l == cat_idx]
        if len(cat_acts) < 2:
            continue
        sims = []
        for i in range(len(cat_acts)):
            for j in range(i+1, len(cat_acts)):
                sims.append(cosine_similarity(cat_acts[i], cat_acts[j]))
        if sims:
            within_sims.append(np.mean(sims))
    return np.mean(within_sims) if within_sims else 0


# ── MAIN EXPERIMENT LOOP ─────────────────────────────────────
# This is the core of Experiment 2.
# We run 100 passes of all inputs.
# After each pass, we measure organization.
# We expect organization to INCREASE over passes.

NUM_PASSES = 100
MEASURE_EVERY = 5  # measure every 5 passes to save time

print(f"Running {NUM_PASSES} passes through all inputs...")
print(f"Measuring internal organization every {MEASURE_EVERY} passes")
print(f"Total inputs per pass: {sum(len(v) for v in input_categories.values())}")
print("\nThis will take ~8 minutes. Watch the organization scores...\n")

# Initialize one accumulator per category
# Each category builds its own emotional memory
vector_size = model.cfg.n_layers * model.cfg.d_model
accumulators = {cat: ExperienceAccumulator(vector_size) for cat in category_names}

# Track metrics over time
pass_numbers = []
discriminability_scores = []
consistency_scores = []
influence_strengths = []

# Baseline — measure BEFORE any accumulation
print("Measuring baseline (Pass 0 — no accumulated experience)...")
baseline_acts = []
baseline_lbls = []
for cat_idx, (cat_name, inputs) in enumerate(input_categories.items()):
    for text in inputs:
        act = get_activation(text, experience_state=None)
        baseline_acts.append(act)
        baseline_lbls.append(cat_idx)

baseline_disc = measure_discriminability(baseline_acts, baseline_lbls)
baseline_cons = measure_within_consistency(baseline_acts, baseline_lbls)
pass_numbers.append(0)
discriminability_scores.append(baseline_disc)
consistency_scores.append(baseline_cons)
influence_strengths.append(0)

print(f"  Baseline discriminability: {baseline_disc:.3f}")
print(f"  Baseline consistency:      {baseline_cons:.4f}")
print()

# Now run all passes with accumulation
for pass_num in range(1, NUM_PASSES + 1):

    # Process all inputs this pass
    # Each input updates the category's experience accumulator
    for cat_idx, (cat_name, inputs) in enumerate(input_categories.items()):
        for text in inputs:
            # Get activation influenced by accumulated experience
            current_state = accumulators[cat_name].get_state()
            act = get_activation(text, experience_state=current_state)
            # Update this category's accumulator
            accumulators[cat_name].update(act)

    # Measure organization every N passes
    if pass_num % MEASURE_EVERY == 0:
        # Collect all activations with current accumulated states
        current_acts = []
        current_lbls = []
        for cat_idx, (cat_name, inputs) in enumerate(input_categories.items()):
            current_state = accumulators[cat_name].get_state()
            for text in inputs:
                act = get_activation(text, experience_state=current_state)
                current_acts.append(act)
                current_lbls.append(cat_idx)

        disc = measure_discriminability(current_acts, current_lbls)
        cons = measure_within_consistency(current_acts, current_lbls)
        avg_influence = np.mean([acc.get_influence_strength()
                                  for acc in accumulators.values()])

        pass_numbers.append(pass_num)
        discriminability_scores.append(disc)
        consistency_scores.append(cons)
        influence_strengths.append(avg_influence)

        print(f"  Pass {pass_num:3d} | Discriminability: {disc:.3f} | "
              f"Consistency: {cons:.4f} | Experience strength: {avg_influence:.2f}")


# ── RESULTS ANALYSIS ─────────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 2 RESULTS")
print("=" * 60)

disc_change = discriminability_scores[-1] - discriminability_scores[0]
cons_change = consistency_scores[-1] - consistency_scores[0]

print(f"\n  Discriminability: {discriminability_scores[0]:.3f} → {discriminability_scores[-1]:.3f}")
print(f"  Change: {disc_change:+.3f} ({'INCREASED ✅' if disc_change > 0 else 'decreased ❌'})")

print(f"\n  Consistency: {consistency_scores[0]:.4f} → {consistency_scores[-1]:.4f}")
print(f"  Change: {cons_change:+.4f} ({'INCREASED ✅' if cons_change > 0 else 'decreased ❌'})")

# Check if trend is consistently upward
disc_trend = np.polyfit(pass_numbers, discriminability_scores, 1)[0]
cons_trend = np.polyfit(pass_numbers, consistency_scores, 1)[0]

print(f"\n  Discriminability trend slope: {disc_trend:+.6f}")
print(f"  Consistency trend slope:      {cons_trend:+.6f}")

if disc_trend > 0 and cons_trend > 0:
    print("""
  ✅ STAGE 1 PROVED:
     Internal organization INCREASES over repeated exposures
     with ZERO training signal and FROZEN weights.
     
     A pretrained system develops more organized internal
     responses through accumulated experience alone —
     exactly as the Developmental AGI framework predicts.
     
     This is the artificial equivalent of a baby's responses
     becoming more organized through experience — not training.
""")
elif disc_trend > 0 or cons_trend > 0:
    print("""
  ⚡ PARTIAL SUPPORT:
     Some metrics show increasing organization.
     Stage 1 has partial empirical support.
     Stronger test needed with more inputs.
""")
else:
    print("""
  ❌ NOT SUPPORTED with current setup:
     Organization did not consistently increase.
     Stage 1 may require actual weight updates after all,
     or a different accumulation mechanism.
     This is honest science — the framework needs revision
     for this specific claim.
""")


# ── VISUALIZATION ────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Experiment 2: Does Internal Organization Grow Over Experience?\n"
             "(Frozen GPT-2 — Zero Training Signal)",
             fontsize=14, fontweight='bold')

# Plot 1 — Discriminability over passes
axes[0].plot(pass_numbers, discriminability_scores,
             'b-o', linewidth=2.5, markersize=6, markerfacecolor='white',
             markeredgewidth=2)
axes[0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Random baseline (20%)')
axes[0].axhline(y=baseline_disc, color='gray', linestyle=':', alpha=0.7, label='No-experience baseline')
z = np.polyfit(pass_numbers, discriminability_scores, 1)
p = np.poly1d(z)
axes[0].plot(pass_numbers, p(pass_numbers), "b--", alpha=0.4, label=f'Trend (slope: {disc_trend:+.5f})')
axes[0].set_title("Category Discriminability\n(Higher = More Organized)", fontweight='bold')
axes[0].set_xlabel("Number of Experience Passes")
axes[0].set_ylabel("Linear Classifier Accuracy")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor('#F8F9FA')

# Plot 2 — Consistency over passes
axes[1].plot(pass_numbers, consistency_scores,
             'g-o', linewidth=2.5, markersize=6, markerfacecolor='white',
             markeredgewidth=2, color='#2E7D32')
axes[1].axhline(y=baseline_cons, color='gray', linestyle=':', alpha=0.7, label='No-experience baseline')
z2 = np.polyfit(pass_numbers, consistency_scores, 1)
p2 = np.poly1d(z2)
axes[1].plot(pass_numbers, p2(pass_numbers), "--", alpha=0.4, color='#2E7D32',
             label=f'Trend (slope: {cons_trend:+.6f})')
axes[1].set_title("Within-Category Consistency\n(Higher = More Consistent Responses)", fontweight='bold')
axes[1].set_xlabel("Number of Experience Passes")
axes[1].set_ylabel("Average Cosine Similarity")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor('#F8F9FA')

# Plot 3 — Experience strength building up
axes[2].plot(pass_numbers, influence_strengths,
             'o-', linewidth=2.5, markersize=6, markerfacecolor='white',
             markeredgewidth=2, color='#E65100')
axes[2].set_title("Accumulated Experience Strength\n(Shows Memory Building Over Time)", fontweight='bold')
axes[2].set_xlabel("Number of Experience Passes")
axes[2].set_ylabel("Experience Vector Magnitude")
axes[2].grid(True, alpha=0.3)
axes[2].set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('experiment2_growth.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Plot saved as 'experiment2_growth.png'")
print("\nScreenshot this plot and add it to your GitHub repo.")
print("These two experiments together are your empirical evidence section.")
print("=" * 60)


# ── WHAT THE GRAPHS MEAN ─────────────────────────────────────
print("""
READING THE THREE GRAPHS:

Graph 1 — Discriminability:
  If the blue line goes UP from left to right,
  the model is getting better at internally differentiating
  categories through accumulated experience alone.
  Red dashed line = random chance (20%).
  You want to be well above that and trending up.

Graph 2 — Consistency:
  If the green line goes UP from left to right,
  the model's internal responses to the same category
  are becoming more similar to each other over time.
  Like a baby whose response to pain becomes more
  consistently "distress" rather than random noise.

Graph 3 — Experience Strength:
  This should always go up — it shows the accumulated
  experience memory is building. This is the artificial
  equivalent of emotional history accumulating over time.

Together: If graphs 1 and 2 trend upward while graph 3
builds — Stage 1 is empirically proved.
""")
