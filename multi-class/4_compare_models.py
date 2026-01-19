"""
Step 4: Compare Binary vs 3-Class Models
Shows trade-offs between the two approaches

Compares:
- Binary: Ultra-safe artifact filter (99.88% seizure preservation)
- 3-Class: Active seizure detection + artifact filtering

Author: Ceribell Multi-Class System
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

print("="*70)
print("STEP 4: COMPARING BINARY VS 3-CLASS MODELS")
print("Trade-off Analysis")
print("="*70)

# ============================================================================
# LOAD BINARY MODEL RESULTS
# ============================================================================

print("\n" + "="*70)
print("LOADING BINARY MODEL RESULTS")
print("="*70)

binary_results_path = '../full_dataset/results/cross_validation_results.npz'

if not os.path.exists(binary_results_path):
    print(f"\n[WARNING] Binary model results not found")
    print(f"Expected: {binary_results_path}")
    print("\nUsing default values from previous run...")
    
    binary_results = {
        'seizure_preservation': 0.9988,
        'seizures_blocked': 0.0012,
        'background_filtering': 0.0072,
        'total_seizures': 2404,
        'total_background': 13800
    }
else:
    print(f"Loading: {binary_results_path}")
    data = np.load(binary_results_path, allow_pickle=True)
    
    # Calculate metrics from binary results
    binary_results = {
        'seizure_preservation': float(np.mean(data['seizure_clean_rate'])),
        'seizures_blocked': float(np.mean(data['seizure_artifact_rate'])),
        'background_filtering': float(np.mean(data['background_artifact_rate'])),
        'total_seizures': int(data['num_seizures']),
        'total_background': int(data['num_background'])
    }
    
    print(f"[OK] Binary Model Results:")
    print(f"  * Seizure Preservation: {100*binary_results['seizure_preservation']:.2f}%")
    print(f"  * Seizures Blocked: {100*binary_results['seizures_blocked']:.2f}%")
    print(f"  * Background Filtering: {100*binary_results['background_filtering']:.2f}%")

# ============================================================================
# LOAD 3-CLASS MODEL RESULTS
# ============================================================================

print("\n" + "="*70)
print("LOADING 3-CLASS MODEL RESULTS")
print("="*70)

threeclass_results_path = './results/evaluation_metrics.json'

if not os.path.exists(threeclass_results_path):
    print(f"\n[ERROR] 3-class model results not found")
    print(f"Expected: {threeclass_results_path}")
    print("\nPlease run evaluation first:")
    print("  python 3_evaluate_3class.py --model_path ./models/best_3class_model.pth --test_data ../preprocessed/dev.npz")
    exit(1)

print(f"Loading: {threeclass_results_path}")
with open(threeclass_results_path, 'r') as f:
    threeclass_results = json.load(f)

print(f"\n[OK] 3-Class Model Results:")
print(f"  * Overall Accuracy: {100*threeclass_results['overall_accuracy']:.2f}%")
print(f"  * Background Recall: {100*threeclass_results['per_class_metrics']['Background']['recall']:.2f}%")
print(f"  * Seizure Recall: {100*threeclass_results['per_class_metrics']['Seizure']['recall']:.2f}%")
print(f"  * Seizure Precision: {100*threeclass_results['per_class_metrics']['Seizure']['precision']:.2f}%")

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: BINARY VS 3-CLASS")
print("="*70)

print("\n" + "="*70)
print(f"{'Metric':<30} {'Binary':<15} {'3-Class':<15} {'Difference':<15}")
print("="*70)

# Seizure handling
binary_sz_pres = binary_results['seizure_preservation'] * 100
threeclass_sz_recall = threeclass_results['per_class_metrics']['Seizure']['recall'] * 100
diff_sz = threeclass_sz_recall - binary_sz_pres

print(f"{'Seizure Handling':<30} {binary_sz_pres:.2f}%{'':<8} {threeclass_sz_recall:.2f}%{'':<8} {diff_sz:+.2f}%")
print(f"  Binary: Preservation (passes through filter)")
print(f"  3-Class: Detection (active identification)")

# Background handling
binary_bg_clean = (1 - binary_results['background_filtering']) * 100
threeclass_bg_recall = threeclass_results['per_class_metrics']['Background']['recall'] * 100
diff_bg = threeclass_bg_recall - binary_bg_clean

print(f"\n{'Background Correct':<30} {binary_bg_clean:.2f}%{'':<8} {threeclass_bg_recall:.2f}%{'':<8} {diff_bg:+.2f}%")

# Seizure precision
threeclass_sz_precision = threeclass_results['per_class_metrics']['Seizure']['precision'] * 100
print(f"\n{'Seizure Precision':<30} {'N/A':<15} {threeclass_sz_precision:.2f}%{'':<8} {'New metric'}")

# Overall accuracy
binary_overall = ((binary_results['seizure_preservation'] * binary_results['total_seizures'] +
                  (1 - binary_results['background_filtering']) * binary_results['total_background']) /
                 (binary_results['total_seizures'] + binary_results['total_background'])) * 100
threeclass_overall = threeclass_results['overall_accuracy'] * 100
diff_overall = threeclass_overall - binary_overall

print(f"\n{'Overall Accuracy':<30} {binary_overall:.2f}%{'':<8} {threeclass_overall:.2f}%{'':<8} {diff_overall:+.2f}%")

print("="*70)

# ============================================================================
# TRADE-OFF ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("TRADE-OFF ANALYSIS")
print("="*70)

print("\n[BINARY MODEL - ARTIFACT FILTER]")
print(f"  Strengths:")
print(f"    ✓ Ultra-safe seizure preservation ({binary_sz_pres:.2f}%)")
print(f"    ✓ Very few false negatives ({binary_results['seizures_blocked']*100:.2f}%)")
print(f"    ✓ High confidence for clinical deployment")
print(f"  Limitations:")
print(f"    - Doesn't detect seizures (only preserves them)")
print(f"    - Requires downstream seizure detector")
print(f"    - Two-stage pipeline")

print("\n[3-CLASS MODEL - MULTI-TASK DETECTOR]")
print(f"  Strengths:")
print(f"    ✓ Active seizure detection ({threeclass_sz_recall:.2f}% recall)")
print(f"    ✓ Unified artifact removal + detection")
print(f"    ✓ Single-stage pipeline")
print(f"    ✓ Seizure precision: {threeclass_sz_precision:.2f}%")
print(f"  Limitations:")
print(f"    - Lower seizure handling ({threeclass_sz_recall:.2f}% vs {binary_sz_pres:.2f}%)")
print(f"    - More false negatives for seizures")
print(f"    - More complex to optimize")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("CLINICAL DEPLOYMENT RECOMMENDATIONS")
print("="*70)

print("\n[OPTION A: USE BINARY MODEL (SAFEST)]")
print("  When to use:")
print("    * Safety is paramount (can't miss seizures)")
print("    * Already have good seizure detector downstream")
print("    * Conservative clinical environment")
print("  Pipeline: EEG → Binary Filter → Seizure Detector → Alert")

print("\n[OPTION B: USE 3-CLASS MODEL (UNIFIED)]")
print("  When to use:")
print("    * {:.2f}% seizure detection is acceptable".format(threeclass_sz_recall))
print("    * Want simpler pipeline (one model)")
print("    * Research/pilot deployment")
print("  Pipeline: EEG → 3-Class Detector → Alert if seizure")

print("\n[OPTION C: HYBRID APPROACH (BEST OF BOTH)]")
print("  Architecture:")
print("    * Use binary filter first (ultra-safe)")
print("    * Use 3-class on filtered data (active detection)")
print("    * Get both safety AND functionality")
print("  Pipeline: EEG → Binary Filter → 3-Class Detector → Alert")

print("\n[OPTION D: A/B TESTING (RECOMMENDED)]")
print("  Strategy:")
print("    * Deploy both models in parallel")
print("    * Compare real-world performance")
print("    * Choose based on clinical outcomes")
print("    * Allows evidence-based decision")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("GENERATING COMPARISON VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Seizure Handling
ax1 = axes[0, 0]
models = ['Binary\n(Preservation)', '3-Class\n(Detection)']
seizure_scores = [binary_sz_pres, threeclass_sz_recall]
colors = ['#2ecc71' if s > 95 else '#e74c3c' for s in seizure_scores]
bars = ax1.bar(models, seizure_scores, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.set_title('Seizure Handling Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 105])
ax1.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
for i, (bar, score) in enumerate(zip(bars, seizure_scores)):
    ax1.text(bar.get_x() + bar.get_width()/2, score + 2, 
             f'{score:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Overall Accuracy
ax2 = axes[0, 1]
models = ['Binary', '3-Class']
accuracy_scores = [binary_overall, threeclass_overall]
colors = ['#3498db', '#9b59b6']
bars = ax2.bar(models, accuracy_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 105])
for i, (bar, score) in enumerate(zip(bars, accuracy_scores)):
    ax2.text(bar.get_x() + bar.get_width()/2, score + 2,
             f'{score:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Trade-off Matrix
ax3 = axes[1, 0]
metrics = ['Seizure\nHandling', 'Background\nCorrect', 'Overall\nAccuracy']
binary_vals = [binary_sz_pres, binary_bg_clean, binary_overall]
threeclass_vals = [threeclass_sz_recall, threeclass_bg_recall, threeclass_overall]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, binary_vals, width, label='Binary', color='#3498db', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x + width/2, threeclass_vals, width, label='3-Class', color='#9b59b6', alpha=0.7, edgecolor='black')

ax3.set_ylabel('Percentage (%)', fontsize=12)
ax3.set_title('Performance Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.set_ylim([0, 105])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Clinical Use Case
ax4 = axes[1, 1]
ax4.axis('off')

use_case_text = """
CLINICAL DEPLOYMENT GUIDE

Binary Model (Artifact Filter):
• Ultra-safe (99.88% preservation)
• Use when: Safety is critical
• Pipeline: EEG → Filter → Seizure Detector

3-Class Model (Multi-Task):
• Active detection ({:.1f}% recall)
• Use when: Unified system preferred
• Pipeline: EEG → Detector → Alert

Recommended: A/B test both in parallel
""".format(threeclass_sz_recall)

ax4.text(0.1, 0.5, use_case_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save
os.makedirs('./results', exist_ok=True)
save_path = './results/model_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Comparison visualization saved: {save_path}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[SUCCESS] COMPARISON COMPLETE")
print("="*70)

print(f"\n[KEY FINDINGS]:")
print(f"  * Binary model: {binary_sz_pres:.2f}% seizure preservation (ultra-safe)")
print(f"  * 3-Class model: {threeclass_sz_recall:.2f}% seizure detection (functional)")
print(f"  * Trade-off: Safety vs Functionality")
print(f"  * Both models valid for different use cases")

print(f"\n[OUTPUT FILES]:")
print(f"  * {save_path}")

print(f"\n[RECOMMENDATION]:")
print(f"  * Deploy both models for A/B testing")
print(f"  * Use binary for maximum safety")
print(f"  * Use 3-class for research/pilot")
print(f"  * Choose based on clinical requirements")

print("\n" + "="*70 + "\n")
