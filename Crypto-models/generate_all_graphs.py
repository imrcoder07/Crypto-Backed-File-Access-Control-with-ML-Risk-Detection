"""
generate_all_graphs.py
----------------------
Standalone script: loads the 3 trained pipelines + processed_data.csv
and generates all 10 ML evaluation graphs directly to visualizations/.

Run from: crypto_access_control_app/
    python Crypto-models/generate_all_graphs.py
"""
import sys, os, warnings, pickle
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — no window needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
VIS_DIR       = os.path.join(BASE_DIR, 'visualizations')
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
os.makedirs(VIS_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150})

print("=" * 60)
print("  Crypto Access Control — Graph Generation")
print("=" * 60)

# ── Load models ──────────────────────────────────────────────
print("\n[1/3] Loading trained pipelines...")
with open(os.path.join(MODELS_DIR, 'random_forest_pipeline.pkl'), 'rb') as f:
    rf_pipeline = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'svm_pipeline.pkl'), 'rb') as f:
    svm_pipeline = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'isolation_forest_pipeline.pkl'), 'rb') as f:
    iso_pipeline = pickle.load(f)
print("    Random Forest   : loaded")
print("    Linear SVM      : loaded")
print("    Isolation Forest: loaded")

# ── Build test set ───────────────────────────────────────────
print("\n[2/3] Building test set from processed_data.csv...")
df = pd.read_csv(PROCESSED_CSV)
print(f"    Rows loaded: {len(df):,}")
if len(df) > 200_000:
    df = df.sample(n=200_000, random_state=42)
    print(f"    Sampled to: 200,000 rows")

target = 'risky'
X = df.drop(target, axis=1)
y = df[target]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Test set: {len(X_test):,} samples  "
      f"({int(y_test.sum()):,} risky / {int((y_test==0).sum()):,} normal)")

# ── Predictions + scores ─────────────────────────────────────
print("\n[3/3] Running predictions...")
y_pred_rf  = rf_pipeline.predict(X_test)
y_pred_svm = svm_pipeline.predict(X_test)
raw_iso    = iso_pipeline.predict(X_test)
y_pred_iso = np.array([1 if p == -1 else 0 for p in raw_iso])

y_score_rf  = rf_pipeline.predict_proba(X_test)[:, 1]
y_score_svm = svm_pipeline.decision_function(X_test)
y_score_iso = -iso_pipeline.named_steps['classifier'].score_samples(
    iso_pipeline.named_steps['preprocessor'].transform(X_test)
)

print(f"    RF  Accuracy : {accuracy_score(y_test, y_pred_rf)*100:.2f}%")
print(f"    SVM Accuracy : {accuracy_score(y_test, y_pred_svm)*100:.2f}%")
print(f"    ISO Accuracy : {accuracy_score(y_test, y_pred_iso)*100:.2f}%")

# ═════════════════════════════════════════════════════════════
# GRAPH 4 — Class Distribution
# ═════════════════════════════════════════════════════════════
print("\n[Graph 4/13] Class Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts  = pd.Series(y_test).value_counts().sort_index()
labels  = ['Normal (0)', 'Risky (1)']
colors  = ['#2196F3', '#F44336']
pct     = (counts / len(y_test) * 100).round(1)

bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
axes[0].set_title('Class Distribution — Absolute Count', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Samples', fontsize=11)
axes[0].set_xlabel('Class', fontsize=11)
for bar, cnt, p in zip(bars, counts.values, pct.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{cnt:,}\n({p}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylim(0, max(counts.values) * 1.15)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 12}
)
for at in autotexts: at.set_fontweight('bold')
axes[1].set_title('Class Distribution — Proportion', fontsize=14, fontweight='bold')

plt.suptitle('Dataset Class Imbalance Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
out = os.path.join(VIS_DIR, '4_class_distribution.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")

# ═════════════════════════════════════════════════════════════
# GRAPH 5 — Confusion Matrices
# ═════════════════════════════════════════════════════════════
print("[Graph 5/13] Confusion Matrices...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
model_data = [
    ('Random Forest',    y_pred_rf,  '#1565C0'),
    ('Linear SVM',       y_pred_svm, '#6A1B9A'),
    ('Isolation Forest', y_pred_iso, '#B71C1C'),
]
for ax, (name, y_pred, _) in zip(axes, model_data):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Risky'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    acc = accuracy_score(y_test, y_pred) * 100
    ax.set_title(f'{name}\nAccuracy: {acc:.2f}%', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

plt.suptitle('Confusion Matrices — All Models (Test Set)', fontsize=15, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '5_confusion_matrices.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")

# Print classification reports
for name, y_pred, _ in model_data:
    print(f"\n    --- {name} ---")
    print(classification_report(y_test, y_pred, target_names=['Normal','Risky'],
                                 zero_division=0))

# ═════════════════════════════════════════════════════════════
# GRAPH 6 — Model Comparison
# ═════════════════════════════════════════════════════════════
print("[Graph 6/13] Model Comparison...")
models  = ['Random Forest', 'Linear SVM', 'Isolation Forest']
y_preds = [y_pred_rf, y_pred_svm, y_pred_iso]
metrics_data = {}
for name, y_pred in zip(models, y_preds):
    metrics_data[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred) * 100,
        'Precision': precision_score(y_test, y_pred, zero_division=0) * 100,
        'Recall':    recall_score(y_test, y_pred, zero_division=0) * 100,
        'F1 Score':  f1_score(y_test, y_pred, zero_division=0) * 100,
    }
df_metrics  = pd.DataFrame(metrics_data).T
metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors_bar  = ['#1976D2', '#7B1FA2', '#D32F2F', '#388E3C']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x, width = np.arange(len(models)), 0.2
for i, (metric, color) in enumerate(zip(metric_cols, colors_bar)):
    bars = axes[0].bar(x + i*width, df_metrics[metric], width,
                       label=metric, color=color, alpha=0.88, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x()+bar.get_width()/2, h+0.3,
                     f'{h:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
axes[0].set_xticks(x + width*1.5)
axes[0].set_xticklabels(models, fontsize=11)
axes[0].set_ylim(0, 115)
axes[0].set_ylabel('Score (%)', fontsize=12)
axes[0].set_title('Model Performance Comparison\n(All Metrics)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

table_vals = [[f"{df_metrics.loc[m, c]:.2f}%" for c in metric_cols] for m in models]
table = axes[1].table(cellText=table_vals, rowLabels=models, colLabels=metric_cols,
                      cellLoc='center', loc='center',
                      rowColours=['#E3F2FD','#F3E5F5','#FFEBEE'],
                      colColours=['#FFFDE7']*4)
table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.4, 2.0)
axes[1].axis('off')
axes[1].set_title('Performance Summary Table', fontsize=13, fontweight='bold', pad=20)

plt.suptitle('ML Model Comparison — Crypto Access Control', fontsize=15, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '6_model_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")
print(f"\n    Results:\n{df_metrics.round(2).to_string()}")

# ═════════════════════════════════════════════════════════════
# GRAPH 7 — Feature Importance
# ═════════════════════════════════════════════════════════════
print("\n[Graph 7/13] Feature Importance (Random Forest)...")
rf_clf       = rf_pipeline.named_steps['classifier']
preprocessor = rf_pipeline.named_steps['preprocessor']
num_features = list(preprocessor.transformers_[0][2])
cat_encoder  = preprocessor.transformers_[1][1]
cat_features = list(cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
all_features = num_features + cat_features
importances  = rf_clf.feature_importances_
std          = np.std([t.feature_importances_ for t in rf_clf.estimators_], axis=0)
sorted_idx   = np.argsort(importances)[::-1]
top_n        = min(20, len(all_features))
top_idx      = sorted_idx[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors_imp = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, top_n))
axes[0].barh(range(top_n), importances[top_idx][::-1],
             xerr=std[top_idx][::-1],
             color=colors_imp[::-1], edgecolor='white', height=0.7)
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels([all_features[i] for i in top_idx[::-1]], fontsize=9)
axes[0].set_xlabel('Feature Importance (Mean Decrease in Impurity)', fontsize=11)
axes[0].set_title(f'Top {top_n} Feature Importances\n(Random Forest)', fontsize=13, fontweight='bold')
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

cumulative = np.cumsum(importances[sorted_idx])
axes[1].plot(range(1, len(all_features)+1), cumulative*100,
             color='#1565C0', linewidth=2.5, marker='o', markersize=4)
axes[1].axhline(y=90, color='red', linestyle='--', linewidth=1.5, label='90% threshold')
axes[1].axhline(y=95, color='orange', linestyle='--', linewidth=1.5, label='95% threshold')
n_90 = np.searchsorted(cumulative, 0.90) + 1
n_95 = np.searchsorted(cumulative, 0.95) + 1
axes[1].fill_between(range(1, len(all_features)+1), cumulative*100, alpha=0.1, color='#1565C0')
axes[1].set_xlabel('Number of Features (ranked by importance)', fontsize=11)
axes[1].set_ylabel('Cumulative Importance (%)', fontsize=11)
axes[1].set_title('Cumulative Feature Importance', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].text(n_90+0.3, 55, f'n={n_90}', color='red', fontsize=10, fontweight='bold')
axes[1].text(n_95+0.3, 42, f'n={n_95}', color='orange', fontsize=10, fontweight='bold')
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.suptitle('Random Forest — Feature Importance Analysis', fontsize=15, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '7_feature_importance.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")
print("    Top 5 features:")
for i in sorted_idx[:5]:
    print(f"      {all_features[i]:<35} {importances[i]:.4f}")

# ═════════════════════════════════════════════════════════════
# GRAPH 8 — ROC Curves
# ═════════════════════════════════════════════════════════════
print("\n[Graph 8/13] ROC Curves + AUC...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
scores_map = [
    ('Random Forest',    y_score_rf,  '#1565C0', '-'),
    ('Linear SVM',       y_score_svm, '#6A1B9A', '--'),
    ('Isolation Forest', y_score_iso, '#B71C1C', '-.'),
]
for name, scores, color, ls in scores_map:
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc     = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, lw=2.5, linestyle=ls,
                 label=f'{name} (AUC = {roc_auc:.4f})')

axes[0].plot([0,1],[0,1],'k--',lw=1.5,label='Random (AUC=0.50)')
axes[0].set_xlim([-0.01,1.01]); axes[0].set_ylim([-0.01,1.05])
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
opt_idx = np.argmax(tpr_rf - fpr_rf)
axes[0].scatter(fpr_rf[opt_idx], tpr_rf[opt_idx], marker='*', color='#1565C0', s=200, zorder=5)

ax_ins = axes[0].inset_axes([0.45, 0.05, 0.50, 0.50])
for name, scores, color, ls in scores_map:
    fpr, tpr, _ = roc_curve(y_test, scores)
    ax_ins.plot(fpr, tpr, color=color, lw=2, linestyle=ls)
ax_ins.set_xlim(0, 0.15); ax_ins.set_ylim(0.85, 1.01)
ax_ins.set_title('Zoom: Low FPR', fontsize=8)
ax_ins.tick_params(labelsize=7)
axes[0].indicate_inset_zoom(ax_ins, edgecolor='gray')

auc_vals, auc_labels, auc_colors = [], [], []
for name, scores, color, _ in scores_map:
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_vals.append(auc(fpr, tpr))
    auc_labels.append(name); auc_colors.append(color)
bars = axes[1].barh(auc_labels, auc_vals, color=auc_colors, edgecolor='white', height=0.5)
axes[1].set_xlim(0, 1.12)
axes[1].set_xlabel('AUC Score', fontsize=12)
axes[1].set_title('AUC Comparison', fontsize=13, fontweight='bold')
for bar, val in zip(bars, auc_vals):
    axes[1].text(val+0.005, bar.get_y()+bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=13, fontweight='bold')
axes[1].axvline(x=0.5, color='red', linestyle='--', lw=1.5, label='Random baseline')
axes[1].axvline(x=1.0, color='green', linestyle=':', lw=1.5, label='Perfect')
axes[1].legend(fontsize=10)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.suptitle('ROC Curve Analysis — Crypto Access Control', fontsize=15, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '8_roc_curves.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")
for name, scores, _, _ in scores_map:
    fpr, tpr, _ = roc_curve(y_test, scores)
    print(f"      {name:<20} AUC = {auc(fpr,tpr):.4f}")

# ═════════════════════════════════════════════════════════════
# GRAPH 9 — Precision-Recall Curves
# ═════════════════════════════════════════════════════════════
print("\n[Graph 9/13] Precision-Recall Curves...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
scores_map2 = [
    ('Random Forest',    y_score_rf,  y_pred_rf,  '#1565C0', '-'),
    ('Linear SVM',       y_score_svm, y_pred_svm, '#6A1B9A', '--'),
    ('Isolation Forest', y_score_iso, y_pred_iso, '#B71C1C', '-.'),
]
baseline_p = y_test.mean()
for name, scores, _, color, ls in scores_map2:
    precision, recall, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)
    axes[0].plot(recall, precision, color=color, lw=2.5, linestyle=ls,
                 label=f'{name} (AP={ap:.4f})')
axes[0].axhline(y=baseline_p, color='gray', linestyle=':', lw=1.5,
                label=f'Baseline={baseline_p:.3f}')
axes[0].set_xlim([-0.01,1.01]); axes[0].set_ylim([-0.01,1.05])
axes[0].set_xlabel('Recall', fontsize=12); axes[0].set_ylabel('Precision', fontsize=12)
axes[0].set_title('Precision-Recall Curves\n(Risky Class)', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper right', fontsize=10)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

ap_vals   = [average_precision_score(y_test, s) for _, s, _, _, _ in scores_map2]
ap_labels = [n for n, _, _, _, _ in scores_map2]
ap_colors = [c for _, _, _, c, _ in scores_map2]
bars = axes[1].barh(ap_labels, ap_vals, color=ap_colors, edgecolor='white', height=0.5)
axes[1].set_xlim(0, 1.12)
axes[1].set_xlabel('Average Precision (AP)', fontsize=12)
axes[1].set_title('AP Score Comparison\n(Higher = better on imbalanced class)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, ap_vals):
    axes[1].text(val+0.005, bar.get_y()+bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=13, fontweight='bold')
axes[1].axvline(x=baseline_p, color='red', linestyle='--', lw=1.5, label=f'Baseline={baseline_p:.3f}')
axes[1].legend(fontsize=10)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.suptitle('Precision-Recall Analysis — Imbalanced Dataset', fontsize=15, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '9_precision_recall_curves.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")

# ═════════════════════════════════════════════════════════════
# GRAPH 10 — Isolation Forest Anomaly Scores
# ═════════════════════════════════════════════════════════════
print("\n[Graph 10/13] Isolation Forest Anomaly Scores...")
from scipy.stats import gaussian_kde
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
scores_normal = y_score_iso[y_test == 0]
scores_risky  = y_score_iso[y_test == 1]
threshold     = np.percentile(y_score_iso, (1 - y_test.mean()) * 100)

axes[0].hist(scores_normal, bins=60, density=True, alpha=0.45, color='#1976D2',
             label='Normal (class 0)', edgecolor='none')
axes[0].hist(scores_risky,  bins=60, density=True, alpha=0.45, color='#D32F2F',
             label='Risky (class 1)', edgecolor='none')
x_range    = np.linspace(y_score_iso.min(), y_score_iso.max(), 300)
kde_normal = gaussian_kde(scores_normal)
kde_risky  = gaussian_kde(scores_risky)
axes[0].plot(x_range, kde_normal(x_range), color='#1565C0', lw=2.5)
axes[0].plot(x_range, kde_risky(x_range),  color='#B71C1C', lw=2.5)
axes[0].axvline(x=threshold, color='black', linestyle='--', lw=2,
                label=f'Threshold={threshold:.3f}')
axes[0].set_xlabel('Anomaly Score', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Anomaly Score Distribution\n(Isolation Forest)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

bp = axes[1].boxplot([scores_normal, scores_risky], patch_artist=True, notch=True,
                     medianprops={'color':'black','linewidth':2.5})
for patch, color in zip(bp['boxes'], ['#BBDEFB','#FFCDD2']):
    patch.set_facecolor(color)
axes[1].set_xticklabels(['Normal (0)', 'Risky (1)'], fontsize=12)
axes[1].set_ylabel('Anomaly Score', fontsize=12)
axes[1].set_title('Anomaly Score — Box Plot', fontsize=13, fontweight='bold')
axes[1].axhline(y=threshold, color='black', linestyle='--', lw=2, label=f'Threshold={threshold:.3f}')
axes[1].legend(fontsize=10)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.suptitle('Isolation Forest — Anomaly Score Analysis', fontsize=15, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '10_isolation_forest_anomaly_scores.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")
print(f"      Normal score mean: {scores_normal.mean():.4f}")
print(f"      Risky  score mean: {scores_risky.mean():.4f}")

# ═════════════════════════════════════════════════════════════
# GRAPH 11 — Per-Class Metrics
# ═════════════════════════════════════════════════════════════
print("\n[Graph 11/13] Per-Class Metrics (P/R/F1)...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_info   = [('Random Forest',y_pred_rf,'#1565C0'),
                ('Linear SVM',y_pred_svm,'#6A1B9A'),
                ('Isolation Forest',y_pred_iso,'#B71C1C')]
metric_names = ['Precision','Recall','F1-Score']
class_names  = ['Normal (0)','Risky (1)']
class_colors = ['#1E88E5','#E53935']

for ax, (model_name, y_pred, _) in zip(axes, model_info):
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    metric_vals = [prec, rec, f1]
    x = np.arange(len(metric_names))
    for ci, (cls_name, cls_color) in enumerate(zip(class_names, class_colors)):
        vals = [metric_vals[mi][ci]*100 for mi in range(3)]
        bars = ax.bar(x + ci*0.3, vals, 0.3, label=cls_name,
                      color=cls_color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x + 0.15)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title(model_name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.suptitle('Per-Class Performance Metrics — All Models\n(Risky class detection focus)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '11_per_class_metrics.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")

# ═════════════════════════════════════════════════════════════
# GRAPH 12 — Hyperparameter Search Visualization
# ═════════════════════════════════════════════════════════════
print("\n[Graph 12/13] Hyperparameter Search Results...")
cat_features_hp = ['activity','role']
num_features_hp = ['hour_of_day','day_of_week','is_weekend','avg_actions_per_day']
preprocessor_hp = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features_hp),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features_hp)
])
rf_hp = Pipeline(steps=[
    ('preprocessor', preprocessor_hp),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
])
param_dist_hp = {
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__max_depth':    [5, 10, 20, 30, None],
}
search_hp = RandomizedSearchCV(rf_hp, param_distributions=param_dist_hp,
                                n_iter=8, cv=3, random_state=42, n_jobs=-1,
                                return_train_score=True, scoring='f1')
search_hp.fit(X_test, y_test)
cv_results = pd.DataFrame(search_hp.cv_results_)
print(f"    Best params : {search_hp.best_params_}")
print(f"    Best F1     : {search_hp.best_score_:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
scatter_df = cv_results[['param_classifier__n_estimators',
                          'param_classifier__max_depth',
                          'mean_test_score','std_test_score']].copy()
scatter_df.columns = ['n_estimators','max_depth','mean_f1','std_f1']
depths_unique = scatter_df['max_depth'].unique()
palette = plt.cm.Blues(np.linspace(0.4, 0.9, len(depths_unique)))
for depth, color in zip(sorted(depths_unique, key=lambda x: x if x is not None else 999), palette):
    mask  = scatter_df['max_depth'] == depth
    label = f'max_depth={depth}'
    axes[0].errorbar(scatter_df.loc[mask,'n_estimators'],
                     scatter_df.loc[mask,'mean_f1'],
                     yerr=scatter_df.loc[mask,'std_f1'],
                     fmt='o-', color=color, label=label, markersize=8, lw=1.5, capsize=4)
axes[0].set_xlabel('n_estimators', fontsize=12)
axes[0].set_ylabel('Mean CV F1 Score', fontsize=12)
axes[0].set_title('Hyperparameter Search\n(n_estimators vs F1)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

cv_sorted  = cv_results.sort_values('mean_test_score', ascending=True)
bar_colors = ['#1565C0' if i == len(cv_sorted)-1 else '#90CAF9' for i in range(len(cv_sorted))]
axes[1].barh(range(len(cv_sorted)), cv_sorted['mean_test_score'],
             xerr=cv_sorted['std_test_score'],
             color=bar_colors, edgecolor='white', height=0.6)
axes[1].set_yticks(range(len(cv_sorted)))
axes[1].set_yticklabels(
    [f"n={r['param_classifier__n_estimators']}, d={r['param_classifier__max_depth']}"
     for _, r in cv_sorted.iterrows()], fontsize=9)
axes[1].set_xlabel('Mean CV F1 Score', fontsize=12)
axes[1].set_title('All Combinations Ranked by F1', fontsize=13, fontweight='bold')
axes[1].text(cv_sorted['mean_test_score'].max()+0.002,
             len(cv_sorted)-1, '<- Best', color='#1565C0', fontweight='bold', fontsize=10)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.suptitle('Random Forest — Hyperparameter Tuning Results', fontsize=15, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '12_hyperparameter_search.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")

# ═════════════════════════════════════════════════════════════
# GRAPH 13 — Correlation Heatmap
# ═════════════════════════════════════════════════════════════
print("\n[Graph 13/13] Correlation Heatmap...")
num_cols = ['hour_of_day','day_of_week','is_weekend','avg_actions_per_day']
X_num    = X_test[num_cols].copy()
X_num['risky (target)'] = y_test.values
corr = X_num.corr(method='pearson')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=axes[0],
            annot_kws={'size':11,'weight':'bold'}, cbar_kws={'shrink':0.8})
axes[0].set_title('Pearson Correlation Heatmap\n(Features + Target)', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=10)
axes[0].tick_params(axis='y', rotation=0, labelsize=10)

target_corr = corr['risky (target)'].drop('risky (target)').sort_values()
colors_corr = ['#D32F2F' if v < 0 else '#1565C0' for v in target_corr.values]
axes[1].barh(target_corr.index, target_corr.values, color=colors_corr, edgecolor='white', height=0.6)
for i, val in enumerate(target_corr.values):
    axes[1].text(val + (0.01 if val >= 0 else -0.01), i,
                 f'{val:.3f}', va='center',
                 ha='left' if val >= 0 else 'right',
                 fontsize=11, fontweight='bold')
axes[1].axvline(x=0, color='black', linewidth=1.2)
axes[1].set_xlabel('Correlation with Target (risky)', fontsize=12)
axes[1].set_title('Feature Correlation with Target\n(Predictive power ranking)', fontsize=13, fontweight='bold')
axes[1].set_xlim(-1.1, 1.1)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.suptitle('Feature Correlation Analysis', fontsize=15, fontweight='bold')
plt.tight_layout()
out = os.path.join(VIS_DIR, '13_correlation_heatmap.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.basename(out)}")

# ═════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════
import glob
vis_files = sorted(glob.glob(os.path.join(VIS_DIR, '*.png')))
print("\n" + "=" * 60)
print(f"  DONE — {len(vis_files)} graphs in {VIS_DIR}")
print("=" * 60)
for f in vis_files:
    size = os.path.getsize(f) / 1024
    print(f"  {os.path.basename(f):<50} {size:>6.1f} KB")

print("\nDocumentation Readiness:")
print("  README.md  : READY  (use graphs 3, 6, 7, 8)")
print("  Thesis     : READY  (all 13 graphs)")
print("  PPT/Slides : READY  (use graphs 3, 4, 5, 6, 7, 8)")
print("\nAll evaluation graphs generated successfully!")
