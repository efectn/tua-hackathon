#!/usr/bin/env python3
"""
Test a trained Storm LSTM model on a separate dataset.
Usage: python test_model.py
  - Loads model from ./storm_results3/model.pt
  - Loads test data from preprocessed_data_test.csv
  - Applies same feature engineering + labeling
  - Reports per-horizon accuracy, macro F1, per-class F1
  - Generates confusion matrices
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import os, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
    f1_score, accuracy_score)

# ── CONFIG ──
MODEL_PATH     = "./storm_results3/model.pt"
TEST_CSV       = "preprocessed_data_test.csv"
OUTPUT_DIR     = "./storm_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════
# 1. LOAD MODEL CHECKPOINT
# ══════════════════════════════════════════
print("="*70+"\n1. LOADING MODEL\n"+"="*70)

ckpt = torch.load(MODEL_PATH, weights_only=False)
feat_cols_saved = ckpt['features']
CLASS_NAMES     = ckpt['class_names']
NUM_CLASSES     = len(CLASS_NAMES)
HORIZONS_MAP    = ckpt['horizons']          # e.g. {'1h':60, '2h':120, ...}
HORIZON_NAMES   = list(HORIZONS_MAP.keys())
HORIZONS        = list(HORIZONS_MAP.values())
WINDOW          = ckpt['window']
new_feat_cols   = ckpt.get('new_feat_cols', [])
scaler_mean     = ckpt.get('scaler_mean')
scaler_scale    = ckpt.get('scaler_scale')

print(f"  Classes: {NUM_CLASSES}  {CLASS_NAMES}")
print(f"  Horizons: {HORIZONS_MAP}")
print(f"  Window: {WINDOW}")
print(f"  Features: {len(feat_cols_saved)}")

# Determine model architecture from checkpoint weights
model_state = ckpt['model']
# Count LSTM layers
lstm_layers = sum(1 for k in model_state if k.startswith('lstm.weight_ih_l'))
# Hidden size from first LSTM weight
hidden_size = model_state['lstm.weight_hh_l0'].shape[1]
# Check if intermediate FC exists
has_fc = 'fc.weight' in model_state
dropout = 0.5  # doesn't matter for eval

print(f"  Architecture: LSTM({lstm_layers} layer, hidden={hidden_size})"
      f"{' → FC' if has_fc else ''} → {len(HORIZON_NAMES)} heads × {NUM_CLASSES} classes")

# ══════════════════════════════════════════
# 2. REBUILD MODEL & LOAD WEIGHTS
# ══════════════════════════════════════════
print("\n"+"="*70+"\n2. REBUILDING MODEL\n"+"="*70)

class StormLSTM(nn.Module):
    def __init__(self, inp, hid, layers, drop, n_heads, n_classes, use_fc=False, fc_size=48):
        super().__init__()
        self.use_fc = use_fc
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hid)
        self.drop = nn.Dropout(drop)
        if use_fc:
            self.fc = nn.Linear(hid, fc_size)
            self.drop2 = nn.Dropout(drop)
            self.heads = nn.ModuleList([nn.Linear(fc_size, n_classes) for _ in range(n_heads)])
        else:
            self.heads = nn.ModuleList([nn.Linear(hid, n_classes) for _ in range(n_heads)])

    def forward(self, x):
        o, _ = self.lstm(x)
        h = self.drop(self.bn(o[:, -1, :]))
        if self.use_fc:
            h = self.drop2(torch.relu(self.fc(h)))
        return [head(h) for head in self.heads]

n_features = len(feat_cols_saved)
fc_size = model_state['fc.weight'].shape[0] if has_fc else 48

model = StormLSTM(n_features, hidden_size, lstm_layers, dropout,
                  len(HORIZON_NAMES), NUM_CLASSES, use_fc=has_fc, fc_size=fc_size)
model.load_state_dict(model_state)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Loaded model: {n_params:,} parameters")

# ══════════════════════════════════════════
# 3. LOAD & PREPARE TEST DATA
# ══════════════════════════════════════════
print("\n"+"="*70+"\n3. LOADING TEST DATA\n"+"="*70)

df = pd.read_csv(TEST_CSV)
print(f"  Loaded: {len(df)} rows")

time_col = [c for c in df.columns if c.lower() in ["time","timestamp","datetime"]][0]
df["Time"] = pd.to_datetime(df[time_col])
df.sort_values("Time", inplace=True)
df.set_index("Time", inplace=True)

for col in ["F","BZ_GSM","flow_speed","proton_density","T","E","SYM_H"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

fills = {"F":9999.99,"BZ_GSM":9999.99,"flow_speed":99999.9,
         "proton_density":999.99,"T":9999999.0,"E":999.99,"SYM_H":99999}
for col, fv in fills.items():
    n = (df[col]==fv).sum()
    df[col] = df[col].replace(fv, np.nan)
    if n>0: print(f"  {col}: {n} fill values → NaN")

df = df.ffill(limit=5)
df.dropna(inplace=True)
print(f"  Clean: {len(df)} rows ({len(df)/1440:.1f} days)")
print(f"  Range: {df.index[0]} → {df.index[-1]}")

# ══════════════════════════════════════════
# 4. FEATURE ENGINEERING (same as training)
# ══════════════════════════════════════════
print("\n"+"="*70+"\n4. FEATURE ENGINEERING\n"+"="*70)

raw_feats = ["F", "BZ_GSM", "flow_speed", "proton_density", "T"]

for c in raw_feats:
    df[f"{c}_m30"]   = df[c].rolling(30, min_periods=1).mean()
    df[f"{c}_s30"]   = df[c].rolling(30, min_periods=1).std().fillna(0)
    df[f"{c}_min30"] = df[c].rolling(30, min_periods=1).min()
    df[f"{c}_max30"] = df[c].rolling(30, min_periods=1).max()

df["Bz_south"]     = (df["BZ_GSM"] < 0).astype(float)
df["Bz_south_dur"] = df["Bz_south"].rolling(60, min_periods=1).sum()
df["dyn_pressure"]  = df["proton_density"] * df["flow_speed"]**2 * 1.6726e-6

df.dropna(inplace=True)

# Use same feature columns as training
feat_cols = [c for c in feat_cols_saved if c in df.columns]
missing = [c for c in feat_cols_saved if c not in df.columns]
if missing:
    print(f"  ⚠ Missing features: {missing}")
    exit(1)
print(f"  Features: {len(feat_cols)}")

# ══════════════════════════════════════════
# 5. LABELS
# ══════════════════════════════════════════
print("\n"+"="*70+"\n5. LABELS\n"+"="*70)

def sym_to_class(sym_val):
    """Convert SYM-H to class — must match training."""
    if NUM_CLASSES == 2:
        if sym_val > -50:   return 0
        else:               return 1
    elif NUM_CLASSES == 3:
        if sym_val > -50:   return 0
        elif sym_val > -100: return 1
        else:                return 2
    else:  # 4-class
        if sym_val > -50:   return 0
        elif sym_val > -100: return 1
        elif sym_val > -200: return 2
        else:                return 3

sym = df["SYM_H"].values
n = len(df)

for hname, hmins in zip(HORIZON_NAMES, HORIZONS):
    col = f"storm_{hname}"
    arr = np.full(n, np.nan)
    for i in range(n - hmins):
        worst_sym = sym[i+1:i+1+hmins].min()
        arr[i] = sym_to_class(worst_sym)
    df[col] = arr

label_cols = [f"storm_{h}" for h in HORIZON_NAMES]
df.dropna(subset=label_cols, inplace=True)
for col in label_cols:
    df[col] = df[col].astype(int)

df["current_class"] = df["SYM_H"].apply(sym_to_class)

print(f"  Usable rows: {len(df)}")
print(f"\n  Class distribution per horizon:")
header = "  " + f"{'':>6s}" + "".join(f"  {cn:>8s}" for cn in CLASS_NAMES)
print(header)
for h in HORIZON_NAMES:
    counts = df[f"storm_{h}"].value_counts().sort_index()
    vals = [counts.get(i, 0) for i in range(NUM_CLASSES)]
    pcts = [f"{v/len(df)*100:.1f}%" for v in vals]
    print(f"  {h:>6s}" + "".join(f"  {p:>8s}" for p in pcts))

# ══════════════════════════════════════════
# 6. BUILD SEQUENCES & SCALE
# ══════════════════════════════════════════
print("\n"+"="*70+"\n6. SEQUENCES\n"+"="*70)

# Scale new features using saved scaler stats
original_csv_cols = set(pd.read_csv(TEST_CSV, nrows=0).columns) - {"Time", "SYM_H"}
new_fc_test = [c for c in feat_cols if c not in original_csv_cols]

if new_fc_test and scaler_mean is not None and scaler_scale is not None:
    # Apply the TRAINING scaler to test data (no fit, just transform)
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    scaler.var_ = scaler_scale ** 2
    scaler.n_features_in_ = len(scaler_mean)
    df[new_fc_test] = scaler.transform(df[new_fc_test].values)
    print(f"  Scaled {len(new_fc_test)} new features using training statistics")

feat_values = df[feat_cols].values.astype(np.float32)
label_data = {h: df[f"storm_{h}"].values for h in HORIZON_NAMES}
current_cls = df["current_class"].values

X_test = np.array([feat_values[i-WINDOW:i] for i in range(WINDOW, len(feat_values))],
                  dtype=np.float32)
y_test = {h: label_data[h][WINDOW:] for h in HORIZON_NAMES}
cc_test = current_cls[WINDOW:]

print(f"  Test sequences: {len(X_test)}, shape: {X_test.shape}")

# DataLoader
from torch.utils.data import DataLoader, TensorDataset
BATCH = 256
tensors = [torch.FloatTensor(X_test)] + [torch.LongTensor(y_test[h]) for h in HORIZON_NAMES]
te_dl = DataLoader(TensorDataset(*tensors), batch_size=BATCH)

# ══════════════════════════════════════════
# 7. INFERENCE
# ══════════════════════════════════════════
print("\n"+"="*70+"\n7. INFERENCE\n"+"="*70)

model.eval()
preds = {h: [] for h in HORIZON_NAMES}
probs = {h: [] for h in HORIZON_NAMES}
trues = {h: [] for h in HORIZON_NAMES}

n_heads = len(HORIZON_NAMES)
with torch.no_grad():
    for batch in te_dl:
        xb = batch[0]
        targets = [batch[i+1] for i in range(n_heads)]
        logits = model(xb)
        for i, h in enumerate(HORIZON_NAMES):
            p = torch.softmax(logits[i], dim=1)
            preds[h].extend(logits[i].argmax(1).numpy())
            probs[h].extend(p.numpy())
            trues[h].extend(targets[i].numpy())

for h in HORIZON_NAMES:
    preds[h] = np.array(preds[h])
    probs[h] = np.array(probs[h])
    trues[h] = np.array(trues[h])

# Persistence baseline
persist_pred = cc_test[:len(trues[HORIZON_NAMES[0]])]

# ══════════════════════════════════════════
# 8. RESULTS
# ══════════════════════════════════════════
print("\n"+"="*70+"\n8. TEST RESULTS\n"+"="*70)

class_labels = list(range(NUM_CLASSES))

print(f"\n{'─'*95}")
print(f"  {'Horizon':<8s} │ {'Model':<12s} │ {'Accuracy':>9s} │ {'Macro F1':>9s} │ "
      f"{'Wgt F1':>9s} │ {'per-class F1':>30s}")
print(f"{'─'*95}")

results = {}
for h in HORIZON_NAMES:
    yt = trues[h]; yp = preds[h]

    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro', zero_division=0)
    wf1 = f1_score(yt, yp, average='weighted', zero_division=0)
    cf1 = f1_score(yt, yp, average=None, labels=class_labels, zero_division=0)
    cf1_str = " ".join([f"c{i}:{v:.3f}" for i, v in enumerate(cf1)])

    # Persistence
    pp = persist_pred[:len(yt)]
    p_acc = accuracy_score(yt, pp)
    p_mf1 = f1_score(yt, pp, average='macro', zero_division=0)
    p_wf1 = f1_score(yt, pp, average='weighted', zero_division=0)
    p_cf1 = f1_score(yt, pp, average=None, labels=class_labels, zero_division=0)
    p_cf1_str = " ".join([f"c{i}:{v:.3f}" for i, v in enumerate(p_cf1)])

    results[h] = {
        'acc': acc, 'macro_f1': mf1, 'weighted_f1': wf1, 'per_class_f1': cf1,
        'pred': yp, 'true': yt, 'prob': probs[h],
        'persist_acc': p_acc, 'persist_mf1': p_mf1,
    }

    print(f"  {h:<8s} │ {'LSTM':<12s} │ {acc:>9.4f} │ {mf1:>9.4f} │ {wf1:>9.4f} │ {cf1_str}")
    print(f"  {'':8s} │ {'Persistence':<12s} │ {p_acc:>9.4f} │ {p_mf1:>9.4f} │ {p_wf1:>9.4f} │ {p_cf1_str}")
    print(f"{'─'*95}")

# Classification reports
for h in HORIZON_NAMES:
    r = results[h]
    print(f"\n  ── {h.upper()} Classification Report ──")
    print(classification_report(r['true'], r['pred'],
          labels=class_labels, target_names=CLASS_NAMES, zero_division=0))

# ══════════════════════════════════════════
# 9. PLOTS
# ══════════════════════════════════════════
print("="*70+"\n9. PLOTS\n"+"="*70)
plt.style.use('seaborn-v0_8-whitegrid')
O = OUTPUT_DIR
clrs = ['#1976D2','#388E3C','#F57C00','#D32F2F']

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, h in zip(axes.flat, HORIZON_NAMES):
    r = results[h]
    cm = confusion_matrix(r['true'], r['pred'], labels=class_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm.astype(float), row_sums, where=row_sums!=0,
                       out=np.zeros_like(cm, dtype=float))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=30)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{h} Forecast (macF1={r["macro_f1"]:.3f})', fontsize=13, fontweight='bold')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            c = 'white' if cm_pct[i,j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n{cm_pct[i,j]:.0%}', ha='center', va='center',
                    fontsize=10, color=c, fontweight='bold')
plt.suptitle(f'Test Set — Confusion Matrices ({NUM_CLASSES}-Class)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{O}/test_confusion.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ test_confusion.png")

# LSTM vs Persistence
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
x = np.arange(len(HORIZON_NAMES)); w = 0.35

ax = axes[0]
lstm_acc = [results[h]['acc'] for h in HORIZON_NAMES]
pers_acc = [results[h]['persist_acc'] for h in HORIZON_NAMES]
b1 = ax.bar(x-w/2, lstm_acc, w, label='LSTM', color='#1976D2', alpha=.85)
b2 = ax.bar(x+w/2, pers_acc, w, label='Persistence', color='#F57C00', alpha=.85)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f'{b.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f'{b.get_height():.3f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(HORIZON_NAMES, fontsize=12)
ax.set_title('Accuracy: LSTM vs Persistence', fontweight='bold'); ax.legend(); ax.set_ylim(0, 1.15)

ax = axes[1]
lstm_mf1 = [results[h]['macro_f1'] for h in HORIZON_NAMES]
pers_mf1 = [results[h]['persist_mf1'] for h in HORIZON_NAMES]
b1 = ax.bar(x-w/2, lstm_mf1, w, label='LSTM', color='#1976D2', alpha=.85)
b2 = ax.bar(x+w/2, pers_mf1, w, label='Persistence', color='#F57C00', alpha=.85)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f'{b.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f'{b.get_height():.3f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(HORIZON_NAMES, fontsize=12)
ax.set_title('Macro F1: LSTM vs Persistence', fontweight='bold'); ax.legend(); ax.set_ylim(0, 1.15)

plt.tight_layout()
plt.savefig(f"{O}/test_vs_persistence.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ test_vs_persistence.png")

# Timeline
fig, axes = plt.subplots(len(HORIZON_NAMES)+1, 1,
                         figsize=(18, 3*(len(HORIZON_NAMES)+1)), sharex=True)
sn = min(1500, len(trues['1h'])); xr = np.arange(sn)

ax = axes[0]
ax.plot(xr, trues['1h'][:sn], '-', color='#D32F2F', lw=0.8, alpha=0.7)
ax.fill_between(xr, 0, trues['1h'][:sn], alpha=0.3, color='#D32F2F')
ax.set_ylabel('Actual\nClass')
ax.set_yticks(list(range(NUM_CLASSES))); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_title(f'Test Set — {NUM_CLASSES}-Class Forecast Timeline', fontsize=14, fontweight='bold')

short_names = [n[0] for n in CLASS_NAMES]
for i, h in enumerate(HORIZON_NAMES):
    ax = axes[i+1]
    ax.plot(xr, preds[h][:sn], '-', color=clrs[i], lw=0.8, alpha=0.7, label=f'Predicted {h}')
    ax.plot(xr, trues[h][:sn], '-', color='gray', lw=0.5, alpha=0.3, label='Actual')
    ax.set_ylabel(f'{h}')
    ax.set_yticks(list(range(NUM_CLASSES))); ax.set_yticklabels(short_names, fontsize=8)
    ax.legend(fontsize=8, loc='upper right')

axes[-1].set_xlabel('Minutes')
plt.tight_layout()
plt.savefig(f"{O}/test_timeline.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ test_timeline.png")

# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n"+"="*70+"\nSUMMARY\n"+"="*70)
print(f"""
  Test data: {TEST_CSV}
  {NUM_CLASSES}-Class: {' / '.join(CLASS_NAMES)}

┌──────────┬───────────┬───────────┬───────────┐
│ Horizon  │  Accuracy │  Macro F1 │ vs Persist│
├──────────┼───────────┼───────────┼───────────┤""")
for h in HORIZON_NAMES:
    r = results[h]
    diff = r['macro_f1'] - r['persist_mf1']
    print(f"│ {h:>6s}   │   {r['acc']:.4f}  │   {r['macro_f1']:.4f}  │  {diff:>+.4f}  │")
print(f"""└──────────┴───────────┴───────────┴───────────┘

  Plots → {O}/
""")
