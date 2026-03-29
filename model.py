#!/usr/bin/env python3
"""
Solar Storm Forecasting — 3-Class, 4-Horizon LSTM
===================================================
Classes (based on WORST SYM-H in forecast window):
  0 = Quiet        (SYM-H > -50)
  1 = Weak storm   (-100 < SYM-H ≤ -50)
  2 = Storm        (SYM-H ≤ -100)  [merged Moderate+Intense]

Horizons: 1h, 2h, 4h, 12h

Fixes applied:
  - E field REMOVED (it's a linear combo of BZ and V — leaks info)
  - Rolling features only on RAW solar wind, not derived quantities
  - 3 classes (Moderate+Intense merged — too few samples to split)
  - Simplified LSTM (1 layer, hidden=32) to reduce overfitting
  - Label smoothing (0.1) prevents overconfident logits
  - LR warmup (5 epochs) prevents early memorization
  - Gap buffer between train/val/test
  - Persistence baseline comparison
  - Macro-F1 (penalizes ignoring rare classes)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
    f1_score, accuracy_score)

# ── CONFIG ──
CSV_PATH       = "preprocessed_data.csv"
OUTPUT_DIR     = "./storm_results"
HORIZONS       = [60, 120, 240, 720]   # 1h, 2h, 4h, 12h
HORIZON_NAMES  = ["1h", "2h", "4h", "12h"]
NUM_CLASSES    = 3
CLASS_NAMES    = ["Quiet", "Weak", "Storm"]
CLASS_THRESHOLDS = [-50, -100]          # SYM-H boundaries

WINDOW         = 30
GAP            = 120       # gap between splits
BATCH          = 256
EPOCHS         = 100
PATIENCE       = 15
HIDDEN         = 32
LAYERS         = 1
DROPOUT        = 0.5
LR             = 0.0005
WEIGHT_DECAY   = 1e-3
LABEL_SMOOTH   = 0.1      # prevents overconfident predictions
WARMUP_EPOCHS  = 5         # low LR at start prevents early memorization

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)

# ══════════════════════════════════════════
# 1. LOAD CSV
# ══════════════════════════════════════════
print("="*70+"\n1. LOADING DATA\n"+"="*70)

df = pd.read_csv(CSV_PATH)
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
before = len(df); df.dropna(inplace=True)
print(f"  Clean: {len(df)} rows ({len(df)/1440:.1f} days)")
print(f"  Range: {df.index[0]} → {df.index[-1]}")

# ══════════════════════════════════════════
# 2. FEATURES (careful — no leaky shortcuts)
# ══════════════════════════════════════════
print("\n"+"="*70+"\n2. FEATURE ENGINEERING\n"+"="*70)

# Raw features — NOTE: E is EXCLUDED (it's V×B, a linear combo of BZ and speed)
# Including it gives the model a trivial shortcut that inflates accuracy
raw_feats = ["F", "BZ_GSM", "flow_speed", "proton_density", "T"]

print(f"  Base features: {raw_feats}")
print(f"  E field EXCLUDED (= V×B, leaks BZ+speed info as shortcut)")

# Rolling stats — only on raw solar wind parameters
for c in raw_feats:
    df[f"{c}_m30"]   = df[c].rolling(30, min_periods=1).mean()
    df[f"{c}_s30"]   = df[c].rolling(30, min_periods=1).std().fillna(0)
    df[f"{c}_min30"] = df[c].rolling(30, min_periods=1).min()
    df[f"{c}_max30"] = df[c].rolling(30, min_periods=1).max()

# Physics features (carefully chosen — not redundant with raw)
df["Bz_south"]     = (df["BZ_GSM"] < 0).astype(float)
df["Bz_south_dur"] = df["Bz_south"].rolling(60, min_periods=1).sum()
df["dyn_pressure"]  = df["proton_density"] * df["flow_speed"]**2 * 1.6726e-6

df.dropna(inplace=True)

feat_cols = [c for c in df.columns
             if c not in ["SYM_H","E","Bz_south"]  # exclude intermediates
             and not c.startswith("storm_")]
print(f"  Total: {len(feat_cols)} features")

# ══════════════════════════════════════════
# 3. MULTI-CLASS FORECAST LABELS
# ══════════════════════════════════════════
print("\n"+"="*70+"\n3. 3-CLASS FORECAST LABELS\n"+"="*70)

print(f"""
  For each minute t and horizon H:
    Look at ALL SYM-H values in [t+1, t+H]
    Find the MINIMUM (most negative = worst storm)
    Classify that minimum into 3 classes:

    Class 0 (Quiet):  min(SYM-H) > -50
    Class 1 (Weak):   -100 < min(SYM-H) ≤ -50
    Class 2 (Storm):  min(SYM-H) ≤ -100  [Moderate+Intense merged]

  This asks: "What's the WORST storm level in the next H minutes?"
""")

def sym_to_class(sym_val):
    """Convert SYM-H value to storm class (3-class)."""
    if sym_val > -50:   return 0
    elif sym_val > -100: return 1
    else:                return 2  # Moderate+Intense merged

sym = df["SYM_H"].values
n = len(df)

max_horizon = max(HORIZONS)
valid_mask = np.ones(n, dtype=bool)

for hname, hmins in zip(HORIZON_NAMES, HORIZONS):
    col = f"storm_{hname}"
    arr = np.full(n, np.nan)
    for i in range(n - hmins):
        # Worst (most negative) SYM-H in forecast window
        worst_sym = sym[i+1:i+1+hmins].min()
        arr[i] = sym_to_class(worst_sym)
    df[col] = arr

# Drop incomplete rows
label_cols = [f"storm_{h}" for h in HORIZON_NAMES]
df.dropna(subset=label_cols, inplace=True)
for col in label_cols:
    df[col] = df[col].astype(int)

# Also store current class for persistence baseline
df["current_class"] = df["SYM_H"].apply(sym_to_class)

print(f"  Usable rows: {len(df)}")
print(f"\n  Class distribution per horizon:")
print(f"  {'':>6s}  {'Quiet':>8s}  {'Weak':>8s}  {'Storm':>8s}")
for h in HORIZON_NAMES:
    counts = df[f"storm_{h}"].value_counts().sort_index()
    vals = [counts.get(i, 0) for i in range(NUM_CLASSES)]
    pcts = [f"{v/len(df)*100:.1f}%" for v in vals]
    print(f"  {h:>6s}  {'  '.join(f'{p:>8s}' for p in pcts)}")

# Check we have storm data
storm_count = sum((df[f"storm_{h}"] > 0).sum() for h in HORIZON_NAMES)
if storm_count == 0:
    print("\n  ⚠ No storm events! Need data with SYM-H < -50.")
    exit()

# ══════════════════════════════════════════
# 4. SEQUENCES + GAPPED SPLIT
# ══════════════════════════════════════════
print("\n"+"="*70+"\n4. SEQUENCES + GAPPED TEMPORAL SPLIT\n"+"="*70)

# NOTE: preprocessing.py already applied StandardScaler to feature columns.
# We only need to scale the NEW features we computed above (rolling stats, physics).
# Strategy: identify which columns are new (not in the original CSV) and scale only those.
original_csv_cols = set(pd.read_csv(CSV_PATH, nrows=0).columns) - {"Time", "SYM_H"}
new_feat_cols = [c for c in feat_cols if c not in original_csv_cols]
old_feat_cols = [c for c in feat_cols if c in original_csv_cols]

print(f"  Already-scaled features: {len(old_feat_cols)}")
print(f"  New features to scale:   {len(new_feat_cols)}")

scaler = StandardScaler()
if new_feat_cols:
    df[new_feat_cols] = scaler.fit_transform(df[new_feat_cols].values)

feat_values = df[feat_cols].values.astype(np.float32)

label_data = {h: df[f"storm_{h}"].values for h in HORIZON_NAMES}
current_cls = df["current_class"].values

# Build sequences
X_all = np.array([feat_values[i-WINDOW:i] for i in range(WINDOW, len(feat_values))],
                 dtype=np.float32)
y_all = {h: label_data[h][WINDOW:] for h in HORIZON_NAMES}
cc_all = current_cls[WINDOW:]

print(f"  Sequences: {len(X_all)}, shape: {X_all.shape}")

# Gapped temporal split
n_total = len(X_all)
n_train = int(0.6 * n_total)
n_val = int(0.2 * n_total)

tr_end = n_train
va_start = min(tr_end + GAP, n_total - n_val - GAP - 100)
va_end = va_start + n_val
te_start = min(va_end + GAP, n_total - 100)

# Safety check
if te_start >= n_total - 50:
    gap_actual = max(5, (n_total - n_train - 2*n_val) // 3)
    print(f"  ⚠ Reducing gap to {gap_actual} (data too short for {GAP})")
    va_start = tr_end + gap_actual
    va_end = va_start + n_val
    te_start = va_end + gap_actual

Xtr = X_all[:tr_end]
Xva = X_all[va_start:va_end]
Xte = X_all[te_start:]

ytr = {h: y_all[h][:tr_end] for h in HORIZON_NAMES}
yva = {h: y_all[h][va_start:va_end] for h in HORIZON_NAMES}
yte = {h: y_all[h][te_start:] for h in HORIZON_NAMES}
cc_te = cc_all[te_start:]

print(f"\n  Split (with gaps between):")
for nm, xs, yd in [("Train",Xtr,ytr),("Val",Xva,yva),("Test",Xte,yte)]:
    dist = " ".join([f"c{i}:{int((yd[HORIZON_NAMES[0]]==i).sum())}" for i in range(NUM_CLASSES)])
    print(f"    {nm:>5s}: {len(xs):>6d} │ 1h classes: {dist}")

# DataLoaders
def make_dl(X, ys, shuffle=False):
    tensors = [torch.FloatTensor(X)] + [torch.LongTensor(ys[h]) for h in HORIZON_NAMES]
    return DataLoader(TensorDataset(*tensors), batch_size=BATCH, shuffle=shuffle)

tr_dl = make_dl(Xtr, ytr, shuffle=True)
va_dl = make_dl(Xva, yva)
te_dl = make_dl(Xte, yte)

# ══════════════════════════════════════════
# 5. LSTM MODEL (4 horizons × 3 classes)
# ══════════════════════════════════════════
print("\n"+"="*70+"\n5. LSTM MODEL\n"+"="*70)

class StormLSTM(nn.Module):
    def __init__(self, inp, hid=HIDDEN, layers=LAYERS, drop=DROPOUT,
                 n_heads=4, n_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hid)
        self.drop = nn.Dropout(drop)
        # Simplified: LSTM → BN → dropout → heads (no intermediate FC)
        self.heads = nn.ModuleList([nn.Linear(hid, n_classes) for _ in range(n_heads)])

    def forward(self, x):
        o, _ = self.lstm(x)
        h = self.drop(self.bn(o[:, -1, :]))
        return [head(h) for head in self.heads]  # raw logits, no softmax

model = StormLSTM(X_all.shape[2])
n_params = sum(p.numel() for p in model.parameters())
print(f"  LSTM({LAYERS} layer, hidden={HIDDEN}) → 4 heads × {NUM_CLASSES} classes")
print(f"  Parameters: {n_params:,}")
print(f"  Output: raw logits → softmax → class probabilities")

# ══════════════════════════════════════════
# 6. TRAIN
# ══════════════════════════════════════════
print("\n"+"="*70+"\n6. TRAINING\n"+"="*70)

# Class weights (sqrt-inverse frequency, zero for unseen classes)
class_weights_list = []
for h in HORIZON_NAMES:
    raw_counts = np.bincount(ytr[h], minlength=NUM_CLASSES).astype(float)
    # Warn about missing classes in training set
    for ci in range(NUM_CLASSES):
        if raw_counts[ci] == 0:
            print(f"  ⚠ {h}: class {ci} ({CLASS_NAMES[ci]}) has 0 training samples → weight=0")
    # Classes with 0 samples get weight 0 (can't learn what you've never seen)
    # For present classes, use sqrt(1/count) to soften extreme ratios
    weights = np.zeros(NUM_CLASSES)
    present_mask = raw_counts > 0
    if present_mask.any():
        inv = 1.0 / raw_counts[present_mask]
        sqrt_inv = np.sqrt(inv)
        sqrt_inv = sqrt_inv / sqrt_inv.sum() * present_mask.sum()
        weights[present_mask] = sqrt_inv
    class_weights_list.append(torch.FloatTensor(weights))
    print(f"  {h} class weights: {dict(enumerate(weights.round(3).tolist()))}")

# Label smoothing CrossEntropy — this is KEY for preventing overconfidence
criterions = [nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH)
              for cw in class_weights_list]

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Warmup + cosine schedule
def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return 0.1 + 0.9 * (epoch / WARMUP_EPOCHS)  # ramp 10%→100%
    else:
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))  # cosine decay

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)

print(f"\n  Label smoothing: {LABEL_SMOOTH}")
print(f"  LR warmup: {WARMUP_EPOCHS} epochs")
print(f"  Schedule: warmup → cosine annealing")
print()

hist = {'tl':[], 'vl':[], 'ta':[], 'va':[], 'tf1':[], 'vf1':[]}
best_vl = float('inf'); best_st = None; wait = 0

for ep in range(EPOCHS):
    # ── Train ──
    model.train()
    losses = []; all_pred = []; all_true = []

    for batch in tr_dl:
        xb = batch[0]
        targets = [batch[i+1] for i in range(4)]
        opt.zero_grad()
        logits = model(xb)
        loss = sum(criterions[i](logits[i], targets[i]) for i in range(4)) / 4
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        # Track 1h predictions for monitoring
        all_pred.extend(logits[0].argmax(1).detach().numpy())
        all_true.extend(targets[0].numpy())

    scheduler.step()
    tl = np.mean(losses)
    ta = accuracy_score(all_true, all_pred)
    tf1 = f1_score(all_true, all_pred, average='macro', zero_division=0)

    # ── Validate ──
    model.eval()
    losses = []; all_pred = []; all_true = []

    with torch.no_grad():
        for batch in va_dl:
            xb = batch[0]
            targets = [batch[i+1] for i in range(4)]
            logits = model(xb)
            loss = sum(criterions[i](logits[i], targets[i]) for i in range(4)) / 4
            losses.append(loss.item())
            all_pred.extend(logits[0].argmax(1).numpy())
            all_true.extend(targets[0].numpy())

    vl = np.mean(losses)
    va = accuracy_score(all_true, all_pred)
    vf1 = f1_score(all_true, all_pred, average='macro', zero_division=0)

    hist['tl'].append(tl); hist['vl'].append(vl)
    hist['ta'].append(ta); hist['va'].append(va)
    hist['tf1'].append(tf1); hist['vf1'].append(vf1)

    mk = ""
    if vl < best_vl:
        best_vl = vl; best_st = {k:v.clone() for k,v in model.state_dict().items()}
        wait = 0; mk = " ★"
    else: wait += 1

    lr_now = opt.param_groups[0]['lr']
    if (ep+1) % 5 == 0 or ep == 0 or mk:
        print(f"  Ep {ep+1:>3d}/{EPOCHS} │ L:{tl:.4f}/{vl:.4f} │ "
              f"1h Acc:{ta:.3f}/{va:.3f} │ 1h macF1:{tf1:.3f}/{vf1:.3f} │ "
              f"lr:{lr_now:.5f}{mk}")

    if wait >= PATIENCE:
        print(f"\n  ⚡ Early stop at epoch {ep+1}"); break

model.load_state_dict(best_st)
print(f"  Best val loss: {best_vl:.4f}")

# ══════════════════════════════════════════
# 7. TEST EVALUATION
# ══════════════════════════════════════════
print("\n"+"="*70+"\n7. TEST EVALUATION\n"+"="*70)

model.eval()
preds = {h: [] for h in HORIZON_NAMES}
probs = {h: [] for h in HORIZON_NAMES}
trues = {h: [] for h in HORIZON_NAMES}

with torch.no_grad():
    for batch in te_dl:
        xb = batch[0]
        targets = [batch[i+1] for i in range(4)]
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

# Persistence baseline: "predict current class continues"
persist_pred = cc_te[:len(trues[HORIZON_NAMES[0]])]

# Results
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
    class_labels = list(range(NUM_CLASSES))
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

# Detailed reports
for h in HORIZON_NAMES:
    r = results[h]
    print(f"\n  ── {h.upper()} Classification Report ──")
    print(classification_report(r['true'], r['pred'],
          labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, zero_division=0))

# ══════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════
print("="*70+"\n8. PLOTS\n"+"="*70)
plt.style.use('seaborn-v0_8-whitegrid')
O = OUTPUT_DIR
clrs = ['#1976D2','#388E3C','#F57C00','#D32F2F']
cls_clrs = ['#4CAF50','#FFC107','#FF9800','#F44336']

# ── Plot 1: Training curves ──
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
ep_r = range(1, len(hist['tl'])+1)

axes[0].plot(ep_r, hist['tl'], '-', color='#1976D2', lw=2, label='Train')
axes[0].plot(ep_r, hist['vl'], '-', color='#F57C00', lw=2, label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('CrossEntropy Loss', fontweight='bold'); axes[0].legend()

axes[1].plot(ep_r, hist['ta'], '-', color='#1976D2', lw=2, label='Train')
axes[1].plot(ep_r, hist['va'], '-', color='#F57C00', lw=2, label='Val')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('1h Accuracy', fontweight='bold'); axes[1].legend()

axes[2].plot(ep_r, hist['tf1'], '-', color='#1976D2', lw=2, label='Train')
axes[2].plot(ep_r, hist['vf1'], '-', color='#F57C00', lw=2, label='Val')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Macro F1')
axes[2].set_title('1h Macro F1 (hard metric)', fontweight='bold'); axes[2].legend()

plt.suptitle('Training Curves (label smoothing + warmup)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{O}/plot1_training.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ plot1_training.png")

# ── Plot 2: Confusion matrices (all 4 horizons) ──
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, h in zip(axes.flat, HORIZON_NAMES):
    r = results[h]
    cm = confusion_matrix(r['true'], r['pred'], labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm.astype(float), row_sums, where=row_sums!=0, out=np.zeros_like(cm, dtype=float))
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
plt.suptitle(f'Confusion Matrices — {NUM_CLASSES}-Class Storm Forecasting', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig(f"{O}/plot2_confusion.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ plot2_confusion.png")

# ── Plot 3: Per-class F1 by horizon ──
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(NUM_CLASSES)
w = 0.18
for i, h in enumerate(HORIZON_NAMES):
    cf1 = results[h]['per_class_f1']
    bars = ax.bar(x + i*w, cf1, w, label=h, color=clrs[i], alpha=0.85)
    for b in bars:
        if b.get_height() > 0.01:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{b.get_height():.3f}', ha='center', fontsize=8.5, fontweight='bold')
ax.set_xticks(x + w*1.5); ax.set_xticklabels(CLASS_NAMES, fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Per-Class F1 by Forecast Horizon\n(lower for rare classes = honest model)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.set_ylim(0, 1.15)
plt.tight_layout(); plt.savefig(f"{O}/plot3_class_f1.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ plot3_class_f1.png")

# ── Plot 4: LSTM vs Persistence ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# Accuracy
ax = axes[0]
x = np.arange(4); w = 0.35
lstm_acc = [results[h]['acc'] for h in HORIZON_NAMES]
pers_acc = [results[h]['persist_acc'] for h in HORIZON_NAMES]
b1 = ax.bar(x-w/2, lstm_acc, w, label='LSTM', color='#1976D2', alpha=.85)
b2 = ax.bar(x+w/2, pers_acc, w, label='Persistence', color='#F57C00', alpha=.85)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f'{b.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f'{b.get_height():.3f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(HORIZON_NAMES, fontsize=12)
ax.set_title('Accuracy: LSTM vs Persistence', fontweight='bold'); ax.legend(); ax.set_ylim(0, 1.15)

# Macro F1
ax = axes[1]
lstm_mf1 = [results[h]['macro_f1'] for h in HORIZON_NAMES]
pers_mf1 = [results[h]['persist_mf1'] for h in HORIZON_NAMES]
b1 = ax.bar(x-w/2, lstm_mf1, w, label='LSTM', color='#1976D2', alpha=.85)
b2 = ax.bar(x+w/2, pers_mf1, w, label='Persistence', color='#F57C00', alpha=.85)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f'{b.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f'{b.get_height():.3f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(HORIZON_NAMES, fontsize=12)
ax.set_title('Macro F1: LSTM vs Persistence\n(macro F1 penalizes ignoring rare classes)', fontweight='bold')
ax.legend(); ax.set_ylim(0, 1.15)

plt.tight_layout(); plt.savefig(f"{O}/plot4_vs_persistence.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ plot4_vs_persistence.png")

# ── Plot 5: Forecast timeline ──
fig, axes = plt.subplots(len(HORIZON_NAMES)+1, 1,
                         figsize=(18, 3*(len(HORIZON_NAMES)+1)), sharex=True)
sn = min(1500, len(trues['1h'])); x = np.arange(sn)

ax = axes[0]
ax.plot(x, trues['1h'][:sn], '-', color='#D32F2F', lw=0.8, alpha=0.7)
ax.fill_between(x, 0, trues['1h'][:sn], alpha=0.3, color='#D32F2F')
ax.set_ylabel('Actual\nClass'); ax.set_yticks(list(range(NUM_CLASSES))); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_title(f'Multi-Horizon {NUM_CLASSES}-Class Forecast Timeline', fontsize=14, fontweight='bold')

short_names = [n[0] for n in CLASS_NAMES]  # Q, W, S
for i, h in enumerate(HORIZON_NAMES):
    ax = axes[i+1]
    ax.plot(x, preds[h][:sn], '-', color=clrs[i], lw=0.8, alpha=0.7, label=f'Predicted {h}')
    ax.plot(x, trues[h][:sn], '-', color='gray', lw=0.5, alpha=0.3, label='Actual')
    ax.set_ylabel(f'{h}'); ax.set_yticks(list(range(NUM_CLASSES))); ax.set_yticklabels(short_names, fontsize=8)
    ax.legend(fontsize=8, loc='upper right')

axes[-1].set_xlabel('Minutes')
plt.tight_layout(); plt.savefig(f"{O}/plot5_timeline.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ plot5_timeline.png")

# ── Plot 6: Summary metrics ──
fig, ax = plt.subplots(figsize=(14, 7))
metrics_list = []
for h in HORIZON_NAMES:
    r = results[h]
    metrics_list.append((f'{h} Acc', r['acc']))
    metrics_list.append((f'{h} macF1', r['macro_f1']))

xp = np.arange(len(metrics_list))
bar_colors = [clrs[i//2] for i in range(len(metrics_list))]
bars = ax.bar(xp, [m[1] for m in metrics_list], color=bar_colors, alpha=0.85)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
            f'{b.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(xp); ax.set_xticklabels([m[0] for m in metrics_list], fontsize=10, rotation=30, ha='right')
ax.set_title(f'Test Metrics — {NUM_CLASSES}-Class Storm Forecasting', fontsize=15, fontweight='bold')
ax.set_ylim(0, 1.15)
plt.tight_layout(); plt.savefig(f"{O}/plot6_metrics.png", dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ plot6_metrics.png")

# ══════════════════════════════════════════
# 9. SAVE
# ══════════════════════════════════════════
torch.save({
    'model': model.state_dict(),
    'scaler_mean': scaler.mean_ if new_feat_cols else None,
    'scaler_scale': scaler.scale_ if new_feat_cols else None,
    'new_feat_cols': new_feat_cols,
    'features': feat_cols, 'class_names': CLASS_NAMES,
    'horizons': dict(zip(HORIZON_NAMES, HORIZONS)),
    'window': WINDOW,
}, f"{O}/model.pt")
print(f"  ✓ model.pt")

# ══════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════
print("\n"+"="*70+"\nFINAL SUMMARY\n"+"="*70)
print(f"""
  {NUM_CLASSES}-Class Forecasting: {' / '.join(CLASS_NAMES)}
  What: "Worst storm class in next H minutes"

┌──────────┬───────────┬───────────┬───────────┐
│ Horizon  │  Accuracy │  Macro F1 │ vs Persist│
├──────────┼───────────┼───────────┼───────────┤""")
for h in HORIZON_NAMES:
    r = results[h]
    diff = r['macro_f1'] - r['persist_mf1']
    print(f"│ {h:>6s}   │   {r['acc']:.4f}  │   {r['macro_f1']:.4f}  │  {diff:>+.4f}  │")
print(f"""└──────────┴───────────┴───────────┴───────────┘

  Key metric: MACRO F1 (not accuracy!)
    Accuracy is misleading when 80%+ of data is "Quiet".
    Macro F1 weights all {NUM_CLASSES} classes equally — if the model
    can't detect rare storms, macro F1 stays low.

  Model simplification:
    ✓ {NUM_CLASSES} classes (Moderate+Intense merged — too few to split)
    ✓ LSTM: {LAYERS} layer, hidden={HIDDEN} (reduced overfitting)
    ✓ No intermediate FC layer (direct LSTM→heads)
    ✓ Dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}
    ✓ Label smoothing ({LABEL_SMOOTH})
    ✓ LR warmup ({WARMUP_EPOCHS} epochs)
    ✓ {GAP}-min gap between train/val/test
    ✓ Persistence baseline comparison

  6 Plots + model → {O}/
""")