"""
storm_analyzer.py — Storm detection, risk analysis, and LSTM model inference.

Uses the same thresholds as model_yeni.py:
  - Quiet:  SYM-H > -50
  - Weak:   -100 < SYM-H <= -50
  - Storm:  SYM-H <= -100
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ── Constants (match model_yeni.py) ──────────────────────
NUM_CLASSES = 3
CLASS_NAMES = ["Sakin", "Zayıf", "Fırtına"]
HORIZON_NAMES = ["1h", "2h", "4h", "12h"]
HORIZONS = [60, 120, 240, 720]
WINDOW = 30
HIDDEN = 32
LAYERS = 1
DROPOUT = 0.5

RAW_FEATS = ["F", "BZ_GSM", "flow_speed", "proton_density", "T"]


# ── LSTM model (must match model_yeni.py exactly) ───────
class StormLSTM(nn.Module):
    def __init__(self, inp, hid=HIDDEN, layers=LAYERS, drop=DROPOUT,
                 n_heads=4, n_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hid)
        self.drop = nn.Dropout(drop)
        self.heads = nn.ModuleList([nn.Linear(hid, n_classes) for _ in range(n_heads)])

    def forward(self, x):
        o, _ = self.lstm(x)
        h = self.drop(self.bn(o[:, -1, :]))
        return [head(h) for head in self.heads]


def sym_to_class(sym_val: float) -> int:
    """Convert SYM-H value to storm class."""
    if sym_val > -50:
        return 0  # Quiet
    elif sym_val > -100:
        return 1  # Weak
    else:
        return 2  # Storm (Active)


def sym_to_label(sym_val: float) -> str:
    return CLASS_NAMES[sym_to_class(sym_val)]


class StormAnalyzer:
    """Loads the trained LSTM and provides rule-based + model-based analysis."""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.checkpoint = None
        self.feat_cols = None
        self.scaler = None

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            print(f"[StormAnalyzer] Model loaded from {model_path}")
        else:
            print(f"[StormAnalyzer] No model loaded (rule-based only)")

    # ── Model loading ─────────────────────────────────────

    def _load_model(self, path: str):
        self.checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.feat_cols = self.checkpoint["features"]
        n_features = len(self.feat_cols)

        self.model = StormLSTM(n_features)
        self.model.load_state_dict(self.checkpoint["model"])
        self.model.eval()

        # Rebuild scaler for new features
        if self.checkpoint.get("scaler_mean") is not None:
            self.scaler = StandardScaler()
            self.scaler.mean_ = self.checkpoint["scaler_mean"]
            self.scaler.scale_ = self.checkpoint["scaler_scale"]
            self.scaler.var_ = self.scaler.scale_ ** 2
            self.scaler.n_features_in_ = len(self.checkpoint["new_feat_cols"])
            self._new_feat_cols = self.checkpoint["new_feat_cols"]
        else:
            self.scaler = None
            self._new_feat_cols = []

    # ── Feature engineering (mirrors model_yeni.py) ───────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as model_yeni.py."""
        df = df.copy()

        # Ensure raw features exist
        for col in RAW_FEATS:
            if col not in df.columns:
                df[col] = 0.0

        # Rolling stats on raw features
        for c in RAW_FEATS:
            df[f"{c}_m30"] = df[c].rolling(30, min_periods=1).mean()
            df[f"{c}_s30"] = df[c].rolling(30, min_periods=1).std().fillna(0)
            df[f"{c}_min30"] = df[c].rolling(30, min_periods=1).min()
            df[f"{c}_max30"] = df[c].rolling(30, min_periods=1).max()

        # Physics features
        df["Bz_south"] = (df["BZ_GSM"] < 0).astype(float)
        df["Bz_south_dur"] = df["Bz_south"].rolling(60, min_periods=1).sum()
        df["dyn_pressure"] = df["proton_density"] * df["flow_speed"] ** 2 * 1.6726e-6

        df.dropna(inplace=True)
        return df

    # ── LSTM inference ────────────────────────────────────

    def predict_with_model(self, df: pd.DataFrame) -> dict | None:
        """
        Run the LSTM on the tail of the dataframe.
        Returns class probabilities for each horizon.
        """
        if self.model is None or self.feat_cols is None:
            return None

        df_feat = self._engineer_features(df)
        if len(df_feat) < WINDOW:
            return None

        # Build feature matrix using exactly the same columns
        available_cols = [c for c in self.feat_cols if c in df_feat.columns]
        if len(available_cols) != len(self.feat_cols):
            missing = set(self.feat_cols) - set(available_cols)
            # Add missing columns as zeros (they'll be scaled)
            for col in missing:
                df_feat[col] = 0.0
            available_cols = self.feat_cols

        feat_values = df_feat[available_cols].values.astype(np.float32)

        # Scale new features (same as training)
        if self.scaler is not None and self._new_feat_cols:
            new_col_indices = [
                available_cols.index(c) for c in self._new_feat_cols
                if c in available_cols
            ]
            if new_col_indices:
                feat_values[:, new_col_indices] = self.scaler.transform(
                    feat_values[:, new_col_indices]
                )

        # Take last WINDOW rows as input sequence
        seq = feat_values[-WINDOW:]
        x = torch.FloatTensor(seq).unsqueeze(0)  # (1, WINDOW, n_features)

        with torch.no_grad():
            logits_list = self.model(x)
            result = {}
            for i, hname in enumerate(HORIZON_NAMES):
                probs = torch.softmax(logits_list[i], dim=1).numpy()[0]
                pred_class = int(probs.argmax())
                result[hname] = {
                    "predicted_class": pred_class,
                    "class_name": CLASS_NAMES[pred_class],
                    "probabilities": {
                        CLASS_NAMES[j]: round(float(probs[j]), 4)
                        for j in range(NUM_CLASSES)
                    },
                    "confidence": round(float(probs.max()), 4),
                }
            return result

    # ── Rule-based risk assessment ────────────────────────

    def assess_risk(self, df: pd.DataFrame) -> dict:
        """
        Rule-based storm risk assessment for upcoming windows.
        Uses recent solar wind conditions to estimate risk levels.
        """
        if df.empty or len(df) < 5:
            return self._empty_risk()

        recent = df.tail(60)  # last 60 minutes
        latest = df.iloc[-1]

        bz = latest.get("BZ_GSM", 0)
        speed = latest.get("flow_speed", 400)
        density = latest.get("proton_density", 5)
        sym_h = latest.get("SYM_H", 0)
        e_field = latest.get("E", 0)

        # Derived
        bz_mean_30 = recent["BZ_GSM"].tail(30).mean() if "BZ_GSM" in recent else 0
        bz_mean_60 = recent["BZ_GSM"].mean() if "BZ_GSM" in recent else 0
        speed_mean = recent["flow_speed"].mean() if "flow_speed" in recent else 400
        dyn_pressure = density * speed ** 2 * 1.6726e-6

        # Current status
        current_class = sym_to_class(sym_h)
        current_status = CLASS_NAMES[current_class].lower()

        predictions = {}
        for hname, hmins in zip(HORIZON_NAMES, HORIZONS):
            score, reasons = self._compute_risk_score(
                bz, bz_mean_30, bz_mean_60, speed, speed_mean,
                density, dyn_pressure, sym_h, e_field, hmins
            )
            risk_level = self._score_to_level(score)
            predictions[hname] = {
                "risk": risk_level,
                "score": round(score, 2),
                "level": ["low", "moderate", "high", "extreme"].index(risk_level),
                "reasons": reasons,
            }

        return {
            "predictions": predictions,
            "current_conditions": {
                "BZ_GSM": round(float(bz), 2),
                "flow_speed": round(float(speed), 1),
                "proton_density": round(float(density), 2),
                "SYM_H": int(sym_h),
                "E": round(float(e_field), 2),
                "dyn_pressure": round(float(dyn_pressure), 3),
                "status": current_status,
                "class": current_class,
            },
        }

    def _compute_risk_score(self, bz, bz_30, bz_60, speed, speed_mean,
                            density, dyn_pressure, sym_h, e_field, horizon_min):
        """Compute 0–10 risk score based on solar wind conditions."""
        score = 0.0
        reasons = []

        # BZ_GSM (most critical) — southward = negative
        if bz < -10:
            score += 3.5
            reasons.append(f"Bz strongly southward ({bz:.1f} nT)")
        elif bz < -5:
            score += 2.0
            reasons.append(f"Bz moderately southward ({bz:.1f} nT)")
        elif bz < 0:
            score += 0.5
            reasons.append(f"Bz slightly southward ({bz:.1f} nT)")
        else:
            reasons.append(f"Bz northward ({bz:.1f} nT) — protective")

        # Sustained Bz southward (30-min mean)
        if bz_30 < -8:
            score += 1.5
            reasons.append(f"Sustained southward Bz (30m avg: {bz_30:.1f} nT)")
        elif bz_30 < -3:
            score += 0.5

        # Flow speed
        if speed > 700:
            score += 2.0
            reasons.append(f"Very fast solar wind ({speed:.0f} km/s)")
        elif speed > 500:
            score += 1.0
            reasons.append(f"Fast solar wind ({speed:.0f} km/s)")
        elif speed > 400:
            score += 0.3

        # Dynamic pressure
        if dyn_pressure > 10:
            score += 1.5
            reasons.append(f"High dynamic pressure ({dyn_pressure:.1f} nPa)")
        elif dyn_pressure > 5:
            score += 0.5

        # Current SYM-H (already in storm?)
        if sym_h < -100:
            score += 2.0
            reasons.append(f"Storm already active (SYM-H: {sym_h} nT)")
        elif sym_h < -50:
            score += 1.0
            reasons.append(f"Weak storm conditions (SYM-H: {sym_h} nT)")

        # Longer horizons have higher uncertainty → slightly increase score
        horizon_factor = 1.0 + (horizon_min / 720) * 0.3
        score *= horizon_factor

        return min(score, 10.0), reasons

    def _score_to_level(self, score: float) -> str:
        if score >= 7:
            return "extreme"
        elif score >= 4:
            return "high"
        elif score >= 2:
            return "moderate"
        return "low"

    def _empty_risk(self) -> dict:
        return {
            "predictions": {
                h: {"risk": "unknown", "score": 0, "level": -1, "reasons": ["No data"]}
                for h in HORIZON_NAMES
            },
            "current_conditions": {},
        }

    # ── Storm event detection ─────────────────────────────

    def detect_storms(self, df: pd.DataFrame) -> list[dict]:
        """
        Detect storm events from SYM-H data.
        A storm starts when SYM-H drops below -50 and ends when it recovers.
        """
        if df.empty or "SYM_H" not in df.columns:
            return []

        storms = []
        in_storm = False
        storm_start = None
        peak_sym_h = 0
        peak_time = None

        for ts, row in df.iterrows():
            sym_h = row["SYM_H"]

            if not in_storm and sym_h <= -50:
                in_storm = True
                storm_start = ts
                peak_sym_h = sym_h
                peak_time = ts
            elif in_storm:
                if sym_h < peak_sym_h:
                    peak_sym_h = sym_h
                    peak_time = ts
                if sym_h > -30:  # recovered
                    in_storm = False
                    storms.append({
                        "start": storm_start.isoformat(),
                        "peak": peak_time.isoformat(),
                        "end": ts.isoformat(),
                        "peak_sym_h": round(float(peak_sym_h), 1),
                        "classification": sym_to_label(peak_sym_h),
                        "class": sym_to_class(peak_sym_h),
                        "duration_hours": round(
                            (ts - storm_start).total_seconds() / 3600, 1
                        ),
                    })

        # If still in storm at end of data
        if in_storm:
            storms.append({
                "start": storm_start.isoformat(),
                "peak": peak_time.isoformat(),
                "end": None,
                "peak_sym_h": round(float(peak_sym_h), 1),
                "classification": sym_to_label(peak_sym_h),
                "class": sym_to_class(peak_sym_h),
                "duration_hours": round(
                    (df.index[-1] - storm_start).total_seconds() / 3600, 1
                ),
                "ongoing": True,
            })

        return storms

    # ── Current storm status (for 3D simulation) ──────────

    def current_storm_status(self, df: pd.DataFrame) -> dict:
        """
        Return the current storm state for the 3D simulation.
        Includes severity, Bz direction, speed, etc.
        """
        if df.empty:
            return {
                "active": False,
                "severity": "none",
                "description": "No data available",
            }

        latest = df.iloc[-1]
        sym_h = latest.get("SYM_H", 0)
        bz = latest.get("BZ_GSM", 0)
        speed = latest.get("flow_speed", 400)
        density = latest.get("proton_density", 5)
        f_total = latest.get("F", 5)
        e_field = latest.get("E", 0)
        temp = latest.get("T", 100000)

        storm_class = sym_to_class(sym_h)
        is_active = storm_class > 0
        dyn_pressure = density * speed ** 2 * 1.6726e-6

        # Determine Bz direction for simulation
        if bz < -5:
            direction = "strongly_southward"
        elif bz < 0:
            direction = "southward"
        elif bz > 5:
            direction = "strongly_northward"
        else:
            direction = "northward"

        # Find storm start time if active
        started_at = None
        if is_active and len(df) > 1:
            for i in range(len(df) - 1, -1, -1):
                if df.iloc[i]["SYM_H"] > -50:
                    if i + 1 < len(df):
                        started_at = df.index[i + 1].isoformat()
                    break

        # Activity level for simulation intensity
        if sym_h <= -200:
            intensity = 1.0
            severity = "extreme"
            desc = "Extreme geomagnetic storm"
        elif sym_h <= -100:
            intensity = 0.7
            severity = "storm"
            desc = "Geomagnetic storm in progress"
        elif sym_h <= -50:
            intensity = 0.4
            severity = "weak"
            desc = "Weak geomagnetic storm"
        else:
            intensity = max(0.0, min(0.2, (-sym_h) / 250))
            severity = "quiet"
            desc = "Geomagnetically quiet"

        # Solar wind visualization params
        wind_intensity = min(1.0, speed / 800)

        return {
            "active": is_active,
            "severity": severity,
            "intensity": round(intensity, 3),
            "sym_h": round(float(sym_h), 1),
            "bz_gsm": round(float(bz), 2),
            "flow_speed": round(float(speed), 1),
            "proton_density": round(float(density), 2),
            "temperature": round(float(temp), 0),
            "f_total": round(float(f_total), 2),
            "e_field": round(float(e_field), 2),
            "dynamic_pressure": round(float(dyn_pressure), 3),
            "direction": direction,
            "wind_intensity": round(wind_intensity, 3),
            "started_at": started_at,
            "classification": CLASS_NAMES[storm_class],
            "description": desc,
            "timestamp": df.index[-1].isoformat(),
        }
