"""
Flask Backend for Solar Storm Real-Time Monitoring
===================================================
Independent from the ML training pipeline.
Fetches live data from NASA CDAWEB HAPI API and provides:

  GET  /latest-values   — real-time graph data
  GET  /notifications   — storm risk + LSTM predictions
  GET  /timeline        — detected storm events
  POST /update-data     — refresh data from HAPI
  GET  /current-storms  — active storm for 3D simulation

Query params:
  ?test=1                — use local test_data.json instead of HAPI
  ?source=<path>         — specify custom test data JSON path
  ?hours=24              — how many hours of data to fetch (default: 24)
"""

import os
import sys
import time
from functools import wraps

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timezone

# Add parent dir so we can find model files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_service import DataService
from storm_analyzer import StormAnalyzer

# ── Configuration ────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.pt"
)
DEFAULT_TEST_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_test.json"
)
CACHE_TTL = 300  # 5 minutes

# ── Init services ────────────────────────────────────────
app = Flask(__name__)
CORS(app)

data_service = DataService(cache_ttl_seconds=CACHE_TTL)
analyzer = StormAnalyzer(model_path=MODEL_PATH)

# Simple generic response cache
cache_store = {}

def simple_cache(ttl=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            key = f.__name__ + str(request.url)
            if key in cache_store:
                val, timestamp = cache_store[key]
                if time.time() - timestamp < ttl:
                    return val
            res = f(*args, **kwargs)
            # Only cache success JSON responses roughly.
            if isinstance(res, tuple) and res[1] != 200:
                return res
            cache_store[key] = (res, time.time())
            return res
        return wrapped
    return decorator


test_override_active = False

def _get_df(no_hours = False, default_hours="24"):
    """Get DataFrame, optionally using test data if ?test=1."""
    test_mode = request.args.get("test", "0") == "1"
    source = request.args.get("source", None)
    if no_hours:
        hours = None
    else:
        hours = int(request.args.get("hours", default_hours))

    if test_mode:
        path = source or DEFAULT_TEST_DATA
        if not os.path.exists(path):
            return None, f"Test data file not found: {path}"
        return data_service.load_test_data(path), None
    else:
        # If test override is active, use injected data directly
        if test_override_active and data_service._df is not None and not data_service._df.empty:
            return data_service._df, None

        backend_data_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
        if os.path.exists(backend_data_json):
            return data_service.load_test_data(backend_data_json), None

        df = data_service.get_dataframe(hours=hours)
        if df.empty:
            return None, "No data available. Try POST /update-data first."
        return df, None


# ── Routes ───────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)),
        "index.html"
    )


@app.route("/api")
def api_info():
    return jsonify({
        "service": "Solar Storm Monitoring API",
        "version": "1.0.0",
        "endpoints": {
            "/latest-values": "GET — Real-time parameter values for graphs",
            "/notifications": "GET — Storm risk predictions (rule-based + LSTM)",
            "/timeline":      "GET — Detected storm events",
            "/update-data":   "POST — Refresh data from CDAWEB HAPI",
            "/current-storms": "GET — Active storm status for 3D simulation",
            "/test/inject-storm": "POST — Inject a fake storm",
            "/test/reset": "POST — Reset to real data.json",
        },
        "params": {
            "test": "Set to 1 to use local test data instead of HAPI",
            "source": "Path to custom test data JSON (used with test=1)",
            "hours": "How many hours of data to fetch (default: 24)",
        },
    })


@app.route("/latest-values")
@simple_cache(ttl=10)
def latest_values():
    """
    Return last N hours of solar wind parameters for real-time graphs.
    Supports optional ?limit=N to return only the last N data points.
    """
    df, err = _get_df()
    if df is None:
        return jsonify({"error": err}), 404

    limit = request.args.get("limit", None)
    if limit:
        df = df.tail(int(limit))

    # Convert to JSON-friendly format
    result = {
        "timestamps": [ts.isoformat() for ts in df.index],
        "F": _safe_list(df, "F"),
        "BZ_GSM": _safe_list(df, "BZ_GSM"),
        "flow_speed": _safe_list(df, "flow_speed"),
        "proton_density": _safe_list(df, "proton_density"),
        "T": _safe_list(df, "T"),
        "E": _safe_list(df, "E"),
        "SYM_H": _safe_list(df, "SYM_H"),
        "count": len(df),
        "last_updated": data_service.last_updated,
        "meta": data_service.param_meta,
    }
    return jsonify(result)


@app.route("/notifications")
@simple_cache(ttl=10)
def notifications():
    """
    Storm risk assessment combining rule-based analysis + LSTM model predictions.
    Returns risk levels for 1h, 2h, 4h, 12h windows.
    """
    df, err = _get_df()
    if df is None:
        return jsonify({"error": err}), 404

    # Rule-based risk analysis
    risk = analyzer.assess_risk(df)

    # LSTM model predictions
    model_predictions = analyzer.predict_with_model(df)

    if model_predictions:
        risk["model_predictions"] = model_predictions
        # Merge model predictions into each horizon's result
        for hname in ["1h", "2h", "4h", "12h"]:
            if hname in risk["predictions"] and hname in model_predictions:
                mp = model_predictions[hname]
                risk["predictions"][hname]["model"] = {
                    "predicted_class": mp["class_name"],
                    "confidence": mp["confidence"],
                    "probabilities": mp["probabilities"],
                }
    else:
        risk["model_predictions"] = None
        risk["model_note"] = "LSTM model not available or insufficient data (need 30+ minutes)"

    return jsonify(risk)


@app.route("/timeline")
@simple_cache(ttl=10)
def timeline():
    """
    Return detected storm events from the data.
    Storms are detected by SYM-H dropping below -50 nT.
    """
    df, err = _get_df()
    if df is None:
        return jsonify({"error": err}), 404

    storms = analyzer.detect_storms(df)

    return jsonify({
        "storms": storms,
        "total": len(storms),
        "data_range": {
            "start": df.index[0].isoformat() if len(df) > 0 else None,
            "end": df.index[-1].isoformat() if len(df) > 0 else None,
        },
    })


@app.route("/update-data", methods=["POST"])
def update_data():
    """
    Trigger a data refresh from CDAWEB HAPI for today.
    Optionally pass ?hours=48 to fetch more history.
    """
    test_mode = request.args.get("test", "0") == "1"
    source = request.args.get("source", None)

    if test_mode:
        path = source or DEFAULT_TEST_DATA
        if not os.path.exists(path):
            return jsonify({"error": f"Test data not found: {path}"}), 404
        df = data_service.load_test_data(path)
        return jsonify({
            "status": "ok",
            "source": "test_data",
            "path": path,
            "rows": len(df),
        })

    backend_data_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
    if os.path.exists(backend_data_json):
        df = data_service.load_test_data(backend_data_json)
        return jsonify({
            "status": "ok",
            "source": "data.json",
            "path": backend_data_json,
            "rows": len(df),
        })

    result = data_service.update_for_today()
    return jsonify(result)


@app.route("/current-storms")
@simple_cache(ttl=10)
def current_storms():
    """
    Return current storm status for the 3D JS/HTML/CSS simulation.
    Includes severity, Bz direction, wind speed, and visualization params.
    """
    df, err = _get_df(default_hours="72")
    if df is None:
        return jsonify({"error": err}), 404

    status = analyzer.current_storm_status(df)

    # Include ongoing storms instead of model predictions (1h, 12h, etc.)
    storms = analyzer.detect_storms(df)
    ongoing_storms = [s for s in storms if s.get("ongoing")]
    if ongoing_storms:
        status["ongoing_storms"] = ongoing_storms

    return jsonify(status)


# ── Test Data Injection ──────────────────────────────────

@app.route("/test/normal", methods=["POST"])
def test_inject_normal():
    """Inject 24h of quiet/normal solar wind data for testing."""
    import numpy as np

    n = 1440  # 24 hours of 1-min data
    now = datetime.now(timezone.utc)
    times = pd.date_range(end=now, periods=n, freq="1min")

    df = pd.DataFrame({
        "F": np.random.normal(5, 0.5, n),
        "BZ_GSM": np.random.normal(2, 1.5, n),       # slightly northward
        "flow_speed": np.random.normal(380, 20, n),    # calm
        "proton_density": np.random.normal(5, 1, n),
        "T": np.random.normal(80000, 10000, n),
        "E": np.random.normal(0.3, 0.2, n),
        "SYM_H": np.random.normal(-10, 5, n).astype(int),  # quiet
    }, index=times)
    df.index.name = "Time"

    data_service._df = df
    data_service._last_fetch = time.time()
    data_service._last_fetch_iso = now.isoformat()
    data_service._last_json_path = None  # bypass file cache
    cache_store.clear()
    global test_override_active
    test_override_active = True

    return jsonify({"status": "ok", "mode": "normal", "rows": len(df),
                    "description": "24h of quiet data injected. BZ~+2, speed~380, SYM-H~-10"})


@app.route("/test/storm", methods=["POST"])
def test_inject_storm():
    """Inject 24h of data with a storm event for testing alerts."""
    import numpy as np

    n = 1440
    now = datetime.now(timezone.utc)
    times = pd.date_range(end=now, periods=n, freq="1min")

    # Build a storm profile: quiet → pre-storm → storm → recovery
    bz = np.full(n, 2.0)
    speed = np.full(n, 400.0)
    density = np.full(n, 5.0)
    sym_h = np.full(n, -10.0)
    e_field = np.full(n, 0.3)

    # Storm onset at ~6h ago (minute 1080), peak at ~3h ago (1260), recovery after
    # Pre-storm ramp (1020-1080): BZ starts going south
    for i in range(1020, 1080):
        t = (i - 1020) / 60
        bz[i] = 2 - 15 * t
        speed[i] = 400 + 300 * t
        density[i] = 5 + 20 * t

    # Main phase (1080-1260): deep storm
    for i in range(1080, 1260):
        t = (i - 1080) / 180
        bz[i] = -13 - 7 * np.sin(np.pi * t)
        speed[i] = 700 + 100 * np.sin(np.pi * t)
        density[i] = 25 + 10 * np.sin(np.pi * t)
        sym_h[i] = -50 - 150 * np.sin(np.pi * t)  # peaks at -200
        e_field[i] = 8 + 7 * np.sin(np.pi * t)

    # Recovery (1260-1440): gradually return to quiet
    for i in range(1260, n):
        t = (i - 1260) / 180
        bz[i] = -13 + 15 * min(t, 1)
        speed[i] = 700 - 300 * min(t, 1)
        density[i] = 25 - 20 * min(t, 1)
        sym_h[i] = -200 + 190 * min(t, 1)
        e_field[i] = 15 - 14.7 * min(t, 1)

    # Add noise
    bz += np.random.normal(0, 0.5, n)
    speed += np.random.normal(0, 10, n)
    density += np.random.normal(0, 0.5, n)
    sym_h += np.random.normal(0, 3, n)
    e_field += np.random.normal(0, 0.2, n)

    df = pd.DataFrame({
        "F": np.abs(bz) + np.random.normal(2, 0.3, n),
        "BZ_GSM": bz,
        "flow_speed": np.clip(speed, 250, 1200),
        "proton_density": np.clip(density, 1, 60),
        "T": np.random.normal(150000, 30000, n),
        "E": np.clip(e_field, -5, 25),
        "SYM_H": sym_h.astype(int),
    }, index=times)
    df.index.name = "Time"

    data_service._df = df
    data_service._last_fetch = time.time()
    data_service._last_fetch_iso = now.isoformat()
    data_service._last_json_path = None
    cache_store.clear()
    global test_override_active
    test_override_active = True

    return jsonify({"status": "ok", "mode": "storm", "rows": len(df),
                    "description": "24h with storm injected. Peak: BZ~-20, speed~800, SYM-H~-200 about 3h ago"})


@app.route("/test/reset", methods=["POST"])
def test_reset():
    """Reset back to real data.json."""
    global test_override_active
    test_override_active = False
    data_service._last_json_path = None
    cache_store.clear()
    return jsonify({"status": "ok", "description": "Cache cleared. Next request will reload data.json"})


# ── Helpers ──────────────────────────────────────────────

def _safe_list(df, col):
    """Convert a DataFrame column to a JSON-safe list (NaN → None)."""
    if col not in df.columns:
        return []
    values = df[col].tolist()
    return [None if (isinstance(v, float) and (v != v)) else round(v, 4) if isinstance(v, float) else v for v in values]


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Solar Storm Monitoring API")
    print("=" * 60)
    print(f"  Model: {'Loaded ✓' if analyzer.model else 'Not found ✗'}")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Test data: {DEFAULT_TEST_DATA}")
    print(f"  Cache TTL: {CACHE_TTL}s")
    print()
    print("  Endpoints:")
    print("    GET  /latest-values   — real-time graph data")
    print("    GET  /notifications   — storm risk + LSTM predictions")
    print("    GET  /timeline        — detected storm events")
    print("    POST /update-data     — refresh data from HAPI")
    print("    GET  /current-storms  — 3D simulation data")
    print("    POST /test/inject-storm — inject a fake storm")
    print("    POST /test/reset      — reset to real data.json")    
    print()
    print("  Add ?test=1 to any GET to use local test data")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=True)
