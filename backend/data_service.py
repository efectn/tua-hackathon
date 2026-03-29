"""
data_service.py — Fetches solar wind data from NASA CDAWEB HAPI API.

Caches data in-memory with a configurable TTL. Independent from the ML
training dataset — this is for real-time monitoring/visualization only.
"""

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

HAPI_BASE = "https://cdaweb.gsfc.nasa.gov/hapi/data"
DATASET_ID = "OMNI_HRO_1MIN"
PARAMETERS = "F,BZ_GSM,flow_speed,proton_density,T,E,SYM_H"

# NASA OMNI fill values → treat as missing
FILL_VALUES = {
    "F": 9999.99,
    "BZ_GSM": 9999.99,
    "flow_speed": 99999.9,
    "proton_density": 999.99,
    "T": 9999999.0,
    "E": 999.99,
    "SYM_H": 99999,
}

PARAM_META = {
    "F":              {"unit": "nT",   "desc": "Total magnetic field magnitude"},
    "BZ_GSM":         {"unit": "nT",   "desc": "Bz component in GSM (CRITICAL)"},
    "flow_speed":     {"unit": "km/s", "desc": "Solar wind flow speed"},
    "proton_density": {"unit": "n/cc", "desc": "Proton density"},
    "T":              {"unit": "K",    "desc": "Proton temperature"},
    "E":              {"unit": "mV/m", "desc": "Electric field"},
    "SYM_H":          {"unit": "nT",   "desc": "Geomagnetic storm index"},
}


class DataService:
    """In-memory cache + HAPI fetcher for real-time solar wind data."""

    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache_ttl = cache_ttl_seconds
        self._df: pd.DataFrame | None = None
        self._last_fetch: float = 0
        self._last_fetch_iso: str | None = None

    # ── public ────────────────────────────────────────────

    def get_dataframe(self, hours: int = 24, force: bool = False) -> pd.DataFrame:
        """Return cached DataFrame, refreshing from HAPI if stale or needs more hours."""
        last_hours = getattr(self, "_last_hours", 0)
        if force or self._is_stale() or hours > last_hours:
            self._fetch(hours=hours)
            self._last_hours = hours
        return self._df if self._df is not None else pd.DataFrame()

    def update_for_today(self) -> dict:
        """Force-fetch today's full data from HAPI and return summary."""
        # --- TIME SHIFT ---
        # OMNI data is delayed by ~1-2 months. We shift the clock back by 60 days
        # to ensure we get fully populated solar wind data for the simulation.
        now = datetime.now(timezone.utc) #- timedelta(days=60)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = start - timedelta(days=15)
        end = now
        df = self._fetch_range(start, end)
        if df is not None and not df.empty:
            self._df = df
            self._last_fetch = time.time()
            self._last_fetch_iso = now.isoformat()
        return {
            "status": "ok" if df is not None and not df.empty else "no_data",
            "rows": len(df) if df is not None else 0,
            "range": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "last_updated": self._last_fetch_iso,
        }

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch a specific time range (used for test data)."""
        df = self._fetch_range(start, end)
        if df is not None and not df.empty:
            self._df = df
            self._last_fetch = time.time()
            self._last_fetch_iso = datetime.now(timezone.utc).isoformat()
        return df if df is not None else pd.DataFrame()

    @property
    def last_updated(self) -> str | None:
        return self._last_fetch_iso

    @property
    def param_meta(self) -> dict:
        return PARAM_META

    # ── private ───────────────────────────────────────────

    def _is_stale(self) -> bool:
        return (time.time() - self._last_fetch) > self.cache_ttl

    def _fetch(self, hours: int = 24):
        # --- TIME SHIFT ---
        # OMNI data is delayed by ~1-2 months. We shift the clock back by 60 days
        # to ensure we get fully populated solar wind data for the simulation.
        now = datetime.now(timezone.utc) - timedelta(days=60)
        start = now - timedelta(hours=hours)
        df = self._fetch_range(start, now)
        if df is not None and not df.empty:
            self._df = df
            self._last_fetch = time.time()
            self._last_fetch_iso = now.isoformat()

    def _fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        """Hit the HAPI endpoint and return a clean DataFrame."""
        params = {
            "id": DATASET_ID,
            "parameters": PARAMETERS,
            "time.min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time.max": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "format": "json",
        }
        try:
            print(f"[DataService] Fetching HAPI: {start.isoformat()} → {end.isoformat()}")
            resp = requests.get(HAPI_BASE, params=params, timeout=60)
            print(resp.url)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"[DataService] HAPI fetch failed: {e}")
            return None

        if "data" not in payload or not payload["data"]:
            print("[DataService] No data in HAPI response")
            return None

        columns = [p["name"] for p in payload["parameters"]]
        df = pd.DataFrame(payload["data"], columns=columns)

        # Parse time
        df["Time"] = pd.to_datetime(df["Time"])
        df.set_index("Time", inplace=True)
        df.sort_index(inplace=True)

        # Numeric conversion
        for col in FILL_VALUES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace fill values with NaN
        for col, fv in FILL_VALUES.items():
            if col in df.columns:
                df[col] = df[col].replace(fv, np.nan)

        # Interpolate missing values (time-series aware)
        df = df.interpolate(method="time").ffill().bfill()

        print(f"[DataService] Loaded {len(df)} rows")
        return df

    def load_test_data(self, json_path: str) -> pd.DataFrame:
        """Load data from a local JSON file (same HAPI format) for testing."""
        import os
        import json as _json

        mtime = os.path.getmtime(json_path)
        if getattr(self, "_last_json_path", None) == json_path and getattr(self, "_last_json_mtime", 0) == mtime:
            return self._df

        with open(json_path, "r") as f:
            payload = _json.load(f)

        columns = [p["name"] for p in payload["parameters"]]
        df = pd.DataFrame(payload["data"], columns=columns)

        df["Time"] = pd.to_datetime(df["Time"])
        df.set_index("Time", inplace=True)
        df.sort_index(inplace=True)

        for col in FILL_VALUES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col, fv in FILL_VALUES.items():
            if col in df.columns:
                df[col] = df[col].replace(fv, np.nan)

        df = df.interpolate(method="time").ffill().bfill()

        self._df = df
        self._last_fetch = time.time()
        self._last_fetch_iso = datetime.now(timezone.utc).isoformat()
        self._last_json_path = json_path
        self._last_json_mtime = mtime
        print(f"[DataService] Loaded test data: {len(df)} rows from {json_path}")
        return df
