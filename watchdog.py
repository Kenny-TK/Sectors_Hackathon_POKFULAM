#!/usr/bin/env python3
"""
IDX Smart Alerts & Automated Watchdog
======================================

Pipeline:
  1. Fetch IDX market data from the Sectors API (api.sectors.app).
  2. Run deterministic Python logic (0 LLM tokens) to find anomalies:
     top foreign net buy/sell, broker accumulation, volume spikes.
  3. Compress the findings into a tiny dict (~<100 input tokens).
  4. Ask an LLM (OpenAI gpt-4o-mini by default) for a 2-sentence
     executive summary of that tiny dict.
  5. Post a rich embed with the summary + detail tables to Discord.

Design notes:
  - All numeric thresholds/filtering happen in plain Python BEFORE
    anything is sent to the LLM, so the LLM never sees raw JSON.
  - Network calls are isolated behind small functions with explicit
    timeouts and status checks so failures are easy to diagnose in
    GitHub Actions logs.
  - Field names for the Sectors API responses are read defensively
    (multiple candidate keys, `.get()` everywhere) because the exact
    response schema can vary by plan/version. Search the printed
    `[DEBUG] raw sample:` output (see --debug) against your actual
    API response and adjust the `_pick()` calls in the parsing
    functions below if your account returns different field names.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SECTORS_BASE_URL = "https://api.sectors.app"
REQUEST_TIMEOUT_SECONDS = 15

# Deterministic anomaly thresholds -- tune these to taste.
TOP_N_FOREIGN_FLOW = 3
BROKER_ACCUMULATION_RATIO_THRESHOLD = 2.0
VOLUME_SPIKE_MULTIPLE_THRESHOLD = 1.5
TOP_N_BROKER_ACCUMULATION = 3
TOP_N_VOLUME_SPIKES = 3

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()  # "openai" or "gemini"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

LLM_SYSTEM_PROMPT = (
    "You are a concise financial market analyst. Summarize the provided "
    "IDX market anomalies in exactly 2 executive sentences. Do not repeat "
    "numbers given in the prompt verbatim; give strategic market context."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("idx_watchdog")


# --------------------------------------------------------------------------
# Config / secrets loading
# --------------------------------------------------------------------------

@dataclass
class Settings:
    sectors_api_key: str
    llm_api_key: str
    discord_webhook_url: str
    debug: bool = False


def load_settings() -> Settings:
    """Read required secrets from environment variables (GitHub Secrets)."""
    missing = []
    sectors_api_key = os.environ.get("SECTORS_API_KEY", "")
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not sectors_api_key:
        missing.append("SECTORS_API_KEY")
    if not llm_api_key:
        missing.append("LLM_API_KEY")
    if not discord_webhook_url:
        missing.append("DISCORD_WEBHOOK_URL")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them as GitHub Secrets (see README)."
        )

    return Settings(
        sectors_api_key=sectors_api_key,
        llm_api_key=llm_api_key,
        discord_webhook_url=discord_webhook_url,
        debug=os.environ.get("WATCHDOG_DEBUG", "false").lower() == "true",
    )


# --------------------------------------------------------------------------
# Step 1: Fetch market data from the Sectors API
# --------------------------------------------------------------------------

class SectorsAPIError(RuntimeError):
    """Raised when the Sectors API returns an error or unusable response."""


def _sectors_get(path: str, api_key: str, params: Optional[dict] = None) -> Any:
    """
    GET a Sectors API endpoint with auth header, timeout, and error handling.
    Returns parsed JSON on success; raises SectorsAPIError otherwise.
    """
    url = f"{SECTORS_BASE_URL}{path}"
    headers = {"Authorization": api_key}
    try:
        resp = requests.get(
            url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.exceptions.Timeout as exc:
        raise SectorsAPIError(f"Timeout calling {path}") from exc
    except requests.exceptions.RequestException as exc:
        raise SectorsAPIError(f"Network error calling {path}: {exc}") from exc

    if resp.status_code != 200:
        raise SectorsAPIError(
            f"Sectors API {path} returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    try:
        return resp.json()
    except ValueError as exc:
        raise SectorsAPIError(f"Non-JSON response from {path}") from exc


def fetch_daily_foreign_flow(api_key: str) -> list[dict]:
    """GET /v2/daily-foreign-flow/ -- per-ticker foreign buy/sell for the day."""
    try:
        data = _sectors_get("/v2/daily-foreign-flow/", api_key)
    except SectorsAPIError as exc:
        logger.warning("daily-foreign-flow fetch failed: %s", exc)
        return []
    return _as_list(data)


def fetch_top_brokers(api_key: str) -> list[dict]:
    """GET /v2/top-brokers/ -- brokers with the largest net activity."""
    try:
        data = _sectors_get("/v2/top-brokers/", api_key)
    except SectorsAPIError as exc:
        logger.warning("top-brokers fetch failed: %s", exc)
        return []
    return _as_list(data)


def fetch_broker_activity(api_key: str) -> list[dict]:
    """GET /v2/broker-activity/ -- per-ticker broker accumulation/volume data."""
    try:
        data = _sectors_get("/v2/broker-activity/", api_key)
    except SectorsAPIError as exc:
        logger.warning("broker-activity fetch failed: %s", exc)
        return []
    return _as_list(data)


def _as_list(data: Any) -> list[dict]:
    """Normalize a Sectors API payload into a list of row dicts."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def _pick(row: dict, *candidate_keys: str, default: Any = None) -> Any:
    """Return the first present key from a list of candidate field names."""
    for key in candidate_keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Step 2: Deterministic anomaly detection (0 LLM tokens)
# --------------------------------------------------------------------------

@dataclass
class Anomalies:
    top_foreign_buy: list[dict] = field(default_factory=list)
    top_foreign_sell: list[dict] = field(default_factory=list)
    broker_accumulation: list[dict] = field(default_factory=list)
    volume_spikes: list[dict] = field(default_factory=list)


def detect_foreign_flow_anomalies(rows: list[dict], top_n: int = TOP_N_FOREIGN_FLOW):
    """Rank tickers by net foreign flow; return top buys and top sells."""
    parsed = []
    for row in rows:
        ticker = _pick(row, "symbol", "ticker", "code", default="UNKNOWN")
        net = _to_float(_pick(row, "net_foreign_flow", "foreign_net", "net", default=0))
        if ticker == "UNKNOWN" and net == 0:
            continue
        parsed.append({"ticker": ticker, "net_foreign_flow": net})

    parsed.sort(key=lambda r: r["net_foreign_flow"], reverse=True)
    top_buy = [r for r in parsed if r["net_foreign_flow"] > 0][:top_n]
    top_sell = sorted(
        [r for r in parsed if r["net_foreign_flow"] < 0],
        key=lambda r: r["net_foreign_flow"],
    )[:top_n]
    return top_buy, top_sell


def detect_broker_accumulation(
    rows: list[dict],
    ratio_threshold: float = BROKER_ACCUMULATION_RATIO_THRESHOLD,
    top_n: int = TOP_N_BROKER_ACCUMULATION,
) -> list[dict]:
    """
    Flag tickers where buy volume / sell volume by top brokers exceeds
    `ratio_threshold` (i.e. brokers are net accumulating that name).
    """
    flagged = []
    for row in rows:
        ticker = _pick(row, "symbol", "ticker", "code", default="UNKNOWN")
        buy_vol = _to_float(_pick(row, "buy_volume", "total_buy", "buy", default=0))
        sell_vol = _to_float(_pick(row, "sell_volume", "total_sell", "sell", default=0))
        broker = _pick(row, "broker", "broker_name", "top_broker", default="N/A")

        if sell_vol <= 0:
            continue
        ratio = buy_vol / sell_vol
        if ratio >= ratio_threshold:
            flagged.append(
                {
                    "ticker": ticker,
                    "ratio": round(ratio, 2),
                    "top_broker": broker,
                }
            )

    flagged.sort(key=lambda r: r["ratio"], reverse=True)
    return flagged[:top_n]


def detect_volume_spikes(
    rows: list[dict],
    multiple_threshold: float = VOLUME_SPIKE_MULTIPLE_THRESHOLD,
    top_n: int = TOP_N_VOLUME_SPIKES,
) -> list[dict]:
    """Flag tickers whose trading volume today exceeds `multiple_threshold`x their average."""
    spikes = []
    for row in rows:
        ticker = _pick(row, "symbol", "ticker", "code", default="UNKNOWN")
        volume = _to_float(_pick(row, "volume", "today_volume", default=0))
        avg_volume = _to_float(
            _pick(row, "avg_volume", "average_volume", "volume_avg_30d", default=0)
        )
        if avg_volume <= 0:
            continue
        multiple = volume / avg_volume
        if multiple >= multiple_threshold:
            spikes.append({"ticker": ticker, "volume_multiple": round(multiple, 2)})

    spikes.sort(key=lambda r: r["volume_multiple"], reverse=True)
    return spikes[:top_n]


def build_anomalies(
    foreign_flow_rows: list[dict],
    broker_rows: list[dict],
    activity_rows: list[dict],
) -> Anomalies:
    top_buy, top_sell = detect_foreign_flow_anomalies(foreign_flow_rows)
    return Anomalies(
        top_foreign_buy=top_buy,
        top_foreign_sell=top_sell,
        broker_accumulation=detect_broker_accumulation(broker_rows),
        volume_spikes=detect_volume_spikes(activity_rows),
    )


def anomalies_to_minimal_dict(anomalies: Anomalies) -> dict:
    """
    Collapse the anomalies into a tiny dict for the LLM prompt.
    Target: well under 100 input tokens.
    """
    return {
        "foreign_buy": [
            f"{r['ticker']}:+{r['net_foreign_flow']:.0f}" for r in anomalies.top_foreign_buy
        ],
        "foreign_sell": [
            f"{r['ticker']}:{r['net_foreign_flow']:.0f}" for r in anomalies.top_foreign_sell
        ],
        "broker_accum": [
            f"{r['ticker']}:{r['ratio']}x" for r in anomalies.broker_accumulation
        ],
        "vol_spike": [
            f"{r['ticker']}:{r['volume_multiple']}x" for r in anomalies.volume_spikes
        ],
    }


# --------------------------------------------------------------------------
# Step 3: Low-token LLM summary
# --------------------------------------------------------------------------

def generate_executive_summary(minimal_dict: dict, api_key: str) -> str:
    """
    Send only the tiny filtered dict to the LLM and return a 2-sentence
    executive summary. Falls back to a templated summary if the LLM call
    fails, so the pipeline never breaks on an LLM outage.
    """
    if not any(minimal_dict.values()):
        return "No significant IDX anomalies were detected in today's session."

    user_prompt = f"IDX anomalies today: {json.dumps(minimal_dict, separators=(',', ':'))}"

    try:
        if LLM_PROVIDER == "gemini":
            return _call_gemini(user_prompt, api_key)
        return _call_openai(user_prompt, api_key)
    except Exception as exc:  # noqa: BLE001 - we want a graceful fallback, not a crash
        logger.warning("LLM call failed, falling back to templated summary: %s", exc)
        return _fallback_summary(minimal_dict)


def _call_openai(user_prompt: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 120,
            "temperature": 0.4,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_gemini(user_prompt: str, api_key: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": LLM_SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": 120, "temperature": 0.4},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _fallback_summary(minimal_dict: dict) -> str:
    """Deterministic backup summary used if the LLM call fails entirely."""
    parts = []
    if minimal_dict.get("foreign_buy"):
        parts.append(f"Foreign inflows concentrated in {', '.join(minimal_dict['foreign_buy'])}.")
    if minimal_dict.get("broker_accum"):
        parts.append(f"Notable broker accumulation seen in {', '.join(minimal_dict['broker_accum'])}.")
    if not parts:
        parts.append("Market activity was within normal ranges today.")
    return " ".join(parts[:2])


# --------------------------------------------------------------------------
# Step 4: Discord rich embed formatting + delivery
# --------------------------------------------------------------------------

DISCORD_EMBED_COLOR = 0xE30022  # IDX red
DISCORD_CONTENT_LIMIT = 1024  # Discord embed field value character limit


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data available._"
    lines = [f"`{' | '.join(headers)}`"]
    for row in rows:
        lines.append(f"`{' | '.join(row)}`")
    table = "\n".join(lines)
    return table[:DISCORD_CONTENT_LIMIT]


def build_discord_embed(anomalies: Anomalies, summary: str, execution_ok: bool) -> dict:
    foreign_rows = [
        [r["ticker"], f"{r['net_foreign_flow']:+.0f}"] for r in anomalies.top_foreign_buy
    ] + [[r["ticker"], f"{r['net_foreign_flow']:+.0f}"] for r in anomalies.top_foreign_sell]

    broker_rows = [
        [r["ticker"], f"{r['ratio']}x", r["top_broker"]] for r in anomalies.broker_accumulation
    ]

    volume_rows = [[r["ticker"], f"{r['volume_multiple']}x"] for r in anomalies.volume_spikes]

    fields = [
        {
            "name": "🧠 AI Executive Summary",
            "value": summary[:DISCORD_CONTENT_LIMIT],
            "inline": False,
        },
        {
            "name": "💱 Foreign Flow (Top Buy/Sell, IDR mn)",
            "value": _markdown_table(["Ticker", "Net Flow"], foreign_rows),
            "inline": False,
        },
        {
            "name": "🏦 Broker Accumulation",
            "value": _markdown_table(["Ticker", "Ratio", "Top Broker"], broker_rows),
            "inline": False,
        },
    ]

    if volume_rows:
        fields.append(
            {
                "name": "📈 Volume Spikes",
                "value": _markdown_table(["Ticker", "x Avg Volume"], volume_rows),
                "inline": False,
            }
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "✅ OK" if execution_ok else "⚠️ Partial data (see logs)"

    return {
        "embeds": [
            {
                "title": "🇮🇩 IDX Daily Market Watchdog Briefing",
                "color": DISCORD_EMBED_COLOR,
                "fields": fields,
                "footer": {"text": f"{now} · Execution status: {status}"},
            }
        ]
    }


def send_discord_alert(payload: dict, webhook_url: str) -> None:
    try:
        resp = requests.post(webhook_url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to reach Discord webhook: {exc}") from exc

    # Discord webhooks return 204 No Content on success.
    if resp.status_code not in (200, 204):
        raise RuntimeError(
            f"Discord webhook returned HTTP {resp.status_code}: {resp.text[:300]}"
        )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run(settings: Settings) -> int:
    logger.info("Starting IDX watchdog run...")

    foreign_flow_rows = fetch_daily_foreign_flow(settings.sectors_api_key)
    broker_rows = fetch_top_brokers(settings.sectors_api_key)
    activity_rows = fetch_broker_activity(settings.sectors_api_key)

    execution_ok = bool(foreign_flow_rows or broker_rows or activity_rows)
    if not execution_ok:
        logger.error("All Sectors API fetches failed or returned empty data.")

    if settings.debug:
        logger.info(
            "[DEBUG] raw sample: foreign_flow=%s broker=%s activity=%s",
            foreign_flow_rows[:1],
            broker_rows[:1],
            activity_rows[:1],
        )

    anomalies = build_anomalies(foreign_flow_rows, broker_rows, activity_rows)
    minimal_dict = anomalies_to_minimal_dict(anomalies)
    logger.info("Minimal anomaly payload for LLM: %s", minimal_dict)

    summary = generate_executive_summary(minimal_dict, settings.llm_api_key)
    logger.info("Executive summary: %s", summary)

    embed_payload = build_discord_embed(anomalies, summary, execution_ok)

    try:
        send_discord_alert(embed_payload, settings.discord_webhook_url)
    except RuntimeError as exc:
        logger.error("Discord delivery failed: %s", exc)
        return 1

    logger.info("Discord alert sent successfully.")
    return 0 if execution_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="IDX Smart Alerts & Automated Watchdog")
    parser.add_argument(
        "--debug", action="store_true", help="Log raw API sample rows for schema debugging."
    )
    args = parser.parse_args()

    if args.debug:
        os.environ["WATCHDOG_DEBUG"] = "true"

    try:
        settings = load_settings()
    except EnvironmentError as exc:
        logger.error(str(exc))
        sys.exit(2)

    exit_code = run(settings)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
