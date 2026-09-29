#!/usr/bin/env python3
"""Vex-Watcher: token-frugal IDX daily market watchdog.

Pipeline:
  1. Fetch raw data from the Sectors API (foreign flow, top brokers, broker activity).
  2. Filter deterministically in Python (0 LLM tokens) down to a handful of tickers.
  3. Send ONLY a tiny dict to an LLM for a 2-sentence strategic summary.
  4. Post a Rich Embed to Discord as "Vex-Watcher".

Required env vars (GitHub Secrets):
  SECTORS_API_KEY, LLM_API_KEY, DISCORD_WEBHOOK_URL
Optional env vars:
  AVATAR_URL    - public URL of the avatar image (defaults to image.png in this repo's raw URL)
  LLM_PROVIDER  - "openai" or "gemini" (auto-detected from key prefix if unset)
  LLM_MODEL     - override the default model name
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import requests
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:  # .env is only for local runs; in GitHub Actions the env is already set.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SECTORS_BASE_URL = "https://api.sectors.app"
# Endpoints requested in the spec. Adjust paths/params here if your plan differs.
ENDPOINTS: dict[str, tuple[str, dict[str, Any]]] = {
    "foreign_flow": ("/v2/daily-foreign-flow/", {}),
    "top_brokers": ("/v2/top-brokers/", {}),
    "broker_activity": ("/v2/broker-activity/", {}),
}

TOP_N = 5                 # tickers shown per list in Discord
LLM_TOP_N = 3             # tickers passed to the LLM (keeps input < 100 tokens)
MIN_ACCUM_RATIO = 2.0     # buy_value / sell_value threshold
MIN_VOLUME_SPIKE = 1.5    # volume / avg_volume threshold
HTTP_TIMEOUT = (5, 20)    # (connect, read) seconds

SYSTEM_PROMPT = (
    "You are a concise financial market analyst. Summarize the provided IDX market "
    "anomalies in exactly 2 executive sentences. Do not repeat numbers given in the "
    "prompt verbatim; give strategic market context."
)

WIB = timezone(timedelta(hours=7))
BOT_NAME = "Vex-Watcher"

# The Sectors response schema for these endpoints can vary, so field lookups accept
# several candidate names. Add your real field names here if needed.
TICKER_KEYS = ("symbol", "ticker", "stock", "code", "stock_code")
FOREIGN_NET_KEYS = ("net_foreign", "net_foreign_value", "foreign_net", "net_foreign_flow", "foreign_net_value")
FOREIGN_BUY_KEYS = ("foreign_buy", "foreign_buy_value")
FOREIGN_SELL_KEYS = ("foreign_sell", "foreign_sell_value")
BROKER_KEYS = ("broker_code", "broker", "code_broker", "broker_id")
BUY_KEYS = ("buy_value", "buy", "b_val", "total_buy_value", "buy_amount")
SELL_KEYS = ("sell_value", "sell", "s_val", "total_sell_value", "sell_amount")
NET_KEYS = ("net_value", "net", "net_buy", "net_amount")
VOLUME_KEYS = ("volume", "total_volume")
AVG_VOLUME_KEYS = ("avg_volume", "average_volume", "volume_avg", "avg_volume_20d")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vex-watcher")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class Settings(BaseModel):
    """Runtime settings read from the environment (never logged)."""

    sectors_api_key: str
    llm_api_key: str
    discord_webhook_url: str
    avatar_url: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        required = ("SECTORS_API_KEY", "LLM_API_KEY", "DISCORD_WEBHOOK_URL")
        missing = [k for k in required if not os.getenv(k, "").strip()]
        if missing:
            raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

        avatar = os.getenv("AVATAR_URL", "").strip() or None
        repo = os.getenv("GITHUB_REPOSITORY")
        if not avatar and repo:  # works only if the repo is public
            branch = os.getenv("GITHUB_REF_NAME", "main")
            avatar = f"https://raw.githubusercontent.com/{repo}/{branch}/image.png"

        return cls(
            sectors_api_key=os.environ["SECTORS_API_KEY"].strip(),
            llm_api_key=os.environ["LLM_API_KEY"].strip(),
            discord_webhook_url=os.environ["DISCORD_WEBHOOK_URL"].strip(),
            avatar_url=avatar,
            llm_provider=(os.getenv("LLM_PROVIDER", "").strip().lower() or None),
            llm_model=os.getenv("LLM_MODEL", "").strip() or None,
        )


class ForeignFlow(BaseModel):
    ticker: str
    net_value: float  # IDR; positive = net buy, negative = net sell


class Accumulation(BaseModel):
    ticker: str
    ratio: float                     # buy_value / sell_value (capped at 99)
    volume_spike: Optional[float] = None
    buyers: list[str] = Field(default_factory=list)  # top net-buying broker codes


class Findings(BaseModel):
    """The minimal, deterministic result of all filtering."""

    foreign_buys: list[ForeignFlow] = Field(default_factory=list)
    foreign_sells: list[ForeignFlow] = Field(default_factory=list)
    accumulations: list[Accumulation] = Field(default_factory=list)
    top_brokers: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Step 1: Fetch market data
# --------------------------------------------------------------------------- #
def build_session() -> requests.Session:
    """Session with automatic retries for transient errors (429/5xx)."""
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    """Normalise a JSON payload to a list of dict rows."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("results", "data", "items", "records"):
            inner = payload.get(key)
            if isinstance(inner, (list, dict)):
                return _extract_rows(inner)
    return []


def fetch_rows(session: requests.Session, api_key: str, name: str) -> Optional[list[dict[str, Any]]]:
    """GET one Sectors endpoint. Returns None on any failure (never raises)."""
    path, params = ENDPOINTS[name]
    try:
        resp = session.get(
            f"{SECTORS_BASE_URL}{path}",
            headers={"Authorization": api_key},
            params=params,
            timeout=HTTP_TIMEOUT,
        )
    except requests.Timeout:
        log.error("[%s] request timed out", name)
        return None
    except requests.RequestException as exc:
        log.error("[%s] request failed: %s", name, type(exc).__name__)
        return None

    if resp.status_code != 200:
        log.error("[%s] HTTP %s: %s", name, resp.status_code, resp.text[:200])
        return None
    try:
        return _extract_rows(resp.json())
    except ValueError:
        log.error("[%s] response was not valid JSON", name)
        return None


# --------------------------------------------------------------------------- #
# Step 2: Deterministic filtering (0 LLM tokens)
# --------------------------------------------------------------------------- #
def _num(row: dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _txt(row: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    return None


def _ticker(row: dict[str, Any]) -> Optional[str]:
    t = _txt(row, TICKER_KEYS)
    return t[:-3] if t and t.endswith(".JK") else t


def top_foreign_flows(rows: list[dict[str, Any]], n: int = TOP_N) -> tuple[list[ForeignFlow], list[ForeignFlow]]:
    """Top-n net foreign buys and sells (rows for the same ticker are summed)."""
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        ticker = _ticker(row)
        net = _num(row, FOREIGN_NET_KEYS)
        if net is None:
            buy, sell = _num(row, FOREIGN_BUY_KEYS), _num(row, FOREIGN_SELL_KEYS)
            net = buy - sell if buy is not None and sell is not None else None
        if ticker and net is not None:
            totals[ticker] += net

    flows = [ForeignFlow(ticker=t, net_value=v) for t, v in totals.items()]
    buys = sorted((f for f in flows if f.net_value > 0), key=lambda f: f.net_value, reverse=True)[:n]
    sells = sorted((f for f in flows if f.net_value < 0), key=lambda f: f.net_value)[:n]
    return buys, sells


def find_accumulations(rows: list[dict[str, Any]], n: int = TOP_N) -> list[Accumulation]:
    """Tickers with accumulation ratio > 2.0x OR volume spike > 1.5x, plus key buyer brokers."""
    agg: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"buy": 0.0, "sell": 0.0, "vol": None, "avg": None, "brokers": defaultdict(float)}
    )
    for row in rows:
        ticker = _ticker(row)
        if not ticker:
            continue
        a = agg[ticker]
        buy = _num(row, BUY_KEYS) or 0.0
        sell = _num(row, SELL_KEYS) or 0.0
        net = _num(row, NET_KEYS)
        a["buy"] += buy
        a["sell"] += sell
        # Volume fields are per-ticker (repeated on each broker row), so take the max.
        for slot, keys in (("vol", VOLUME_KEYS), ("avg", AVG_VOLUME_KEYS)):
            v = _num(row, keys)
            if v is not None:
                a[slot] = max(a[slot] or 0.0, v)
        code = _txt(row, BROKER_KEYS)
        if code:
            a["brokers"][code] += net if net is not None else buy - sell

    results: list[tuple[float, Accumulation]] = []
    for ticker, a in agg.items():
        ratio = a["buy"] / a["sell"] if a["sell"] > 0 else (99.0 if a["buy"] > 0 else 0.0)
        ratio = min(ratio, 99.0)
        spike = a["vol"] / a["avg"] if a["vol"] and a["avg"] else None
        if not (ratio > MIN_ACCUM_RATIO or (spike is not None and spike > MIN_VOLUME_SPIKE)):
            continue
        buyers = [c for c, v in sorted(a["brokers"].items(), key=lambda kv: kv[1], reverse=True) if v > 0][:3]
        score = max(ratio / MIN_ACCUM_RATIO, (spike or 0.0) / MIN_VOLUME_SPIKE)
        results.append((score, Accumulation(ticker=ticker, ratio=ratio, volume_spike=spike, buyers=buyers)))

    results.sort(key=lambda x: x[0], reverse=True)
    return [acc for _, acc in results[:n]]


def top_broker_codes(rows: list[dict[str, Any]], n: int = TOP_N) -> list[str]:
    """Broker codes ranked by net value (falls back to buy value, then API order)."""
    scored: list[tuple[float, str]] = []
    for i, row in enumerate(rows):
        code = _txt(row, BROKER_KEYS)
        if not code:
            continue
        score = _num(row, NET_KEYS)
        if score is None:
            score = _num(row, BUY_KEYS)
        scored.append((score if score is not None else -float(i), code))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen: list[str] = []
    for _, code in scored:
        if code not in seen:
            seen.append(code)
    return seen[:n]


def llm_payload(f: Findings) -> dict[str, Any]:
    """Tiny dict for the LLM (~60-80 tokens). Tickers/brokers only, ratios rounded."""
    return {
        "foreign_buy": [x.ticker for x in f.foreign_buys[:LLM_TOP_N]],
        "foreign_sell": [x.ticker for x in f.foreign_sells[:LLM_TOP_N]],
        "accumulation": [
            {"t": a.ticker, "x": round(a.ratio, 1), "b": a.buyers[:2]} for a in f.accumulations[:LLM_TOP_N]
        ],
    }


# --------------------------------------------------------------------------- #
# Step 3: Low-token LLM summary
# --------------------------------------------------------------------------- #
def _enforce_two_sentences(text: str) -> Optional[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return " ".join(sentences[:2]) if len(sentences) >= 2 else None


def summarize(settings: Settings, payload: dict[str, Any]) -> Optional[str]:
    """Call OpenAI or Gemini. Returns a 2-sentence string, or None on any failure."""
    user_prompt = json.dumps(payload, separators=(",", ":"))
    provider = settings.llm_provider or ("gemini" if settings.llm_api_key.startswith("AIza") else "openai")
    try:
        if provider == "gemini":
            model = settings.llm_model or "gemini-2.5-flash-lite"
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": settings.llm_api_key},
                json={
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": {"maxOutputTokens": 80, "temperature": 0.4},
                },
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            model = settings.llm_model or "gpt-4o-mini"
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 60,
                    "temperature": 0.4,
                },
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        log.error("LLM call failed (%s): %s", provider, type(exc).__name__)
        return None

    summary = _enforce_two_sentences(text)
    if summary is None:
        log.error("LLM returned fewer than 2 sentences")
    return summary


# --------------------------------------------------------------------------- #
# Step 4: Discord formatting + delivery
# --------------------------------------------------------------------------- #
def fmt_idr(value: float) -> str:
    """Compact signed IDR, e.g. +1.20T, -310.5B."""
    a = abs(value)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"{value / div:+.2f}{suffix}"
    return f"{value:+,.0f}"


def _clip(text: str, limit: int = 1024) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _foreign_table(f: Findings) -> str:
    rows = max(len(f.foreign_buys), len(f.foreign_sells))
    if rows == 0:
        return "No foreign flow data available."
    lines = [f"{'NET BUY':<17}NET SELL"]
    for i in range(rows):
        left = f"{f.foreign_buys[i].ticker:<6}{fmt_idr(f.foreign_buys[i].net_value):>9}" if i < len(f.foreign_buys) else ""
        right = f"{f.foreign_sells[i].ticker:<6}{fmt_idr(f.foreign_sells[i].net_value):>9}" if i < len(f.foreign_sells) else ""
        lines.append(f"{left:<17}{right}")
    return "```\n" + "\n".join(lines) + "\n```"


def _accumulation_text(f: Findings) -> str:
    if not f.accumulations:
        text = "No tickers crossed the accumulation or volume-spike thresholds."
    else:
        items = []
        for a in f.accumulations:
            parts = [f"{a.ratio:.1f}x accumulation"] if a.ratio > MIN_ACCUM_RATIO else []
            if a.volume_spike and a.volume_spike > MIN_VOLUME_SPIKE:
                parts.append(f"{a.volume_spike:.1f}x volume")
            buyers = ", ".join(f"`{b}`" for b in a.buyers) or "n/a"
            items.append(f"• **{a.ticker}**: {' · '.join(parts)} · buyers: {buyers}")
        text = "\n".join(items)
    if f.top_brokers:
        text += "\n\n**Top brokers:** " + ", ".join(f"`{b}`" for b in f.top_brokers)
    return text


def build_payload(settings: Settings, f: Findings, summary: Optional[str], status: str, now: datetime) -> dict[str, Any]:
    color = {"OK": 0x2ECC71, "FAILED": 0xE74C3C}.get(status.split(" ")[0], 0xF39C12)
    ai_text = summary or "AI summary is unavailable for this run. Refer to the flow and accumulation data below."
    footer = (
        f"{now.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC / "
        f"{now.astimezone(WIB):%H:%M} WIB • Pipeline: {status}"
    )
    payload: dict[str, Any] = {
        "username": BOT_NAME,
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": "🇮🇩 IDX Daily Market Watchdog Briefing",
                "color": color,
                "fields": [
                    {"name": "🧠 AI Executive Summary", "value": _clip(ai_text), "inline": False},
                    {"name": "🌏 Foreign Flow", "value": _clip(_foreign_table(f)), "inline": False},
                    {"name": "🏦 Broker Accumulation", "value": _clip(_accumulation_text(f)), "inline": False},
                ],
                "footer": {"text": footer},
                "timestamp": now.astimezone(timezone.utc).isoformat(),
            }
        ],
    }
    if settings.avatar_url:
        payload["avatar_url"] = settings.avatar_url
    return payload


def post_discord(webhook_url: str, payload: dict[str, Any]) -> bool:
    """POST to the webhook, honouring one 429 retry_after. Returns success."""
    for attempt in range(2):
        try:
            resp = requests.post(webhook_url, json=payload, timeout=HTTP_TIMEOUT)
        except requests.RequestException as exc:
            log.error("Discord post failed: %s", type(exc).__name__)
            return False
        if resp.status_code in (200, 204):
            return True
        if resp.status_code == 429 and attempt == 0:
            try:
                wait = float(resp.json().get("retry_after", 1))
            except ValueError:
                wait = 1.0
            time.sleep(min(wait, 10))
            continue
        log.error("Discord returned HTTP %s: %s", resp.status_code, resp.text[:200])
        return False
    return False


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> int:
    settings = Settings.from_env()
    now = datetime.now(timezone.utc)
    session = build_session()

    raw = {name: fetch_rows(session, settings.sectors_api_key, name) for name in ENDPOINTS}
    failed = [name for name, rows in raw.items() if rows is None]

    findings = Findings()
    if raw["foreign_flow"]:
        findings.foreign_buys, findings.foreign_sells = top_foreign_flows(raw["foreign_flow"])
    if raw["broker_activity"]:
        findings.accumulations = find_accumulations(raw["broker_activity"])
    if raw["top_brokers"]:
        findings.top_brokers = top_broker_codes(raw["top_brokers"])
    log.info(
        "Findings: %d buys, %d sells, %d accumulations",
        len(findings.foreign_buys), len(findings.foreign_sells), len(findings.accumulations),
    )

    has_data = bool(findings.foreign_buys or findings.foreign_sells or findings.accumulations)
    summary = summarize(settings, llm_payload(findings)) if has_data else None
    if has_data and summary is None:
        failed.append("llm")

    if len(failed) >= len(ENDPOINTS):
        status = "FAILED (" + ", ".join(failed) + ")"
    elif failed:
        status = "DEGRADED (" + ", ".join(failed) + ")"
    elif not has_data:
        status = "OK (no anomalies; market may be closed)"
    else:
        status = "OK"

    sent = post_discord(settings.discord_webhook_url, build_payload(settings, findings, summary, status, now))
    log.info("Pipeline status: %s | Discord delivered: %s", status, sent)
    return 0 if sent and not status.startswith("FAILED") else 1


if __name__ == "__main__":
    sys.exit(main())
