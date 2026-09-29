#!/usr/bin/env python3
"""Vex-Watcher: token-frugal IDX daily market watchdog.

Pipeline:
  1. Fetch raw data from the Sectors API v2 (/v2/foreign-flow/, /v2/brokers/top/,
     /v2/broker-summary/{symbol}/top/, /v2/daily/{symbol}/).
  2. Filter deterministically in Python (0 LLM tokens) down to a handful of tickers.
  3. Send ONLY a tiny dict to an LLM for a 2-sentence strategic summary.
  4. Post a Rich Embed to Discord as "Vex-Watcher".

Required env vars (GitHub Secrets):
  SECTORS_API_KEY, LLM_API_KEY, DISCORD_WEBHOOK_URL
Optional env vars:
  AVATAR_URL    - public URL of the avatar image (defaults to image.png in this repo's raw URL)
  LLM_PROVIDER  - "openai" or "gemini" (keys starting "sk-" => openai, anything else => gemini)
  LLM_MODEL     - override the default model name
"""
from __future__ import annotations

import argparse
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

TOP_N = 5                 # tickers shown per list in Discord
LLM_TOP_N = 3             # tickers passed to the LLM (keeps input < 100 tokens)
MAX_CANDIDATES = 5        # top foreign-buy tickers deep-checked for accumulation
BROKER_TOP_N = 5          # top buyers / sellers per ticker used for the ratio
VOLUME_LOOKBACK_DAYS = 30 # window for the average-volume baseline
MIN_BASELINE_DAYS = 5     # minimum prior trading days needed to trust the baseline
MIN_ACCUM_RATIO = 2.0     # top-buyer net inflow / top-seller net outflow
MIN_VOLUME_SPIKE = 1.5    # latest volume / average prior volume
HTTP_TIMEOUT = (5, 20)    # (connect, read) seconds

SYSTEM_PROMPT = (
    "You are a concise financial market analyst. Summarize the provided IDX market "
    "anomalies in exactly 2 executive sentences. Do not repeat numbers given in the "
    "prompt verbatim; give strategic market context."
)

WIB = timezone(timedelta(hours=7))
BOT_NAME = "Vex-Watcher"

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


class BrokerFlow(BaseModel):
    code: str
    net_value: float  # IDR, signed


class Accumulation(BaseModel):
    ticker: str
    ratio: float                     # top-buyer net inflow / top-seller net outflow (capped at 99)
    volume_spike: Optional[float] = None  # latest volume / average prior volume
    buyers: list[str] = Field(default_factory=list)  # top net-buying broker codes


class Findings(BaseModel):
    """The minimal, deterministic result of all filtering."""

    foreign_buys: list[ForeignFlow] = Field(default_factory=list)
    foreign_sells: list[ForeignFlow] = Field(default_factory=list)
    accumulations: list[Accumulation] = Field(default_factory=list)
    top_brokers: list[BrokerFlow] = Field(default_factory=list)
    trade_date: Optional[str] = None


# --------------------------------------------------------------------------- #
# Step 1: Fetch market data (Sectors API v2)
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


def get_json(
    session: requests.Session, api_key: str, path: str, params: dict[str, Any], label: str
) -> Optional[Any]:
    """GET a Sectors endpoint and return parsed JSON, or None on any failure (never raises)."""
    try:
        resp = session.get(
            f"{SECTORS_BASE_URL}{path}",
            headers={"Authorization": api_key},
            params=params,
            timeout=HTTP_TIMEOUT,
        )
    except requests.Timeout:
        log.error("[%s] request timed out", label)
        return None
    except requests.RequestException as exc:
        log.error("[%s] request failed: %s", label, type(exc).__name__)
        return None

    if resp.status_code != 200:
        log.error("[%s] HTTP %s: %s", label, resp.status_code, resp.text[:200])
        return None
    try:
        return resp.json()
    except ValueError:
        log.error("[%s] response was not valid JSON", label)
        return None


def _rows(payload: Any) -> list[dict[str, Any]]:
    """Rows from either a bare list or a {"results": [...]} envelope."""
    if isinstance(payload, dict):
        payload = payload.get("results", [])
    return [r for r in payload if isinstance(r, dict)] if isinstance(payload, list) else []


def _bare(symbol: str) -> str:
    """'ANTM.JK' -> 'ANTM'."""
    s = symbol.strip().upper()
    return s[:-3] if s.endswith(".JK") else s


def _f(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Step 2: Deterministic filtering (0 LLM tokens)
# --------------------------------------------------------------------------- #
def parse_foreign_flow(payload: Any, want_buys: bool) -> tuple[list[ForeignFlow], Optional[str]]:
    """Parse /v2/foreign-flow/ rows. Returns (flows, trading_date)."""
    flows: list[ForeignFlow] = []
    trade_date: Optional[str] = None
    for row in _rows(payload):
        symbol, net = row.get("symbol"), _f(row.get("net_foreign_inflow"))
        if not isinstance(symbol, str) or net is None:
            continue
        trade_date = trade_date or row.get("date")
        if (net > 0) == want_buys and net != 0:
            flows.append(ForeignFlow(ticker=_bare(symbol), net_value=net))
    flows.sort(key=lambda f: f.net_value, reverse=want_buys)
    return flows[:TOP_N], trade_date


def parse_top_brokers(payload: Any, n: int = TOP_N) -> list[BrokerFlow]:
    """Top net-buying brokers from /v2/brokers/top/ (ranked by signed net value)."""
    brokers = [
        BrokerFlow(code=str(r["broker_code"]).upper(), net_value=net)
        for r in _rows(payload)
        if r.get("broker_code") and (net := _f(r.get("net"))) is not None and net > 0
    ]
    brokers.sort(key=lambda b: b.net_value, reverse=True)
    return brokers[:n]


def volume_spike(daily_payload: Any, trade_date: Optional[str]) -> Optional[float]:
    """Latest volume / mean of prior volumes from /v2/daily/{symbol}/ (None if baseline too thin)."""
    rows = sorted(
        (r for r in _rows(daily_payload) if r.get("date") and _f(r.get("volume")) is not None),
        key=lambda r: r["date"],
    )
    if trade_date:
        rows = [r for r in rows if r["date"] <= trade_date]
    if len(rows) < MIN_BASELINE_DAYS + 1:
        return None
    latest = float(rows[-1]["volume"])
    prior = [float(r["volume"]) for r in rows[:-1] if float(r["volume"]) > 0]
    if len(prior) < MIN_BASELINE_DAYS:
        return None
    return latest / (sum(prior) / len(prior))


def evaluate_accumulation(symbol: str, summary_payload: Any, daily_payload: Any, trade_date: Optional[str]) -> Optional[Accumulation]:
    """Build an Accumulation if the ticker crosses either threshold, else None.

    Accumulation ratio = net IDR bought by the top buyers / net IDR sold by the top sellers
    (from /v2/broker-summary/{symbol}/top/). > 2.0 means buying pressure clearly dominates.
    """
    if not isinstance(summary_payload, dict):
        return None
    buyers = [b for b in summary_payload.get("top_buyers") or [] if isinstance(b, dict)]
    sellers = [s for s in summary_payload.get("top_sellers") or [] if isinstance(s, dict)]
    buy_net = sum(max(_f(b.get("net_idr")) or 0.0, 0.0) for b in buyers)
    sell_net = abs(sum(min(_f(s.get("net_idr")) or 0.0, 0.0) for s in sellers))
    ratio = min(buy_net / sell_net, 99.0) if sell_net > 0 else (99.0 if buy_net > 0 else 0.0)
    spike = volume_spike(daily_payload, trade_date)

    if not (ratio > MIN_ACCUM_RATIO or (spike is not None and spike > MIN_VOLUME_SPIKE)):
        return None
    codes = [str(b["broker_code"]).upper() for b in buyers if b.get("broker_code") and (_f(b.get("net_idr")) or 0) > 0]
    return Accumulation(ticker=_bare(symbol), ratio=ratio, volume_spike=spike, buyers=codes[:3])


def rank_accumulations(items: list[Accumulation], n: int = TOP_N) -> list[Accumulation]:
    """Strongest signal first, whichever threshold it exceeded by the larger margin."""
    return sorted(
        items,
        key=lambda a: max(a.ratio / MIN_ACCUM_RATIO, (a.volume_spike or 0.0) / MIN_VOLUME_SPIKE),
        reverse=True,
    )[:n]


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


GEMINI_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",  # retired for new users; kept for older accounts
)


def _call_llm(settings: Settings, provider: str, model: str, user_prompt: str) -> str:
    """One LLM request. Raises requests.HTTPError / KeyError / IndexError / ValueError on failure."""
    if provider == "gemini":
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": settings.llm_api_key},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.4},
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        cand = resp.json()["candidates"][0]
        if cand.get("finishReason") == "MAX_TOKENS":
            raise ValueError("output truncated (finishReason=MAX_TOKENS)")
        return "".join(p.get("text", "") for p in cand["content"]["parts"])
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": 200,
            "temperature": 0.4,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def summarize(settings: Settings, payload: dict[str, Any]) -> Optional[str]:
    """Return a 2-sentence summary, or None if every candidate model fails.

    Gemini free tier: model availability and quotas change, so unless LLM_MODEL is set we try
    several Flash-tier models in order. Temporary 503/429 responses get one short retry.
    """
    user_prompt = json.dumps(payload, separators=(",", ":"))
    # Only "sk-" keys are treated as OpenAI; everything else (AIza..., AQ...) is Gemini.
    # Set LLM_PROVIDER to override (e.g. for OpenAI-compatible keys with other prefixes).
    provider = settings.llm_provider or ("openai" if settings.llm_api_key.startswith("sk-") else "gemini")
    if settings.llm_model:
        models: tuple[str, ...] = (settings.llm_model,)
    else:
        models = GEMINI_FALLBACK_MODELS if provider == "gemini" else ("gpt-4o-mini",)

    for model in models:
        for attempt in range(2):
            try:
                text = _call_llm(settings, provider, model, user_prompt)
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else 0
                body = exc.response.text[:300] if exc.response is not None else ""
                log.error("LLM call failed (%s, model=%s): HTTP %s: %s", provider, model, code, body)
                if code in (429, 503) and attempt == 0:
                    time.sleep(5)  # transient overload / per-minute limit: retry once
                    continue
                break  # 404 / 400 / 401 etc.: try the next model
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                log.error("LLM response unusable (%s, model=%s): %s %s", provider, model, type(exc).__name__, exc)
                break

            summary = _enforce_two_sentences(text)
            if summary:
                log.info("LLM summary produced by %s", model)
                return summary
            log.error("LLM returned fewer than 2 sentences (model=%s): %r", model, text[:200])
            break
    return None


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
        text += "\n\n**Top net-buying brokers:** " + ", ".join(
            f"`{b.code}` {fmt_idr(b.net_value)}" for b in f.top_brokers
        )
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
# --------------------------------------------------------------------------- #
# Offline mock mode (--mock): zero Sectors credits, same response shapes as the real API
# --------------------------------------------------------------------------- #
_MOCK_DATE = "2026-09-28"
_MOCK_BUYS = [("BBCA", 1.2e12), ("BMRI", 6.4e11), ("ANTM", 3.0e11), ("TLKM", 1.9e11), ("ASII", 1.1e11)]
_MOCK_SELLS = [("GOTO", -3.1e11), ("ADRO", -2.2e11), ("UNVR", -1.4e11), ("ICBP", -9.0e10), ("BUKA", -6.0e10)]
# symbol -> (top-buyer net IDR, top-seller net IDR, latest volume multiple of baseline)
_MOCK_SIGNALS = {"ANTM": (4.0e11, 1.0e11, 2.5), "BBCA": (6.0e11, 5.0e11, 1.1), "BMRI": (2.0e11, 1.5e11, 1.6)}


def mock_get_json(session: Any, api_key: str, path: str, params: dict[str, Any], label: str) -> Optional[Any]:
    """Drop-in replacement for get_json() returning fabricated, schema-accurate data."""
    log.info("[mock] %s", label)
    if path == "/v2/foreign-flow/":
        picks = _MOCK_SELLS if params.get("order_by") else _MOCK_BUYS
        rows = [
            {"symbol": f"{t}.JK", "date": _MOCK_DATE, "net_foreign_inflow": int(v),
             "foreign_buy_idr": int(abs(v) * 2), "foreign_sell_idr": int(abs(v))}
            for t, v in picks
        ]
        return {"results": rows, "pagination": {"has_next": False}}
    if path == "/v2/brokers/top/":
        codes = [("AK", -3.0e11), ("YP", 5.0e11), ("PD", 2.4e11), ("CC", 1.1e11), ("KZ", 9.0e10), ("BK", -2.0e11)]
        return {"date": _MOCK_DATE, "results": [
            {"rank": i + 1, "broker_code": c, "gross": 1_000_000_000_000, "net": int(n),
             "foreign_gross": 0, "foreign_net": 0} for i, (c, n) in enumerate(codes)]}
    symbol = path.split("/")[3]
    buy, sell, mult = _MOCK_SIGNALS.get(symbol, (1.0e11, 1.0e11, 1.0))
    if "/broker-summary/" in path:
        return {
            "symbol": f"{symbol}.JK",
            "top_buyers": [
                {"rank": 1, "broker_code": "YP", "net_idr": int(buy), "buy_idr": 0, "sell_idr": 0},
                {"rank": 2, "broker_code": "PD", "net_idr": int(buy / 2), "buy_idr": 0, "sell_idr": 0},
            ],
            "top_sellers": [{"rank": 1, "broker_code": "BK", "net_idr": -int(sell), "buy_idr": 0, "sell_idr": 0}],
        }
    if "/daily/" in path:
        base = [{"symbol": f"{symbol}.JK", "date": f"2026-09-{d:02d}", "volume": 10_000_000} for d in range(10, 28)]
        return base + [{"symbol": f"{symbol}.JK", "date": _MOCK_DATE, "volume": int(10_000_000 * mult)}]
    return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vex-Watcher IDX daily watchdog")
    p.add_argument("--mock", action="store_true", help="use built-in fake Sectors data (0 Sectors credits)")
    p.add_argument("--no-llm", action="store_true", help="skip the LLM call (needs no LLM key)")
    p.add_argument("--dry-run", action="store_true", help="print the Discord payload instead of posting it")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    # Keys that a given test mode does not use become optional placeholders.
    if args.mock:
        os.environ.setdefault("SECTORS_API_KEY", "mock")
    if args.no_llm:
        os.environ.setdefault("LLM_API_KEY", "unused")
    if args.dry_run:
        os.environ.setdefault("DISCORD_WEBHOOK_URL", "unused")
    fetch = mock_get_json if args.mock else get_json
    settings = Settings.from_env()
    now = datetime.now(timezone.utc)
    session = build_session()
    key = settings.sectors_api_key
    failed: list[str] = []

    # --- Step 1: top-level market-wide calls (4 credits total) ---
    buys_payload = fetch(session, key, "/v2/foreign-flow/", {"limit": TOP_N}, "foreign_buys")
    sells_payload = fetch(
        session, key, "/v2/foreign-flow/", {"order_by": "net_foreign_inflow", "limit": TOP_N}, "foreign_sells"
    )
    brokers_payload = fetch(session, key, "/v2/brokers/top/", {"metric": "net"}, "top_brokers")
    top_level_failures = sum(p is None for p in (buys_payload, sells_payload, brokers_payload))
    if buys_payload is None or sells_payload is None:
        failed.append("foreign_flow")
    if brokers_payload is None:
        failed.append("top_brokers")

    # --- Step 2: deterministic filtering ---
    findings = Findings()
    findings.foreign_buys, buy_date = parse_foreign_flow(buys_payload, want_buys=True)
    findings.foreign_sells, sell_date = parse_foreign_flow(sells_payload, want_buys=False)
    findings.trade_date = buy_date or sell_date
    findings.top_brokers = parse_top_brokers(brokers_payload)

    # Deep-check the top foreign-buy tickers for accumulation / volume spikes.
    if findings.trade_date:
        end = datetime.strptime(findings.trade_date, "%Y-%m-%d")
        baseline_start = (end - timedelta(days=VOLUME_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        hits: list[Accumulation] = []
        for flow in findings.foreign_buys[:MAX_CANDIDATES]:
            summary = fetch(
                session, key, f"/v2/broker-summary/{flow.ticker}/top/",
                {"start": findings.trade_date, "end": findings.trade_date, "n_brokers": BROKER_TOP_N},
                f"broker_summary:{flow.ticker}",
            )
            daily = fetch(
                session, key, f"/v2/daily/{flow.ticker}/",
                {"start": baseline_start, "end": findings.trade_date},
                f"daily:{flow.ticker}",
            )
            if summary is None and "broker_summary" not in failed:
                failed.append("broker_summary")
            if daily is None and "daily" not in failed:
                failed.append("daily")
            if summary is None and daily is None:
                continue
            hit = evaluate_accumulation(flow.ticker, summary, daily, findings.trade_date)
            if hit:
                hits.append(hit)
        findings.accumulations = rank_accumulations(hits)

    log.info(
        "Findings (%s): %d buys, %d sells, %d accumulations",
        findings.trade_date, len(findings.foreign_buys), len(findings.foreign_sells), len(findings.accumulations),
    )

    # --- Step 3: LLM summary (only if there is something to say) ---
    has_data = bool(findings.foreign_buys or findings.foreign_sells or findings.accumulations)
    summary_text = summarize(settings, llm_payload(findings)) if has_data and not args.no_llm else None
    if has_data and not args.no_llm and summary_text is None:
        failed.append("llm")

    if top_level_failures == 3:
        status = "FAILED (" + ", ".join(failed) + ")"
    elif failed:
        status = "DEGRADED (" + ", ".join(failed) + ")"
    elif not has_data:
        status = "OK (no data returned; market may be closed)"
    else:
        status = "OK"

    # --- Step 4: Discord ---
    if args.mock:
        status += " · MOCK DATA"
    payload = build_payload(settings, findings, summary_text, status, now)
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        sent = True
    else:
        sent = post_discord(settings.discord_webhook_url, payload)
    log.info("Pipeline status: %s | Discord delivered: %s", status, sent)
    return 0 if sent and not status.startswith("FAILED") else 1


if __name__ == "__main__":
    sys.exit(main())
