"""Coverage report: what has been downloaded, what remains, what it cost.

Written to data/catalog/PROGRESS.md on every run so the state of the
extraction survives restarts, machine changes and long gaps between sessions.
Everything here is derived from the ledger and the parquet on disk -- nothing
is hand-maintained, so it cannot drift from reality.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..config import CATALOG_DIR, PARQUET_DIR
from ..db import query
from ..ledger import Ledger
from ..universe import START_DATE, select

# endpoint -> (human label, how the expected total is derived)
PLANNED: list[tuple[str, str, str]] = [
    ("exchanges-list", "Exchange directory", "const:1"),
    ("exchange-symbol-list", "Ticker lists (active + delisted)", "const:141"),
    ("eod", "Equity EOD history", "universe"),
    ("splits", "Equity splits", "universe"),
    ("dividends", "Equity dividends", "universe"),
    ("eod-nonequity", "Forex / crypto / bond / money EOD", "const:9819"),
    ("fundamentals-index", "Index constituents (point-in-time)", "discovered"),
    ("exchange-details", "Trading hours + holidays", "const:70"),
    ("calendar-earnings", "Historical earnings calendar", "const:319"),
    ("macro-indicator", "Macro indicators", "const:1287"),
    ("fundamentals", "Equity fundamentals", "universe"),
    ("eod-probe", "History-depth probes (sampling)", "discovered"),
]


def _expected(rule: str, universe_size: int, seen: int) -> int | None:
    if rule.startswith("const:"):
        return int(rule.split(":", 1)[1])
    if rule == "universe":
        return universe_size
    return seen or None


def build_report(ledger: Ledger) -> dict:
    universe = select()
    n_universe = len(universe)

    by_endpoint: dict[str, dict[str, int]] = {}
    for row in ledger.summary():
        d = by_endpoint.setdefault(row["endpoint"], {})
        d[row["status"]] = row["n"]
        d["calls"] = d.get("calls", 0) + (row["calls"] or 0)
        d["bytes"] = d.get("bytes", 0) + (row["bytes"] or 0)

    datasets = []
    for endpoint, label, rule in PLANNED:
        d = by_endpoint.get(endpoint, {})
        ok = d.get("ok", 0)
        empty = d.get("empty", 0)
        nf = d.get("not_found", 0)
        failed = sum(v for k, v in d.items()
                     if k in ("http_error", "transport_error"))
        done = ok + empty + nf
        exp = _expected(rule, n_universe, done)
        if exp and done >= exp:
            state = "complete"
        elif done == 0:
            state = "not started"
        else:
            state = "partial"
        datasets.append({
            "endpoint": endpoint, "label": label, "state": state,
            "ok": ok, "empty": empty, "not_found": nf, "failed": failed,
            "done": done, "expected": exp,
            "remaining": max((exp or done) - done, 0),
            "calls": d.get("calls", 0), "bytes": d.get("bytes", 0),
        })

    artifacts = []
    for path in sorted(PARQUET_DIR.glob("*.parquet")):
        rows, = query(f"SELECT COUNT(*) FROM read_parquet('{path}')")[0]
        artifacts.append({"name": path.name, "rows": rows,
                          "bytes": path.stat().st_size, "files": 1})
    for d in sorted(p for p in PARQUET_DIR.iterdir() if p.is_dir()):
        files = sorted(d.rglob("*.parquet"))
        if not files:
            continue
        rows, = query(f"SELECT COUNT(*) FROM read_parquet('{d}/**/*.parquet')")[0]
        artifacts.append({"name": f"{d.name}/", "rows": rows,
                          "bytes": sum(f.stat().st_size for f in files),
                          "files": len(files)})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_size": n_universe,
        "start_date": START_DATE,
        "datasets": datasets,
        "artifacts": artifacts,
        "budget": [dict(r) for r in ledger.budget_history()],
        "failures": [
            {"endpoint": r["endpoint"], "key": r["key"], "status": r["status"],
             "http": r["http_status"], "error": (r["error"] or "")[:200]}
            for r in ledger.failures()
        ],
    }


def _mb(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    return f"{n / (1 << 20):.1f} MB"


def to_markdown(rep: dict) -> str:
    L = [
        "# XCAP extraction progress",
        "",
        f"_Generated {rep['generated_at']} — regenerate with `python -m xcap.cli coverage`._",
        "",
        f"Equity universe: **{rep['universe_size']:,} securities** "
        f"(US NYSE/NASDAQ/AMEX/ARCA/BATS, Common Stock + ETF + Preferred). "
        f"Dataset floor **{rep['start_date']}**.",
        "",
        "## Downloads",
        "",
        "| dataset | state | fetched | expected | remaining | calls | raw |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for d in rep["datasets"]:
        if d["done"] == 0 and d["state"] == "not started":
            L.append(f"| {d['label']} | – not started | 0 | "
                     f"{d['expected'] or '?'} | {d['remaining'] or '?'} | 0 | – |")
            continue
        L.append(
            f"| {d['label']} | {d['state']} | {d['done']:,} | "
            f"{d['expected'] or '?'} | {d['remaining']:,} | {d['calls']:,} | "
            f"{_mb(d['bytes'])} |"
        )

    L += ["", "## Built artifacts", "",
          "| dataset | rows | files | size |", "|---|---:|---:|---:|"]
    for a in rep["artifacts"]:
        L.append(f"| `{a['name']}` | {a['rows']:,} | {a['files']} | {_mb(a['bytes'])} |")

    L += ["", "## API call budget (GMT day)", "", "| day | calls |", "|---|---:|"]
    for b in rep["budget"]:
        L.append(f"| {b['gmt_day']} | {b['calls_spent']:,} |")

    if rep["failures"]:
        L += ["", f"## Unrecovered requests ({len(rep['failures'])})", "",
              "| endpoint | key | status | http |", "|---|---|---|---|"]
        for f in rep["failures"][:40]:
            L.append(f"| {f['endpoint']} | {f['key']} | {f['status']} | {f['http']} |")
    else:
        L += ["", "## Unrecovered requests", "", "None."]

    L += [
        "",
        "## Resuming",
        "",
        "Every request is keyed in `data/catalog/ledger.sqlite` by "
        "`(endpoint, key, params_hash)`. Re-running any fetch command skips "
        "what is already recorded and spends no calls on it, so interrupted "
        "runs cost nothing to resume. Raw responses live in `data/_raw/` "
        "(zstd, sha256 in the ledger) and every parquet is rebuildable from "
        "them offline with the `*-build` commands.",
        "",
    ]
    return "\n".join(L)


def write_report(ledger: Ledger) -> tuple[dict, str]:
    rep = build_report(ledger)
    md = to_markdown(rep)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (CATALOG_DIR / "progress.json").write_text(json.dumps(rep, indent=2))
    (CATALOG_DIR / "PROGRESS.md").write_text(md)
    return rep, md
