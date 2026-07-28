"""Generate research/01_read_ohlcv.ipynb.

The notebook is generated rather than hand-written so it can be regenerated
after schema changes, and so the source stays reviewable as plain Python.
Run: .venv/bin/python research/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = Path(__file__).parent / "01_read_ohlcv.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s.strip())          # noqa: E731
code = lambda s: nbf.v4.new_code_cell(s.strip())            # noqa: E731

cells = [
md("""
# Reading the XCAP OHLCV dataset

First look at the equity price dataset built by `xcap phase1-build`. Covers how
to load it, the two joins that matter, and the traps that will silently corrupt
a backtest if you get them wrong.

**Read [`docs/DATA_RETRIEVAL.md`](../docs/DATA_RETRIEVAL.md) for why the dataset
is shaped this way.** The short version:

- `eod/` holds **raw** OHLCV — adjusted for neither splits nor dividends.
- `adjustments/` holds factors computed locally. `adjusted = close * price_factor`.
- `vendor_adjusted_close` exists **only** for reconciliation. Do not trade on it.
"""),

code("""
import duckdb, pandas as pd, numpy as np
from pathlib import Path

ROOT    = Path.cwd().parent if Path.cwd().name == "research" else Path.cwd()
PARQUET = ROOT / "data" / "parquet"

con = duckdb.connect()
con.execute(f"SET temp_directory='{ROOT / 'data' / '_duckdb_tmp'}'")
for name, src in {
    "eod":        f"{PARQUET}/eod/**/*.parquet",
    "adj":        f"{PARQUET}/adjustments/**/*.parquet",
    "splits":     f"{PARQUET}/splits.parquet",
    "securities": f"{PARQUET}/securities.parquet",
}.items():
    con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{src}')")

print(con.execute("DESCRIBE eod").df().to_string(index=False))
"""),

md("## 1. Shape of the dataset"),

code("""
con.execute('''
    SELECT COUNT(*) AS bars, COUNT(DISTINCT security_id) AS securities,
           MIN(date) AS first_date, MAX(date) AS last_date
    FROM eod
''').df()
"""),

code("""
# Bars per year. The 2000 floor is deliberate: the vendor's delisted archive
# begins ~1997-98, so no earlier start year is survivorship-bias free.
by_year = con.execute('''
    SELECT year(date) AS year, COUNT(*) AS bars,
           COUNT(DISTINCT security_id) AS securities
    FROM eod GROUP BY 1 ORDER BY 1
''').df()
by_year.head(30)
"""),

md("""
## 2. Survivorship — the reason this dataset exists

The universe includes securities that stopped trading. A dataset of *current*
listings would quietly exclude every company that failed, which is the single
most damaging bias in equity backtesting.
"""),

code("""
con.execute('''
    SELECT s.is_delisted,
           COUNT(DISTINCT e.security_id) AS securities,
           MIN(e.date) AS first_bar, MAX(e.date) AS last_bar
    FROM eod e JOIN securities s USING (security_id)
    GROUP BY 1 ORDER BY 1
''').df()
"""),

code("""
# Securities whose price history ends well before the dataset does: these are
# the names a survivorship-biased dataset would be missing entirely.
con.execute('''
    SELECT s.api_ticker, s.name, s.venue,
           MIN(e.date) AS first_bar, MAX(e.date) AS last_bar, COUNT(*) AS bars
    FROM eod e JOIN securities s USING (security_id)
    WHERE s.is_delisted
    GROUP BY 1,2,3
    HAVING MAX(e.date) < DATE '2010-01-01'
    ORDER BY bars DESC
    LIMIT 10
''').df()
"""),

md("""
## 3. Adjusted prices — the join that matters

`eod.close` is **raw**. To get a return series, multiply by `price_factor` from
`adjustments`, joined on `(security_id, date)`.

The factor is anchored so the most recent bar is exactly 1.0, and every earlier
bar carries the cumulative product of corporate actions after it.
"""),

code("""
px = con.execute('''
    SELECT e.date, e.close AS raw_close,
           e.close * a.price_factor AS adj_close,
           a.split_factor, a.price_factor, e.vendor_adjusted_close
    FROM eod e
    JOIN adj a USING (security_id, date)
    WHERE e.api_ticker = 'AAPL.US'
    ORDER BY e.date
''').df()

print(f"{len(px):,} bars")
print("\\nAAPL around the 4-for-1 split on 2020-08-31:")
px[(px.date >= pd.Timestamp('2020-08-26')) & (px.date <= pd.Timestamp('2020-09-03'))]
"""),

md("""
Note the raw close drops ~4x across the split while the adjusted series is
continuous. A strategy computing returns from `raw_close` would see a fictional
-75% day.
"""),

code("""
raw_ret = px.set_index('date').raw_close.pct_change()
adj_ret = px.set_index('date').adj_close.pct_change()
split_day = pd.Timestamp('2020-08-31')
pd.DataFrame({
    'return from raw_close': [raw_ret.loc[split_day]],
    'return from adj_close': [adj_ret.loc[split_day]],
}, index=['2020-08-31'])
"""),

md("""
## 4. Building a price matrix for the backtester

`BACKTEST.py` wants a date x ticker DataFrame of prices. Pivot the adjusted
series. Keep it to a small, liquid universe here — the full dataset is 59M bars
and pivoting all of it is not something to do casually.
"""),

code("""
UNIVERSE = ['AAPL.US','MSFT.US','JNJ.US','XOM.US','KO.US','IBM.US','GE.US','PG.US']

prices = con.execute(f'''
    SELECT e.date, e.api_ticker, e.close * a.price_factor AS px
    FROM eod e JOIN adj a USING (security_id, date)
    WHERE e.api_ticker IN ({",".join(f"'{t}'" for t in UNIVERSE)})
      AND e.date >= DATE '2010-01-01'
''').df().pivot(index='date', columns='api_ticker', values='px').sort_index()

print(prices.shape)
prices.tail(3)
"""),

md("""
## 5. Feeding it to `BACKTEST.py`

Equal-weight, monthly rebalance, purely as an integration check that the dataset
plugs into the backtester. This is **not** a strategy.
"""),

code("""
import sys
sys.path.insert(0, str(Path.cwd() if Path.cwd().name == 'research' else ROOT / 'research'))
from BACKTEST import backtest

w = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
rebal = prices.resample('ME').last().index

res = backtest(w, prices, signal_dates=list(rebal), transaction_cost=0.0005)
{k: (round(v, 4) if isinstance(v, float) else v)
 for k, v in res.items() if isinstance(v, (int, float))}
"""),

md("""
## 6. Caveats you must carry into any research

1. **These are price returns, not total returns.** Dividends have not been
   downloaded yet, so `price_factor` currently equals `split_factor`. Returns are
   systematically understated for dividend payers. Check
   `data/catalog/phase1_manifest.json` -> `adjustments.factor_meaning`.

2. **Join on `security_id`, never on ticker.** Symbols are recycled after
   delisting. `api_ticker` is in `eod` for convenience only.

3. **~13% of securities with corporate actions show events dated outside their
   own price history** — spliced or recycled series. Treat those with suspicion;
   see the `spliced / recycled tickers` check in `xcap phase1-qa`.

4. **Known vendor defects**, all small but present: ~0.04% of bars have
   non-positive prices, ~0.01% violate `low <= {open,close} <= high`. They are
   flagged rather than silently repaired, because dropping vs winsorising is a
   strategy decision.

5. **The universe is deliberately not filtered by liquidity or price.** It
   includes sub-penny microcaps. Apply your own eligibility screen.
"""),

code("""
import json
m = json.load(open(ROOT / 'data' / 'catalog' / 'phase1_manifest.json'))
print('start_date      :', m['start_date'])
print('skipped blocks  :', m.get('skipped', {}))
a = json.load(open(ROOT / 'data' / 'catalog' / 'progress.json'))
print('universe        :', f"{a['universe_size']:,} securities")
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, NB)
print(f"wrote {NB} ({len(cells)} cells)")
