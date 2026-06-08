# N08 #5F-3 — 08X `.txt`→5-min Raw-Bar Loader Design

> Status: design 2026-06-08. First half of the 08X real-data wiring (sub-slice ②
> of Block 1). Adds the genuinely-missing primitive: a package loader that turns
> the 1-minute `.txt` source into the canonical 5-minute pooled frame the data
> layer consumes. #5F-4 (separate slice) then chains
> features→labels→splits→windows into `resolve_train_inner_index`. Tooling:
> inline design + `humanize:ask-codex` review (run
> `.humanize/skill/2026-06-08_00-45-47-1816-c5179652/`). Codex chose the
> loader-only scope split (Q1 P0).

## 1. Goal & Scope

The 08X data source is 5 per-ticker **1-minute** `.txt` files (CSCO/JPM/KO/MSFT/
WMT), but `data/raw_bars.py::load_ticker_bars` expects already-5-minute CSVs. The
`.txt`→5min resample currently lives ONLY in the Stage-0 notebook. This slice
adds it to the package so 08X (and any stage) can load the real source.

**In scope**: three functions in `data/raw_bars.py` —
`load_one_minute_txt`, `resample_to_five_minutes`, `load_ticker_bars_txt` —
plus synthetic-`.txt` tests. **Local file paths only** (Codex Q3): the Drive
download stays notebook glue (configs/data.yaml records the file IDs).

**Out of scope**: the features→labels→splits→windows chain + `resolve_train_inner_index`
production path (#5F-4); the Colab Drive-download code; any change to
`load_ticker_bars` (CSV) or features/labels/splits/windows (frozen + tested);
model fit.

## 2. Canonical recipe (replicate Stage 0 EXACTLY — AGENTS §5)

From `scripts/create_config_screening_colab_notebook.py` (`read_one_minute_txt`
+ `resample_to_five_minutes`):

1. Read `.txt`: comma-separated, columns `Date,Time,Open,High,Low,Close,Volume`
   (`RAW_TXT_COLUMNS`); optional `Date` header row tolerated. Parse
   `timestamp = to_datetime(Date + " " + Time, format="%m/%d/%Y %H:%M")`.
2. Numeric-coerce OHLCV (errors="raise").
3. Filter the 1-minute rows to `timestamp < VAL_END` AND
   `MARKET_OPEN(09:30) <= time <= MARKET_CLOSE(16:00)` (inclusive).
4. Resample 1min→5min on the timestamp index:
   `open=first, high=max, low=min, close=last, volume=sum`, then
   `dropna(subset=[open,high,low,close,volume])`.
5. Filter the resampled 5-min rows again to RTH `09:30..16:00` inclusive.
6. Return columns `(timestamp, open, high, low, close, volume)` in that order.

`VAL_END = 2017-01-25` is the frozen holdout boundary, already a module constant
in `data/raw_bars.py`.

**Capping nuance (Codex Q2/Q4)**: the raw provider file MAY contain rows ≥ VAL_END.
The loader must CAP/DROP them during load (step 3), NOT raise on their mere
presence. After producing the pooled 5-min frame it MUST assert no bar ≥ val_end
survived (reuse the existing holdout-closure postcondition).

## 3. New functions in `data/raw_bars.py`

```python
MARKET_OPEN  = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()
RAW_TXT_COLUMNS = ("Date", "Time", "Open", "High", "Low", "Close", "Volume")

def load_one_minute_txt(
    path: str | Path, *, val_end: str | pd.Timestamp = VAL_END
) -> pd.DataFrame:
    """Read ONE 1-minute .txt, parse + RTH-filter + drop >= val_end, return
    the pre-holdout 1-minute frame [timestamp, open, high, low, close, volume].
    Single ticker; no ticker column. Fail-loud on parse/NaT/tz-aware/missing
    columns (mirrors load_ticker_bars' guards)."""

def resample_to_five_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """1min -> 5min: open=first/high=max/low=min/close=last/volume=sum, dropna,
    RTH 09:30..16:00 inclusive. Returns [timestamp, open, high, low, close,
    volume]. Pure; no I/O."""

def load_ticker_bars_txt(
    manifest: Mapping[str, str | Path], *, val_end: str | pd.Timestamp = VAL_END
) -> pd.DataFrame:
    """Per ticker: load_one_minute_txt -> resample_to_five_minutes -> inject
    ticker. Pool, sort by (ticker, timestamp), run _validated_ohlcv + the
    holdout-closure postcondition (no bar >= val_end). Returns EXACTLY the same
    canonical frame load_ticker_bars returns:
    [ticker, timestamp, open, high, low, close, volume]."""
```

`load_ticker_bars` (CSV) stays byte-for-byte unchanged. The two loaders return an
identical schema so #5F-4 can consume either.

## 4. Tests — `tests/data/test_raw_bars_txt.py` (synthetic only)

Tiny synthetic `.txt` files written to `tmp_path` (Codex Q6) — no real Drive file.
Cover:
1. happy path: 2 tickers × a few RTH days < VAL_END → correct 5-min frame
   (row counts, OHLC aggregation values, volume sum, canonical column order).
2. resample math: a hand-checked 5-bar→1-bar block asserts
   open=first/high=max/low=min/close=last/volume=sum exactly.
3. RTH filter: pre-09:30 and post-16:00 1-min rows are dropped; 09:30 and 16:00
   edges are kept (inclusive).
4. header row tolerated; a literal "Date" header line is skipped.
5. post-VAL_END capping: a file with rows spanning VAL_END loads only the
   pre-VAL_END portion AND the returned frame has no bar ≥ VAL_END (no raise on
   mere presence; raise only if a ≥ VAL_END bar survived).
6. fail-loud: missing column / unparseable timestamp / tz-aware → ValueError.
7. `load_ticker_bars_txt` output schema == `load_ticker_bars` schema (same
   columns + order + dtypes), so #5F-4 is loader-agnostic.
8. **Recipe-parity lock (Codex Q4)**: assert the package loader's 5-min frame
   equals a reference resample encoding the Stage-0 recipe. If the Stage-0
   functions are importable side-effect-free from
   `scripts/create_config_screening_colab_notebook.py`, call them directly;
   otherwise the test embeds a self-contained reference resample with the same
   recipe (documented as the lock). Do NOT make Stage 0 import package code in
   this slice (keeps notebook-generator/static-gate scope unchanged).

## 5. configs/data.yaml (already created, uncommitted)

Records the 5 Drive file IDs + the 1min→5min recipe + boundaries as the single
source of truth. The Colab download cell (notebook glue) reads it; package code
does not (Q3). This slice commits it alongside the loader.

## 6. Files touched
- MODIFY `src/intraday_research/data/raw_bars.py` (3 new functions + 2 constants;
  CSV loader unchanged)
- NEW `tests/data/test_raw_bars_txt.py`
- ADD `configs/data.yaml` (created this session)
- NEW this spec

**Do NOT touch**: features/labels/splits/windows, the stages, `load_ticker_bars`.

## 7. Acceptance criteria
1. `bash scripts/check_n08_resume_gate.sh` exits 0.
2. `pytest tests/data -q` green (existing + new txt-loader tests).
3. `pytest tests/stages/models -q` unchanged (419+2).
4. `load_ticker_bars_txt` returns the canonical 7-column frame; no bar ≥ VAL_END.
5. Recipe-parity lock passes (5-min frame matches the Stage-0 recipe).
6. No real ticker `.txt` touched by any test; no Drive API in package code.

## 8. Next (#5F-4)
Chain `load_ticker_bars_txt → per-ticker build_features/build_no_trade_band_labels/
apply_stage0_chronological_split → pool → build_windows → filter PARTITION_TRAIN`
into `resolve_train_inner_index`'s production path, with
`target_valid_mask = label_valid & split_valid` (Codex Q5) and all params from
`config['frozen_candidate']` (Codex Q7).
