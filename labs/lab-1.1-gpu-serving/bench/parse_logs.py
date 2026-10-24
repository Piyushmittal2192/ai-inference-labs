#!/usr/bin/env python3
"""Extract per-request engine timings from the Lab 1.1 server log.

Usage
  docker logs lab11 2>&1 | python3 bench/parse_logs.py
  docker logs lab11 2>&1 | python3 bench/parse_logs.py --group-by completion_tokens
  docker logs lab11 2>&1 | python3 bench/parse_logs.py --csv results/engine.csv

The server writes one `event=request_done` line per finished request. This
script groups those lines and shows where each TTFT millisecond went
(queue, tokenize, prefill), plus prefill throughput and decode ITL, as seen
from INSIDE the engine. Compare it with bench.py, which measures from the
client: the difference is serving overhead (HTTP, SSE framing, the network).

Requests whose token counts are within 10% of each other are grouped together,
because random prompts land a few tokens either side of the sweep target.
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence

PAIR = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
NUMERIC = (
    "prompt_tokens", "completion_tokens", "queue_ms", "tokenize_ms", "prefill_ms", "ttft_ms", "decode_ms",
    "decode_steps", "itl_mean_ms", "itl_p50_ms", "itl_p99_ms", "prefill_tokens_per_s",
    "decode_tokens_per_s", "peak_memory_mb",
)


def parse_line(line: str) -> dict[str, str] | None:
    marker = line.find("event=")
    if marker < 0:
        return None
    fields = {key: value.strip('"') for key, value in PAIR.findall(line[marker:])}
    return fields if "event" in fields else None


def to_number(value: str | None) -> float | None:
    if value is None or value in ("na", "None", ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pct(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1]


def fmt(value: float | None, width: int = 9, digits: int = 1) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def cluster(rows: list[dict[str, float | None]], key: str, tolerance: float) -> list[list[dict[str, float | None]]]:
    """Group rows whose `key` values are within `tolerance` (relative) of the group's first value."""
    ordered = sorted((r for r in rows if r.get(key) is not None), key=lambda r: r[key])
    groups: list[list[dict[str, float | None]]] = []
    for row in ordered:
        if groups and row[key] <= groups[-1][0][key] * (1.0 + tolerance):
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def column(rows: Iterable[dict[str, float | None]], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Lab 1.1 engine timings from server logs.")
    parser.add_argument("logfile", nargs="?", help="log file to read (default: stdin)")
    parser.add_argument("--group-by", choices=("prompt_tokens", "completion_tokens", "none"), default="prompt_tokens")
    parser.add_argument("--tolerance", type=float, default=0.10, help="relative width of a group (default 0.10)")
    parser.add_argument("--csv", help="also write every parsed request_done line to this CSV file")
    args = parser.parse_args()

    stream = open(args.logfile, encoding="utf-8", errors="replace") if args.logfile else sys.stdin
    done: list[dict[str, float | None]] = []
    raw_done: list[dict[str, str]] = []
    other = Counter()
    failures = Counter()
    with stream:
        for line in stream:
            fields = parse_line(line)
            if fields is None:
                continue
            event = fields["event"]
            if event == "request_done":
                raw_done.append(fields)
                done.append({key: to_number(fields.get(key)) for key in NUMERIC})
            elif event == "request_failed":
                failures[fields.get("error_type", "unknown")] += 1
            else:
                other[event] += 1

    if args.csv and raw_done:
        keys = sorted({key for row in raw_done for key in row})
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(raw_done)

    print(f"parsed {len(done)} request_done lines; failed={dict(failures) or 0}; other events={dict(other) or 0}")
    if not done:
        print("nothing to summarize: send some requests first (bench/bench.py), then re-run")
        return 1

    groups = [done] if args.group_by == "none" else cluster(done, args.group_by, args.tolerance)
    label = "all" if args.group_by == "none" else args.group_by
    print(f"\nengine view, grouped by {label} (p50 unless noted)")
    print(f"{'group':>8} {'n':>4} {'queue':>8} {'tokenize':>9} {'prefill':>8} {'ttft':>8} {'ttft_p99':>9} "
          f"{'prefill_tok/s':>13} {'itl':>7} {'itl_p99':>8} {'decode_tok/s':>12} {'peak_mb':>8}")
    for group in groups:
        key_values = column(group, args.group_by) if args.group_by != "none" else []
        name = fmt(pct(key_values, 50), 8, 0) if key_values else f"{'all':>8}"
        print(
            f"{name} {len(group):>4} {fmt(pct(column(group, 'queue_ms'), 50), 8)} "
            f"{fmt(pct(column(group, 'tokenize_ms'), 50), 9, 2)} {fmt(pct(column(group, 'prefill_ms'), 50), 8)} "
            f"{fmt(pct(column(group, 'ttft_ms'), 50), 8)} {fmt(pct(column(group, 'ttft_ms'), 99), 9)} "
            f"{fmt(pct(column(group, 'prefill_tokens_per_s'), 50), 13, 0)} "
            f"{fmt(pct(column(group, 'itl_p50_ms'), 50), 7, 2)} {fmt(pct(column(group, 'itl_p99_ms'), 50), 8, 2)} "
            f"{fmt(pct(column(group, 'decode_tokens_per_s'), 50), 12)} "
            f"{fmt(max(column(group, 'peak_memory_mb'), default=None), 8, 0)}"
        )
    print("\nttft here = queue + tokenize + prefill, measured inside the server.")
    print("bench.py's ttft adds HTTP, SSE framing and the network on top of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
