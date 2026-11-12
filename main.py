#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 for the scored window.

Model (all parameters estimated from the history at run time, except the
Laplace shape constants, which are regularities of this chart):
- A game's rank drifts DOWN the chart over time (new games push in and old
  ones fade): the hourly drift depends on the rank band and is estimated
  from pairs of reads 12-96h apart, shrunk toward the global mean.
- A game's rank at a future hour is Laplace-distributed around
  last_rank + drift * hours, with a scale that grows slowly with the
  horizon (b = b0 + 0.10 * hours). Laplace fits the observed rank changes
  far better than a normal: most reads move a game by 0-2 places, but
  jumps of 10+ happen.
- A game's recent slope (last up to 6 reads) nudges its drift a little
  (momentum coefficient 0.2).
- The forecast for a place is the time-average of that Laplace density
  over the window's hourly slots, so early hours (tight, near the last
  read) and late hours (broad, drifted down) both contribute.
- A game is named at a place only while its probability beats what that
  place's unnamed share would give it, so the named rows never dilute
  the forecast.
- `created`/`updated` from the details are deliberately NOT used: on
  this chart 23% of the games are older than 28 days by `created`, so
  the 28-day rule does not run on that field.
Standard library only, no network, deterministic.
"""

import json
import math
import os
import statistics
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Laplace shape constants (fitted on the chart's rank-change data).
B0_BASE = 0.2          # base scale for a game's rank distribution
B0_RANK_SD = 0.3       # extra scale per unit of the game's recent rank std
B_SLOPE = 0.10         # scale growth per hour of horizon
DRIFT_SCALE = 1.3      # multiplier on the estimated band drift
MOMENTUM = 0.2         # weight of a game's recent slope vs its band drift
PLACE_CAP = 0.97       # max named probability mass at one place
CAND_DAYS = 3          # candidate window: top 40 of reads in the last N days
CAND_TOPN = 40
BANDS = [(1, 3), (4, 6), (7, 10), (11, 15), (16, 25), (26, 40), (41, 10 ** 9)]


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def band_of(rank):
    for band in BANDS:
        if band[0] <= rank <= band[1]:
            return band
    return BANDS[-1]


def laplace_cdf(x, mu, b):
    if b <= 0:
        b = 1e-9
    if x < mu:
        return 0.5 * math.exp((x - mu) / b)
    return 1.0 - 0.5 * math.exp(-(x - mu) / b)


def load_history(input_dir):
    reads = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            chart = row.get("up_and_coming") or []
            if not chart or not row.get("at"):
                continue
            try:
                at = parse_time(row["at"])
            except (ValueError, AttributeError):
                continue
            ranks = {}
            for game in chart:
                try:
                    ranks[int(game["universe_id"])] = int(game["rank"])
                except (KeyError, TypeError, ValueError):
                    continue
            if ranks:
                reads.append((at, ranks))
    reads.sort(key=lambda item: item[0])
    return reads


def estimate_drift(reads):
    """Hourly rank drift per rank band, from read pairs 12-96h apart."""
    pairs = []  # (hours, start rank, rank change)
    for i, (t0, ranks0) in enumerate(reads):
        for t1, ranks1 in reads[i + 1:]:
            hours = (t1 - t0).total_seconds() / 3600.0
            if hours < 12:
                continue
            if hours > 96:
                break
            for uid, rank0 in ranks0.items():
                if rank0 > 40:
                    continue
                rank1 = ranks1.get(uid)
                if rank1 is not None:
                    pairs.append((hours, rank0, rank1 - rank0))
    if not pairs:
        return {band: 0.2 for band in BANDS}
    global_drift = statistics.mean(change / hours for hours, _, change in pairs)
    drift = {}
    for band in BANDS:
        values = [change / hours for hours, rank, change in pairs
                  if band_of(rank) == band]
        if values:
            # shrink toward the global mean (30 pseudo-observations)
            drift[band] = (sum(values) + 30 * global_drift) / (len(values) + 30)
        else:
            drift[band] = global_drift
    return drift


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window", {}) or {}
    reads = load_history(INPUT_DIR)
    if not reads:
        raise SystemExit("no usable reads in history.ndjson")
    last_time, last_ranks = reads[-1]
    # The window comes from task.json; fall back to 72h after the last
    # read if a kit ever omits it.
    try:
        window_from = parse_time(window["from"])
        window_until = parse_time(window["until"])
    except (KeyError, TypeError, ValueError):
        window_from = last_time
        window_until = last_time + timedelta(hours=72)
    chart_size = len(last_ranks)

    drift = estimate_drift(reads)

    # Per-game rank observations: (time, rank), oldest first.
    games = {}
    for at, ranks in reads:
        for uid, rank in ranks.items():
            games.setdefault(uid, []).append((at, rank))

    # Candidate set: top CAND_TOPN of any read in the last CAND_DAYS,
    # plus every game on the last read's chart.
    candidates = set(last_ranks)
    cutoff = last_time - timedelta(days=CAND_DAYS)
    for at, ranks in reads:
        if at >= cutoff:
            top = sorted(ranks.items(), key=lambda item: item[1])[:CAND_TOPN]
            candidates.update(uid for uid, _ in top)
    candidates = sorted(candidates)

    # Hourly slots of the scored window.
    slots = []
    slot = window_from
    while slot < window_until:
        slots.append((slot - window_from).total_seconds() / 3600.0)
        slot += timedelta(hours=1)
    if not slots:
        slots = [0.0]

    # Time-averaged Laplace density per (game, place).
    prob = {}
    for uid in candidates:
        obs = games.get(uid)
        if not obs:
            continue
        last_obs_time, rank0 = obs[-1]
        recent = obs[-6:]
        recent_ranks = [rank for _, rank in recent]
        rank_sd = statistics.pstdev(recent_ranks) if len(recent_ranks) >= 2 else 1.0
        base_scale = B0_BASE + B0_RANK_SD * rank_sd
        if len(recent) >= 2:
            span = (recent[-1][0] - recent[0][0]).total_seconds() / 3600.0
            slope = (recent[-1][1] - recent[0][1]) / span if span > 1 else 0.0
        else:
            slope = 0.0
        band_drift = drift[band_of(rank0)]
        game_drift = DRIFT_SCALE * band_drift + MOMENTUM * (slope - band_drift)
        age_hours = (last_time - last_obs_time).total_seconds() / 3600.0
        for place in range(1, places + 1):
            total = 0.0
            for hours in slots:
                horizon = hours + age_hours
                mu = rank0 + game_drift * horizon
                scale = base_scale + B_SLOPE * horizon
                if place == 1:
                    total += laplace_cdf(1.5, mu, scale)
                else:
                    total += (laplace_cdf(place + 0.5, mu, scale)
                              - laplace_cdf(place - 0.5, mu, scale))
            prob[(uid, place)] = total / len(slots)

    # Name a game at a place only while its probability beats the
    # unnamed share that place would otherwise give it.
    probabilities = []
    for place in range(1, places + 1):
        ranked = sorted(
            ((prob.get((uid, place), 0.0), uid) for uid in candidates),
            reverse=True,
        )
        named = []
        named_mass = 0.0
        for p, uid in ranked:
            if p <= 0:
                break
            unnamed = (1.0 - named_mass) / max(1, chart_size - len(named))
            if p <= unnamed:
                break
            if named_mass + p > PLACE_CAP:
                break
            named.append((uid, p))
            named_mass += p
        for uid, p in named:
            probabilities.append({"universe_id": uid, "place": place,
                                  "p": round(p, 6)})

    # Top 10 by total probability of being anywhere in the top 10.
    top_mass = {}
    for (uid, place), p in prob.items():
        top_mass[uid] = top_mass.get(uid, 0.0) + p
    top10 = [uid for uid, _ in sorted(top_mass.items(),
                                      key=lambda item: -item[1])[:places]]

    forecast = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as fh:
        json.dump(forecast, fh)
        fh.write("\n")


if __name__ == "__main__":
    main()
