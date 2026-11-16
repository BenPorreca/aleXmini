#!/usr/bin/env python3
"""aleX mini generator. Builds a month of crosswords (5x5/6x6 minis, plus one big newspaper-style puzzle per week) from words.txt into puzzles.json.

Method: crossword filling is a constraint-satisfaction problem (CSP). Each word slot is a variable, its domain is
the words of that length, and crossing slots must agree on the shared letter. Search = backtracking with
  * minimum-remaining-values (fill the most constrained slot first),
  * arc consistency (after every choice, prune letters that can no longer cross),
  * all-different words, and randomised value order with a node limit + restarts (new grid pattern) to avoid dead ends.
Grid patterns are random 180-degree-symmetric black-square layouts; finished puzzles are re-validated independently.

Usage:  python generate.py --month 2026-11
"""
import argparse, calendar, json, random, sys, time
from datetime import date
from pathlib import Path

MIN_LEN, MAX_LEN = 3, 8

def load_words(path):
    clues = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        w, _, c = line.partition("|"); w, c = w.strip().upper(), c.strip()
        if w.isascii() and w.isalpha() and MIN_LEN <= len(w) <= MAX_LEN and c: clues.setdefault(w, []).append(c)
        else: print(f"  skipped (needs 'WORD | clue', 3-8 letters): {line}")
    return clues

def get_slots(pat):
    n = len(pat); white = lambda r, c: 0 <= r < n and 0 <= c < n and pat[r][c] != "#"
    out, num = [], 0
    for r in range(n):
        for c in range(n):
            if not white(r, c): continue
            runs = []
            for k, (dr, dc) in (("A", (0, 1)), ("D", (1, 0))):
                if not white(r - dr, c - dc):
                    cells, rr, cc = [], r, c
                    while white(rr, cc): cells.append((rr, cc)); rr += dr; cc += dc
                    if len(cells) >= MIN_LEN: runs.append((k, cells))
            if runs:
                num += 1
                out += [{"id": f"{num}{k}", "num": num, "dir": k, "cells": cells} for k, cells in runs]
    return out

def pattern_ok(pat, lens):
    n = len(pat); S = get_slots(pat)
    white = {(r, c) for r in range(n) for c in range(n) if pat[r][c] != "#"}
    if len(S) < 4 or {s["dir"] for s in S} != {"A", "D"}: return False
    if any(len(s["cells"]) not in lens for s in S): return False
    if {c for s in S for c in s["cells"]} != white: return False          # every white cell is in a word
    lines = list(pat) + ["".join(pat[r][c] for r in range(n)) for c in range(n)]
    if any(len(run) == 2 for line in lines for run in line.split("#")): return False   # no 2-letter runs
    seen, stack = set(), [next(iter(white))]                               # all white cells connected
    while stack:
        r, c = stack.pop()
        if (r, c) in seen or (r, c) not in white: continue
        seen.add((r, c)); stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    return seen == white

def random_pattern(n, lens, rng):
    half = [(r, c) for r in range(n) for c in range(n)][: n * n // 2]
    for _ in range(200):
        g = [["."] * n for _ in range(n)]
        for r, c in rng.sample(half, rng.randint(1, max(1, int(n * n * 0.14)))):
            g[r][c] = g[n - 1 - r][n - 1 - c] = "#"
        pat = ["".join(row) for row in g]
        if pattern_ok(pat, lens): return pat

def newspaper_pattern(n, lens, rng, maxlong=2):
    """Classic newspaper look for big boards: symmetric blocks, every white square in two words, few long words."""
    def runs_ok(g):                                    # no white run of length 1 or 2
        for line in g + [list(c) for c in zip(*g)]:
            if any(len(r) in (1, 2) for r in "".join("#" if x else "." for x in line).split("#")): return False
        return True
    cells = [(r, c) for r in range(n) for c in range(n)]
    for _ in range(400):
        g = [[False] * n for _ in range(n)]; count = 0
        target = rng.randint(int(n * n * 0.17), int(n * n * 0.25)); rng.shuffle(cells)
        for r, c in cells:
            if count >= target: break
            if g[r][c]: continue
            m = (n - 1 - r, n - 1 - c); add = [(r, c)] if m == (r, c) else [(r, c), m]
            for a, b in add: g[a][b] = True
            if runs_ok(g): count += len(add)
            else:
                for a, b in add: g[a][b] = False
        pat = ["".join("#" if x else "." for x in row) for row in g]
        if pattern_ok(pat, lens) and sum(len(s["cells"]) >= 7 for s in get_slots(pat)) <= maxlong: return pat

def solve(slots, by_len, used, rng, limit, strict, deadline=None):
    dom = [[w for w in by_len[len(s["cells"])] if not (strict and w in used)] for s in slots]
    where = {}
    for i, s in enumerate(slots):
        for p, c in enumerate(s["cells"]): where.setdefault(c, []).append((i, p))
    adj = [[] for _ in slots]
    for v in where.values():
        if len(v) == 2:
            (i, p), (j, q) = v; adj[i].append((p, j, q)); adj[j].append((q, i, p))

    def propagate(d, queue):                            # arc consistency; only re-checks slots whose domain changed
        while queue:
            i = queue.pop()
            for p, j, q in adj[i]:
                ok = {w[p] for w in d[i]}
                nd = [w for w in d[j] if w[q] in ok]
                if len(nd) < len(d[j]):
                    if not nd: return False
                    d[j] = nd; queue.append(j)
        return True

    if not all(dom) or not propagate(dom, list(range(len(slots)))): return None
    nodes = [0]
    def rec(d, done):
        nodes[0] += 1
        if nodes[0] > limit or (deadline and time.time() > deadline): return None
        free = [i for i in range(len(slots)) if i not in done]
        if not free: return {i: d[i][0] for i in range(len(slots))}
        i = min(free, key=lambda k: len(d[k]))           # most constrained slot first
        cands = d[i][:]; rng.shuffle(cands); cands.sort(key=lambda w: w in used)   # prefer unused words
        for w in cands:
            if deadline and time.time() > deadline: return None
            nd = list(d); nd[i] = [w]; queue = [i]; dead = False
            for j in free:                               # a word may only appear once
                if j != i and len(slots[j]["cells"]) == len(w) and w in nd[j]:
                    nd[j] = [x for x in nd[j] if x != w]
                    if not nd[j]: dead = True; break
                    queue.append(j)
            if dead or not propagate(nd, queue): continue
            r = rec(nd, done | {i})
            if r: return r
        return None
    return rec(dom, frozenset())

def generate_crossword(clues, min_size=5, max_size=6, used=(), strict=False, rng=None, attempts=300, limit=3000,
                       newspaper=False, seconds=90):
    """clues: {WORD: [clue, ...]}. Returns {"pattern": [...], "entries": [...]} or None."""
    rng = rng or random.Random(); by_len = {}
    for w in clues: by_len.setdefault(len(w), []).append(w)
    end, tries = time.time() + seconds, 0
    while (time.time() < end) if newspaper else (tries < attempts):
        tries += 1
        n = rng.randint(min_size, max_size)
        pat = (newspaper_pattern if newspaper else random_pattern)(n, set(by_len), rng)
        if not pat: continue
        S = get_slots(pat)
        sol = solve(S, by_len, set(used), rng, limit, strict, min(end, time.time() + 15) if newspaper else None)
        if sol:
            ents = []
            for i, s in enumerate(S):
                r, c = s["cells"][0]; w = sol[i]
                ents.append({"id": s["id"], "num": s["num"], "dir": s["dir"], "answer": w,
                             "clue": rng.choice(clues[w]), "r": r, "c": c, "len": len(w)})
            return {"pattern": pat, "entries": ents}

def validate(puz, clues):
    exp = {s["id"]: s for s in get_slots(puz["pattern"])}; grid, seen, errs = {}, set(), []
    if {e["id"] for e in puz["entries"]} != set(exp): errs.append("slot mismatch")
    for e in puz["entries"]:
        s, a = exp.get(e["id"]), e["answer"]
        if not s or len(a) != len(s["cells"]): errs.append(f"{e['id']} wrong length"); continue
        if e["clue"] not in clues.get(a, []): errs.append(f"{a} not in word list")
        if a in seen: errs.append(f"duplicate {a}")
        seen.add(a)
        for c, ch in zip(s["cells"], a):
            if grid.setdefault(c, ch) != ch: errs.append(f"{e['id']} crossing mismatch")
    return errs

def main():
    ap = argparse.ArgumentParser(description="Generate aleX mini crosswords (adds to puzzles.json unless --fresh)")
    ap.add_argument("--month", help="YYYY-MM (default: the month of --from, else next month)")
    ap.add_argument("--from", dest="start", help="first date to build, YYYY-MM-DD (default: start of the month)")
    ap.add_argument("--to", dest="end", help="last date to build, YYYY-MM-DD (default: end of the month)")
    ap.add_argument("--big-dates", default="", help="comma-separated dates to force as big puzzles, e.g. 2026-11-15")
    ap.add_argument("--fresh", action="store_true", help="start puzzles.json empty instead of adding to what is there")
    ap.add_argument("--words", default="words.txt")
    ap.add_argument("--out", help="default: site/puzzles.json (or ../site/puzzles.json when run from inside generator/)")
    ap.add_argument("--min-size", type=int, default=5); ap.add_argument("--max-size", type=int, default=6)
    ap.add_argument("--big-size", type=int, default=10, help="size of the big puzzles: 7-10, or 0 for none")
    ap.add_argument("--big-time", type=int, default=90, help="max seconds to spend on each big puzzle")
    ap.add_argument("--strict", action="store_true", help="never reuse a word already used (this run or history)")
    ap.add_argument("--history", default="used_words.txt"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if not 3 <= a.min_size <= a.max_size <= 6: sys.exit("Sizes must satisfy 3 <= min <= max <= 6")
    if a.big_size and not 7 <= a.big_size <= 10: sys.exit("--big-size must be 0 (off) or 7-10")
    out_path = a.out or ("site/puzzles.json" if Path("site").is_dir() else "../site/puzzles.json")
    t = date.today()
    if a.month: y, m = map(int, a.month.split("-"))
    elif a.start: y, m = int(a.start[:4]), int(a.start[5:7])
    else: y, m = (t.year + (t.month == 12), t.month % 12 + 1)
    last = calendar.monthrange(y, m)[1]
    lo = date.fromisoformat(a.start) if a.start else date(y, m, 1)
    hi = date.fromisoformat(a.end) if a.end else date(y, m, last)
    month_days = [date(y, m, d) for d in range(1, last + 1)]
    days = [d for d in month_days if lo <= d <= hi]
    if not days: sys.exit("No dates to build: check --month / --from / --to")
    forced = {x.strip() for x in a.big_dates.split(",") if x.strip()}
    for f in forced - {d.isoformat() for d in days}: print(f"  warning: --big-dates {f} is outside the dates being built; ignored")
    clues = load_words(a.words); rng = random.Random(a.seed)
    print("Words per length:", {n: sum(len(w) == n for w in clues) for n in range(MIN_LEN, MAX_LEN + 1)})
    hist = Path(a.history); used = set(hist.read_text().split()) if hist.exists() else set()
    weeks = {}
    for d in days: weeks.setdefault(d.isocalendar()[:2], []).append(d.isoformat())   # Monday-Sunday weeks
    big = set()
    if a.big_size:
        big = forced & {d.isoformat() for d in days}
        for v in weeks.values():                                                     # one random day per week,
            if not (set(v) & forced): big.add(rng.choice(v))                         # unless you already forced one
    out = {}
    for d in (x.isoformat() for x in days):
        p = None
        if d in big:
            print(f"{d}  building the {a.big_size}x{a.big_size} (up to {a.big_time}s)...", flush=True)
            p = generate_crossword(clues, a.big_size, a.big_size, used, a.strict, rng, limit=20000, newspaper=True, seconds=a.big_time)
            if not p: print(f"  could not build a {a.big_size}x{a.big_size} (word list too small?); making a mini instead")
        if not p: p = generate_crossword(clues, a.min_size, a.max_size, used, a.strict, rng)
        if not p: sys.exit(f"Could not build {d}. Add more words to {a.words} (or try --min-size 4 / drop --strict). Nothing written.")
        errs = validate(p, clues)
        if errs: sys.exit(f"Validation failed for {d}: {errs}")
        out[d] = p; used |= {e["answer"] for e in p["entries"]}
        print(f"{d}  {len(p['pattern'])}x{len(p['pattern'])}  " + " ".join(e["answer"] for e in p["entries"]))
    existing, f = {}, Path(out_path)
    if f.exists() and not a.fresh:
        try: existing = json.loads(f.read_text(encoding="utf-8")).get("puzzles", {})
        except Exception: print("  warning: existing puzzles file could not be read; starting fresh")
    existing.update(out)                                                             # replaces only the days just built
    f.write_text(json.dumps({"generated": t.isoformat(), "puzzles": dict(sorted(existing.items()))}, separators=(",", ":")))
    hist.write_text("\n".join(sorted(used)))
    print(f"Wrote {len(out)} new puzzles to {out_path} ({len(existing)} days in the file)")

if __name__ == "__main__": main()