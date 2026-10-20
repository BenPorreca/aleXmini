#!/usr/bin/env python3
"""aleX mini generator. Builds a month of crosswords (up to 6x6) from words.txt into puzzles.json.

Method: crossword filling is a constraint-satisfaction problem (CSP). Each word slot is a variable, its domain is
the words of that length, and crossing slots must agree on the shared letter. Search = backtracking with
  * minimum-remaining-values (fill the most constrained slot first),
  * arc consistency (after every choice, prune letters that can no longer cross),
  * all-different words, and randomised value order with a node limit + restarts (new grid pattern) to avoid dead ends.
Grid patterns are random 180-degree-symmetric black-square layouts; finished puzzles are re-validated independently.

Usage:  python generate.py --month 2026-11
"""
import argparse, calendar, json, random, sys
from datetime import date
from pathlib import Path

MIN_LEN, MAX_LEN = 3, 6

def load_words(path):
    clues = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        w, _, c = line.partition("|"); w, c = w.strip().upper(), c.strip()
        if w.isascii() and w.isalpha() and MIN_LEN <= len(w) <= MAX_LEN and c: clues.setdefault(w, []).append(c)
        else: print(f"  skipped (needs 'WORD | clue', 3-6 letters): {line}")
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

def solve(slots, by_len, used, rng, limit, strict):
    dom = [[w for w in by_len[len(s["cells"])] if not (strict and w in used)] for s in slots]
    where = {}
    for i, s in enumerate(slots):
        for p, c in enumerate(s["cells"]): where.setdefault(c, []).append((i, p))
    cross = [v for v in where.values() if len(v) == 2]

    def propagate(d):                                   # arc consistency on every crossing
        if not all(d): return False
        changed = True
        while changed:
            changed = False
            for (i, p), (j, q) in cross:
                ok = {w[p] for w in d[i]} & {w[q] for w in d[j]}
                for k, x in ((i, p), (j, q)):
                    nd = [w for w in d[k] if w[x] in ok]
                    if len(nd) < len(d[k]):
                        d[k] = nd; changed = True
                        if not nd: return False
        return True

    if not propagate(dom): return None
    nodes = [0]
    def rec(d, done):
        nodes[0] += 1
        if nodes[0] > limit: return None
        free = [i for i in range(len(slots)) if i not in done]
        if not free: return {i: d[i][0] for i in range(len(slots))}
        i = min(free, key=lambda k: len(d[k]))           # most constrained slot first
        cands = d[i][:]; rng.shuffle(cands); cands.sort(key=lambda w: w in used)   # prefer unused words
        for w in cands:
            nd = list(d); nd[i] = [w]
            for j in free:                               # a word may only appear once
                if j != i: nd[j] = [x for x in nd[j] if x != w]
            if propagate(nd):
                r = rec(nd, done | {i})
                if r: return r
        return None
    return rec(dom, frozenset())

def generate_crossword(clues, min_size=5, max_size=6, used=(), strict=False, rng=None, attempts=300, limit=3000):
    """clues: {WORD: [clue, ...]}. Returns {"pattern": [...], "entries": [...]} or None."""
    rng = rng or random.Random()
    by_len = {}
    for w in clues: by_len.setdefault(len(w), []).append(w)
    for _ in range(attempts):
        pat = random_pattern(rng.randint(min_size, max_size), set(by_len), rng)
        if not pat: continue
        S = get_slots(pat)
        sol = solve(S, by_len, set(used), rng, limit, strict)
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
    ap = argparse.ArgumentParser(description="Generate a month of aleX mini crosswords")
    ap.add_argument("--month", help="YYYY-MM (default: next month)")
    ap.add_argument("--words", default="words.txt"); ap.add_argument("--out", default="../site/puzzles.json")
    ap.add_argument("--min-size", type=int, default=5); ap.add_argument("--max-size", type=int, default=6)
    ap.add_argument("--strict", action="store_true", help="never reuse a word already used (this run or history)")
    ap.add_argument("--history", default="used_words.txt"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if not 3 <= a.min_size <= a.max_size <= 6: sys.exit("Sizes must satisfy 3 <= min <= max <= 6")
    t = date.today()
    y, m = map(int, a.month.split("-")) if a.month else (t.year + (t.month == 12), t.month % 12 + 1)
    clues = load_words(a.words); rng = random.Random(a.seed)
    print("Words per length:", {n: sum(len(w) == n for w in clues) for n in range(MIN_LEN, MAX_LEN + 1)})
    hist = Path(a.history); used = set(hist.read_text().split()) if hist.exists() else set()
    out = {}
    for day in range(1, calendar.monthrange(y, m)[1] + 1):
        d = date(y, m, day).isoformat()
        p = generate_crossword(clues, a.min_size, a.max_size, used, a.strict, rng)
        if not p: sys.exit(f"Could not build {d}. Add more words to {a.words} (or try --min-size 4 / drop --strict). Nothing written.")
        errs = validate(p, clues)
        if errs: sys.exit(f"Validation failed for {d}: {errs}")
        out[d] = p; used |= {e["answer"] for e in p["entries"]}
        print(f"{d}  {len(p['pattern'])}x{len(p['pattern'])}  " + " ".join(e["answer"] for e in p["entries"]))
    Path(a.out).write_text(json.dumps({"generated": t.isoformat(), "month": f"{y}-{m:02d}", "puzzles": out}, separators=(",", ":")))
    hist.write_text("\n".join(sorted(used)))
    print(f"Wrote {len(out)} puzzles to {a.out}")

if __name__ == "__main__": main()
