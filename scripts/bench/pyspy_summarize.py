#!/usr/bin/env python
"""Summarise a py-spy `--format raw` (collapsed stacks) capture: top leaf frames and top inclusive frames."""
import collections, sys
leaf, incl, total = collections.Counter(), collections.Counter(), 0
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if not line or " " not in line: continue
    stack, _, n = line.rpartition(" ")
    try: n = int(n)
    except ValueError: continue
    frames = stack.split(";"); total += n; leaf[frames[-1]] += n
    for f in set(frames): incl[f] += n
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
print(f"samples {total}"); print("== top leaf frames (self time)")
for f, n in leaf.most_common(top): print(f"  {100*n/total:5.1f}%  {f[:150]}")
print("== top inclusive frames")
for f, n in incl.most_common(top): print(f"  {100*n/total:5.1f}%  {f[:150]}")
