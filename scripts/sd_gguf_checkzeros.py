#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check a GGUF for corrupt tensors (all-zero data), truncated body, or bad header.

Usage:
    python sd_gguf_checkzeros.py MODEL.gguf [--quiet]

Exit codes:
    0 = OK
    1 = warnings (e.g. some all-zero tensors, probably legit bias=0)
    2 = ERROR (truncated body / many all-zero tensors -> reconvert)

ASCII only on purpose (PS 5.1 reads .ps1 without BOM as ANSI; this is .py,
but keep the console output ASCII-safe anyway).
"""
import os
import sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in sys.argv
    if not args:
        print("usage: sd_gguf_checkzeros.py MODEL.gguf [--quiet]")
        return 2
    path = args[0]
    if not os.path.exists(path):
        print("ERROR: file not found: %s" % path)
        return 2

    try:
        import numpy as np
        import gguf
    except ImportError as e:
        print("ERROR: need python packages 'gguf' and 'numpy' (%s)" % e)
        return 2

    fsize = os.path.getsize(path)
    print("file: %s" % path)
    print("size: %.2f GiB (%d bytes)" % (fsize / (1024.0 ** 3), fsize))

    reader = gguf.GGUFReader(path)
    tensors = reader.tensors
    print("tensors: %d" % len(tensors))

    # 1) body must cover all tensors
    try:
        last_end = max(t.data_offset + int(t.n_bytes) for t in tensors)
    except AttributeError:
        last_end = max(t.data_offset + int(t.n_bytes) for t in tensors)
    if last_end > fsize:
        print("ERROR: body TRUNCATED: last tensor ends at %d, file is %d" % (last_end, fsize))
        return 2
    if not quiet:
        print("body: complete (last tensor ends at EOF)")

    # 2) all-zero tensors
    zeros = []
    per_prefix = {}
    for t in tensors:
        try:
            arr = t.data
        except Exception:
            continue
        if arr is None or arr.size == 0:
            continue
        a = arr.astype(np.float32)
        if not np.any(a):
            zeros.append(t.name)
            prefix = ".".join(t.name.split(".")[:2])
            per_prefix[prefix] = per_prefix.get(prefix, 0) + 1

    if not zeros:
        print("RESULT: OK (no all-zero tensors)")
        return 0

    pct = 100.0 * len(zeros) / max(1, len(tensors))
    print("all-zero tensors: %d (%.1f%%)" % (len(zeros), pct))
    for p, n in sorted(per_prefix.items()):
        print("  %s: %d" % (p, n))
    print("  first: %s" % zeros[:5])
    print("  last : %s" % zeros[-3:])

    # A handful of legit all-zero biases is fine; a contiguous tail is not.
    if pct > 10.0:
        print("RESULT: ERROR - file is CORRUPT (reconvert from the source checkpoint)")
        return 2
    print("RESULT: WARNING - some all-zero tensors (probably legit zero bias)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
