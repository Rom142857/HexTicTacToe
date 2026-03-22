"""Pattern table for N-cell hex window evaluation with symmetry reduction.

Each window is N cells along a hex axis. Cell states: 0=empty, 1=current player, 2=opponent.
Symmetries:
  - Flipping (reverse): same value
  - NO piece swap symmetry — offense and defense can have different weights
"""

WINDOW_LENGTH = 6
NUM_PATTERNS = 3 ** WINDOW_LENGTH


def _int_to_pattern(n, wl):
    pat = []
    for _ in range(wl):
        pat.append(n % 3)
        n //= 3
    return tuple(pat)


def _pattern_to_int(pat):
    n = 0
    for i in range(len(pat) - 1, -1, -1):
        n = n * 3 + pat[i]
    return n


def build_tables(wl=None):
    """Build the canonical pattern table for a given window length.

    Returns:
        canon_patterns: list of canonical pattern tuples (the learnable set)
        pattern_map: dict mapping pattern_int -> (canon_index, sign)
    """
    if wl is None:
        wl = WINDOW_LENGTH
    num_patterns = 3 ** wl
    canon_patterns = []
    canon_lookup = {}
    pattern_map = {}

    for i in range(num_patterns):
        if i in pattern_map:
            continue

        pat = _int_to_pattern(i, wl)

        if all(c == 0 for c in pat):
            pattern_map[i] = (-1, 0)
            continue

        p_flip = pat[::-1]
        canon = min(pat, p_flip)

        if canon not in canon_lookup:
            canon_lookup[canon] = len(canon_patterns)
            canon_patterns.append(canon)
        cidx = canon_lookup[canon]

        for p in (pat, p_flip):
            pi = _pattern_to_int(p)
            if pi not in pattern_map:
                pattern_map[pi] = (cidx, 1)

    return canon_patterns, pattern_map


def build_arrays(wl=None):
    """Build tables and return flat arrays for fast lookup.

    Returns (canon_patterns, canon_index, canon_sign, num_canon, num_patterns).
    """
    if wl is None:
        wl = WINDOW_LENGTH
    num_patterns = 3 ** wl
    canon_patterns, pattern_map = build_tables(wl)
    canon_index = [0] * num_patterns
    canon_sign = [0] * num_patterns
    for pi, (ci, s) in pattern_map.items():
        canon_index[pi] = ci
        canon_sign[pi] = s
    return canon_patterns, canon_index, canon_sign, len(canon_patterns), num_patterns


# Pre-compute default (length 6) on import
CANON_PATTERNS, PATTERN_MAP = build_tables()
NUM_CANON = len(CANON_PATTERNS)

CANON_INDEX = [0] * NUM_PATTERNS
CANON_SIGN = [0] * NUM_PATTERNS
for _pi, (_ci, _s) in PATTERN_MAP.items():
    CANON_INDEX[_pi] = _ci
    CANON_SIGN[_pi] = _s


def pattern_to_int(pat):
    return _pattern_to_int(pat)


if __name__ == "__main__":
    for wl in [6, 7, 8]:
        cp, _, _, nc, np_ = build_arrays(wl)
        print(f"Window length {wl}: {np_} patterns, {nc} canonical")
