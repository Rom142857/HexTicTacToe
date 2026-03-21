"""Pattern table for 6-cell hex window evaluation with symmetry reduction.

Each window is 6 cells along a hex axis. Cell states: 0=empty, 1=A, 2=B.
Symmetries:
  - Flipping (reverse): same value
  - Piece swap (1<->2): negated value
"""

WINDOW_LENGTH = 6
NUM_PATTERNS = 3 ** WINDOW_LENGTH  # 6561 including all-empty


def _swap_piece(cell):
    """Swap player A (1) <-> player B (2), leave empty (0) alone."""
    if cell == 1:
        return 2
    if cell == 2:
        return 1
    return 0


def _int_to_pattern(n):
    """Convert integer [0, 6561) to 8-cell pattern tuple."""
    pat = []
    for _ in range(WINDOW_LENGTH):
        pat.append(n % 3)
        n //= 3
    return tuple(pat)


def _pattern_to_int(pat):
    """Convert 8-cell pattern tuple to integer [0, 6561)."""
    n = 0
    for i in range(WINDOW_LENGTH - 1, -1, -1):
        n = n * 3 + pat[i]
    return n


def _flip(pat):
    return pat[::-1]


def _swap(pat):
    return tuple(_swap_piece(c) for c in pat)


def build_tables():
    """Build the canonical pattern table.

    Returns:
        canon_patterns: list of canonical pattern tuples (the learnable set)
        pattern_map: dict mapping pattern_int -> (canon_index, sign)
            sign is +1 or -1 (piece swap negates)
            canon_index is -1 for all-empty or self-symmetric (forced zero)
    """
    canon_patterns = []
    canon_lookup = {}  # canonical pattern tuple -> index
    pattern_map = {}   # pattern_int -> (canon_index, sign)

    for i in range(NUM_PATTERNS):
        if i in pattern_map:
            continue

        pat = _int_to_pattern(i)

        # Skip all-empty
        if all(c == 0 for c in pat):
            pattern_map[i] = (-1, 0)
            continue

        # Generate equivalence class
        p_flip = _flip(pat)
        p_swap = _swap(pat)
        p_swap_flip = _flip(p_swap)

        # Flip-canonical for original side
        fc_orig = min(pat, p_flip)
        # Flip-canonical for swapped side
        fc_swap = min(p_swap, p_swap_flip)

        # Self-symmetric check: if fc_orig == fc_swap, value must be 0
        if fc_orig == fc_swap:
            for p in (pat, p_flip, p_swap, p_swap_flip):
                pi = _pattern_to_int(p)
                if pi not in pattern_map:
                    pattern_map[pi] = (-1, 0)
            continue

        # Pick canonical as the smaller flip-canonical
        if fc_orig < fc_swap:
            canon = fc_orig
            pos_pats = (pat, p_flip)
            neg_pats = (p_swap, p_swap_flip)
        else:
            canon = fc_swap
            pos_pats = (p_swap, p_swap_flip)
            neg_pats = (pat, p_flip)

        # Register canonical pattern if new
        if canon not in canon_lookup:
            canon_lookup[canon] = len(canon_patterns)
            canon_patterns.append(canon)
        cidx = canon_lookup[canon]

        # Map all variants
        for p in pos_pats:
            pi = _pattern_to_int(p)
            if pi not in pattern_map:
                pattern_map[pi] = (cidx, 1)
        for p in neg_pats:
            pi = _pattern_to_int(p)
            if pi not in pattern_map:
                pattern_map[pi] = (cidx, -1)

    return canon_patterns, pattern_map


# Pre-compute on import
CANON_PATTERNS, PATTERN_MAP = build_tables()
NUM_CANON = len(CANON_PATTERNS)

# Fast lookup array: index by pattern_int -> (canon_index, sign)
# For patterns not in map (shouldn't happen), default to (-1, 0)
CANON_INDEX = [0] * NUM_PATTERNS
CANON_SIGN = [0] * NUM_PATTERNS
for _pi, (_ci, _s) in PATTERN_MAP.items():
    CANON_INDEX[_pi] = _ci
    CANON_SIGN[_pi] = _s


def pattern_to_int(pat):
    return _pattern_to_int(pat)


if __name__ == "__main__":
    print(f"Total non-empty patterns: {NUM_PATTERNS - 1}")
    print(f"Canonical learnable patterns: {NUM_CANON}")
    zero_count = sum(1 for ci, s in PATTERN_MAP.values() if s == 0 and ci == -1) - 1  # minus all-empty
    print(f"Self-symmetric (forced zero): {zero_count}")
    covered = sum(1 for ci, s in PATTERN_MAP.values() if s != 0)
    print(f"Patterns with sign: {covered}")
    print(f"Check: {covered} + {zero_count} + 1 (empty) = {covered + zero_count + 1} (should be {NUM_PATTERNS})")
