# Optimization Ideas

## Speed ideas (same output, faster search — verify with test_correctness.py)

- [ ] Incremental hot window set (count ≥ 4) to speed up _find_instant_win and _find_threat_cells
- [ ] Profile to find actual bottleneck (run test_profile.py)
- [ ] Negamax refactor to eliminate duplicated max/min branches
- [ ] Cache _move_delta results per position to avoid recomputing across turns
- [ ] Avoid closure/lambda in candidate sort — use inline key or precomputed list
- [ ] Replace `list(combinations(...))` with a generator to avoid allocating full list
- [ ] Precompute and cache frequently accessed data structures
- [ ] Reduce object allocations in hot loops (reuse lists, avoid dict copies)
- [ ] Optimize _candidates() — minimize set operations and sorting overhead
- [ ] Inline small helper functions that are called millions of times
- [ ] Convert score table and window counts to flat arrays/tuples instead of lists-of-lists
- [ ] Precompute _WINDOW_OFFSETS as a flat tuple to avoid unpacking overhead
- [ ] Inline _make/_undo hot path — avoid method call overhead in tight loop
- [ ] Use __slots__ on the bot class to speed up attribute access
- [ ] Batch Zobrist key generation — preallocate a larger table instead of lazy per-cell generation

## Hyperparameter tuning ideas (change values, measure win rate)

- [ ] Tune LINE_SCORES values
    - [ ] Try linear, not exponential, or other functions
    - [ ] Try 5 more similar to four
    - [ ] Generally tune each number relative to others
- [ ] Tune _DEF_MULT values
    - [ ] Try no def mult
- [ ] Tune _CANDIDATE_CAP and _ROOT_CANDIDATE_CAP
- [ ] Tune _DELTA_WEIGHT
- [ ] Tune phase-aware eval thresholds and multipliers
