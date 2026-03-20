# Ideas for Hex Tic-Tac-Toe Bot Improvement

## Current champion config
- LINE_SCORES: [0, 1, 10, 200, 1000, 10000, 100000]
- _DEF_MULT: [0, 0.8, 0.8, 1.2, 1.5, 3.0, 1.0]
- Phase-aware eval: 1.5x boost for 4+ windows when move_count > 10
- Time check: every 512 nodes
- Precomputed d2 offsets

## Remaining ideas to try
- [ ] Threat-space search — detect forced sequences
- [ ] MCTS approach (radically different search)
- [ ] Consider both stones of a turn together in search
- [ ] Different TT replacement policy (depth-preferred vs always-replace)
- [ ] Null move heuristic (adapted for 2-stone turns)

## Kept improvements (in order)
1. Heuristic eval — 100% win rate
2. TT + Zobrist + root AB — 51%
3. Defensive asymmetry 1.2x — 65%
4. Precomputed d2 offsets — 51% (simplification)
5. Time check 512 — 77%
6. Nonlinear _DEF_MULT — 54% (35-0 decisive)
7. Reduced 1/2 defense to 0.8 — 70% decisive
8. 5-in-a-row defense to 3.0 — 63% decisive
9. 3-in-a-row score to 200 — 52% (14-0 decisive)
10. Phase-aware eval (4+ boost in late game) — 74% decisive

## Key insights
- **Defense > offense**: Asymmetric defensive weighting is the biggest eval win
- **Time check 512**: Exploits overshoot for free search time. 1024 causes violations.
- **Speed optimizations are counterproductive** with 512 time check — faster eval = less overshoot = worse
- **Python built-ins >> Python loops**: .count() (C) beats manual counting
- **Negamax incompatible** with asymmetric eval (requires symmetric)
- **Random shuffle matters**: deterministic ordering makes bot predictable and exploitable
- **Phase-aware eval works**: boosting 4+ threats in late game converts more wins
- Champion draws 92-96% at 0.1s. Incremental eval tuning has diminishing returns.
- Next step likely requires a fundamentally different search strategy (MCTS, threat-space) to break the draw ceiling.
