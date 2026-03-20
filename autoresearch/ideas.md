# Ideas for Hex Tic-Tac-Toe Bot Improvement

## Priority ideas
- [ ] Threat-space search — detect forced sequences (open 5s, double 4s, etc.)
- [ ] Opening book — precomputed first moves (center is strong)
- [~] Killer move heuristic / history heuristic for move ordering (IN PROGRESS)
- [ ] MCTS approach
- [ ] Node priors
- [ ] Consider both stones of a turn together in search
- [ ] Negamax refactor (simpler, fewer branches in code)
- [ ] Tune LINE_SCORES weights (current 10x growth may be too aggressive)
- [ ] Defensive asymmetry — weight opponent threats more than own
- [ ] Center control bonus in eval

## Tried
- [x] Heuristic eval (line-window scoring) — **100% win rate vs no-eval baseline, KEEP**
- [x] Move ordering by line-neighbor heuristic — 50%, no improvement, DISCARD
- [x] Precompute 6-cell windows — 50%, no improvement, DISCARD (marginal speedup not enough)
- [x] Narrow candidates to distance 1 — 42%, WORSE (misses important moves)
- [x] Transposition table (no root AB) — 50%, no improvement, DISCARD
- [x] TT + Zobrist + root alpha-beta fix — 51%, marginal but depth 2.5→3.2, KEEP
- [x] Precomputed windows eval (on top of TT) — 43%, WORSE despite correct values
- [x] Incremental eval (on top of TT) — 28%, MUCH WORSE despite correct values and deeper search (4.3 vs 3.4)

## Notes
- 0.5s time limit per move (changed from 1s)
- Board is 91 cells, branching factor can be huge — must prune aggressively
- Connect6 literature relevant (6-in-a-row, 2 stones per turn)
- The 2-stones-per-turn structure means threats work differently than standard connect games
- Current champion: TT + root AB, searches avg depth ~3 at 0.5s
- CRITICAL FINDING: Deeper search (depth 4+) HURTS with current eval function. The eval is too naive — it leads to search pathology at deeper levels. Improving eval QUALITY is more important than search speed.
- Distance-1 candidates search deeper (3.1) but miss critical moves — need a smarter way to narrow, not just closer.
- Move ordering overhead wasn't worth it at depth 2 — might help now with root AB and TT (champion at depth 3).
