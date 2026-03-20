# Ideas for Hex Tic-Tac-Toe Bot Improvement

## Priority ideas
- [ ] Threat-space search — detect forced sequences (open 5s, double 4s, etc.)
- [ ] Opening book — precomputed first moves (center is strong)
- [ ] Killer move heuristic / history heuristic for move ordering
- [ ] MCTS approach
- [ ] Node priors
- [ ] Consider both stones of a turn together in search
- [ ] Transposition table with Zobrist hashing
- [ ] Incremental eval (update score on make/undo instead of full rescan)
- [ ] Negamax refactor (simpler, fewer branches in code)

## Tried
- [x] Heuristic eval (line-window scoring) — **100% win rate vs no-eval baseline, KEEP**
- [x] Move ordering by line-neighbor heuristic — 50%, no improvement, DISCARD
- [x] Precompute 6-cell windows — 50%, no improvement, DISCARD (marginal speedup not enough)
- [x] Narrow candidates to distance 1 — 42%, WORSE (misses important moves)

## Notes
- 0.5s time limit per move
- Board is 91 cells, branching factor can be huge — must prune aggressively
- Connect6 literature relevant (6-in-a-row, 2 stones per turn)
- The 2-stones-per-turn structure means threats work differently than standard connect games
- Current champion searches avg depth ~2 with distance-2 candidates. The eval is the bottleneck.
- Distance-1 candidates search deeper (3.1) but miss critical moves — need a smarter way to narrow, not just closer.
- Move ordering overhead wasn't worth it at depth 2 — might help once we search deeper.
