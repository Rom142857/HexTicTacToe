# Ideas for Hex Tic-Tac-Toe Bot Improvement

## Priority ideas
- [ ] Add heuristic evaluation (line counting, threat detection) — the bot currently has NO eval, so it plays randomly at shallow depths
- [ ] Better candidate generation — prioritize cells near existing stones, limit branching factor
- [ ] Move ordering — try winning/threatening moves first for better alpha-beta pruning
- [ ] Threat-space search — detect forced sequences (open 5s, double 4s, etc.)
- [ ] Opening book — precomputed first moves (center is strong)
- [ ] Pattern-based eval — score based on contiguous groups along each axis
- [ ] Killer move heuristic / history heuristic for move ordering
- [ ] Narrow candidates aggressively (distance 1 instead of 2, or top-N by heuristic)
- [ ] MCTS approach
- [ ] Node priors
- [ ] Consider both stones of a turn together in search

## Tried
(none yet)

## Notes
- 50ms time limit is very tight — need fast eval and narrow search
- Board is 91 cells, branching factor can be huge — must prune aggressively
- Connect6 literature relevant (6-in-a-row, 2 stones per turn)
- The 2-stones-per-turn structure means threats work differently than standard connect games
