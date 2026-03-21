---
name: background_evals
description: Run evaluations in the background to avoid hitting time limits on bash commands
type: feedback
---

Run evaluation commands with `run_in_background: true` to avoid hitting the bash time limit.

**Why:** Evaluations take 5-10 minutes and can exceed the default bash timeout.

**How to apply:** Use `run_in_background` for all `run_eval.py` and `test_profile.py --competition` commands.
