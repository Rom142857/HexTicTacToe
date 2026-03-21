---
name: no_cd_commands
description: Never use cd && prefixes for any commands — run everything from autoresearch/ subdirectory
type: feedback
---

NEVER use `cd .. &&` or `cd ../` prefixes for any command. Run git from the subdirectory, run python with `../` paths.

**Why:** User is very frustrated by this pattern — it violates the CLAUDE.md rules explicitly.

**How to apply:** All bash commands run from `autoresearch/`. Git works from any subdirectory. Python scripts use `../` prefix paths.
