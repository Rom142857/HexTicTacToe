---
name: big_changes_first
description: Try bigger structural changes (function type, algorithm shape) before fine-tuning specific values
type: feedback
---

Try bigger changes before smaller ones. For example, try changing the overall function type (linear, exponential, quadratic, etc.) before adjusting specific values.

**Why:** Fine-tuning individual values is slow and unlikely to find big gains. Structural changes can discover fundamentally better scoring shapes.

**How to apply:** When tuning hyperparameters, start with broad structural experiments (change the curve shape, remove/add entire components) before zooming into specific numeric adjustments.
