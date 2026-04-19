"""Layer 2 periodic planner — advisory only.

This package provides the high-level MiniMax-based planning layer.
It is strictly non-authoritative: it reads official experiment state and
writes structured plan files, but never launches experiments, writes official
results, modifies the index, or bypasses any execution guard.

The deterministic executor (autonomous_runner.py) remains the sole authority
for all experiment execution and official state mutations.
"""
