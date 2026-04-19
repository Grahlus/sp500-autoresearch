#!/usr/bin/env python3
"""
Reset stale confirmation state in experiments/memory.json.

Actions taken:
  1. Clear families.momentum.best_config_hash  → null
  2. Clear families.momentum.best_objective_score → null
  3. Delete promotion_states.momentum["d200e40d61a7e10f"] (stale provisional entry)

Safety: backs up memory.json, then writes atomically via tempfile + os.replace.

Run with the autoresearch loop stopped (or during its sleep window):
    uv run python scripts/reset_confirmation_state.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)

MEMORY_PATH = _ROOT / "experiments" / "memory.json"
BACKUP_DIR = _ROOT / "experiments" / "backups"

STALE_CONFIG = "6ea03c3e1ec3f5be"
STALE_PROVISIONAL = "d200e40d61a7e10f"

# ── Backup ────────────────────────────────────────────────────────────────────

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
backup_path = BACKUP_DIR / f"memory_pre_reset_{ts}.json"
shutil.copy2(MEMORY_PATH, backup_path)
print(f"Backed up memory.json → {backup_path}  ({backup_path.stat().st_size:,} bytes)")

# ── Load ──────────────────────────────────────────────────────────────────────

with open(MEMORY_PATH, encoding="utf-8") as fh:
    memory = json.load(fh)

# ── Report before ─────────────────────────────────────────────────────────────

fam_before = (memory.get("families") or {}).get("momentum") or {}
ps_before   = (memory.get("promotion_states") or {}).get("momentum") or {}
print("\n=== BEFORE ===")
print(f"  families.momentum.best_config_hash   : {fam_before.get('best_config_hash')!r}")
print(f"  families.momentum.best_objective_score: {fam_before.get('best_objective_score')!r}")
print(f"  promotion_states.momentum keys       : {list(ps_before.keys())}")
if STALE_PROVISIONAL in ps_before:
    prov = ps_before[STALE_PROVISIONAL]
    print(f"    [{STALE_PROVISIONAL}] state={prov.get('promotion_state')!r}"
          f"  conf_req={prov.get('confirmation_required')!r}")

# ── Mutate ────────────────────────────────────────────────────────────────────

changed = False

# 1. Clear stale best_config_hash
if memory.get("families") and memory["families"].get("momentum"):
    fam = memory["families"]["momentum"]
    if fam.get("best_config_hash") == STALE_CONFIG:
        fam["best_config_hash"] = None
        fam["best_objective_score"] = None
        print(f"\nCleared families.momentum.best_config_hash ({STALE_CONFIG} → null)")
        print(f"Cleared families.momentum.best_objective_score → null")
        changed = True
    else:
        print(f"\nNOTE: best_config_hash is {fam.get('best_config_hash')!r} (not {STALE_CONFIG}), leaving untouched")

# 2. Remove stale provisional promotion entry
if memory.get("promotion_states") and memory["promotion_states"].get("momentum"):
    ps = memory["promotion_states"]["momentum"]
    if STALE_PROVISIONAL in ps:
        removed = ps.pop(STALE_PROVISIONAL)
        print(f"Removed promotion_states.momentum[{STALE_PROVISIONAL!r}]"
              f" (state={removed.get('promotion_state')!r})")
        changed = True
    else:
        print(f"NOTE: {STALE_PROVISIONAL} not found in promotion_states.momentum — already gone")

if not changed:
    print("\nNo changes needed. Exiting.")
    sys.exit(0)

# ── Atomic write ─────────────────────────────────────────────────────────────

with tempfile.NamedTemporaryFile(
    "w", delete=False,
    dir=MEMORY_PATH.parent,
    prefix="memory.json.", suffix=".tmp",
    encoding="utf-8",
) as fh:
    json.dump(memory, fh, indent=2, ensure_ascii=False)
    tmp_path = fh.name
os.replace(tmp_path, str(MEMORY_PATH))
print(f"\nWrote: {MEMORY_PATH}  ({MEMORY_PATH.stat().st_size:,} bytes)")

# ── Verify ────────────────────────────────────────────────────────────────────

with open(MEMORY_PATH, encoding="utf-8") as fh:
    verify = json.load(fh)

fam_after = (verify.get("families") or {}).get("momentum") or {}
ps_after  = (verify.get("promotion_states") or {}).get("momentum") or {}

print("\n=== AFTER ===")
print(f"  families.momentum.best_config_hash   : {fam_after.get('best_config_hash')!r}")
print(f"  families.momentum.best_objective_score: {fam_after.get('best_objective_score')!r}")
print(f"  promotion_states.momentum keys       : {list(ps_after.keys())}")

assert fam_after.get("best_config_hash") is None,  "best_config_hash not null after reset!"
assert fam_after.get("best_objective_score") is None, "best_objective_score not null after reset!"
assert STALE_PROVISIONAL not in ps_after, f"{STALE_PROVISIONAL} still present after reset!"
print("\n✓ All assertions passed — confirmation state reset complete.")
