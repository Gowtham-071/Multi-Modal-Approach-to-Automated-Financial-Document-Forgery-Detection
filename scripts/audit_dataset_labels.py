"""
scripts/audit_dataset_labels.py
Audits Main_Dataset/ for label crossover (same filename in both fraud/ and genuine/).
Usage: python scripts/audit_dataset_labels.py
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET  = BASE_DIR / "Main_Dataset"

def audit_split(split_name: str):
    split_dir = DATASET / split_name
    fraud_dir   = split_dir / "fraud"
    genuine_dir = split_dir / "genuine"

    if not fraud_dir.exists() or not genuine_dir.exists():
        print(f"  [{split_name}] ⚠️  Missing fraud/ or genuine/ subfolder — skipping")
        return

    fraud_files   = set(p.name for p in fraud_dir.iterdir() if p.is_file())
    genuine_files = set(p.name for p in genuine_dir.iterdir() if p.is_file())
    overlap       = fraud_files & genuine_files

    total_fraud   = len(fraud_files)
    total_genuine = len(genuine_files)

    print(f"\n  ── {split_name.upper()} ──")
    print(f"    Fraud   images : {total_fraud}")
    print(f"    Genuine images : {total_genuine}")
    print(f"    Total          : {total_fraud + total_genuine}")

    if overlap:
        print(f"\n  🔴 LABEL CROSSOVER DETECTED — {len(overlap)} file(s) in BOTH folders:")
        for name in sorted(overlap):
            print(f"      ⚠️  {name}")
    else:
        print(f"    ✅ No crossover — labels are clean")


print()
print("╔═══════════════════════════════════════════════════╗")
print("║     FORENSIQ — Dataset Label Audit               ║")
print("╚═══════════════════════════════════════════════════╝")

for split in ["train", "val", "test"]:
    audit_split(split)

print()
print("Audit complete.")
print()
