"""
scripts/test_images_quick.py
Runs the FULL FORENSIQ pipeline on all images in TEST/ folder.
Prints a formatted table of results — no Flask needed.
Usage: python scripts/test_images_quick.py
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "TEST"
sys.path.insert(0, str(BASE_DIR))
os.environ["TF_CPP_MIN_LOG_LEVEL"]                 = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"]                = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"]= "True"

from vision.vision_model         import run_visual_forensics
from ocr.ocr_engine              import run_triple_ocr
from classifier.fraud_classifier import adaptive_fusion
from utils.vendor_db             import lookup_vendor_by_gst

images = sorted([p for p in TEST_DIR.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}])

if not images:
    print("No images found in TEST/")
    sys.exit(1)

print()
print("╔══════════════════════════════════════════════════════════════════╗")
print("║         FORENSIQ — Quick TEST Inference                         ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print(f"  Found {len(images)} image(s) in TEST/\n")

# Header
print(f"  {'File':<30} {'Verdict':<12} {'Score':>6}  {'Visual':>7}  {'OCR Agr':>7}  {'Time':>6}")
print("  " + "─" * 72)

for img_path in images:
    t0 = time.time()
    try:
        visual = run_visual_forensics(str(img_path))
        ocr    = run_triple_ocr(str(img_path))
        gst    = ocr.get("gst_number", "")
        vendor = lookup_vendor_by_gst(gst) if gst else None
        fusion = adaptive_fusion(visual, ocr, vendor)

        verdict      = fusion["verdict"]
        score        = fusion["fraud_score"]
        visual_score = fusion["visual_score"]
        agreement    = fusion["agreement"]
        elapsed      = round(time.time() - t0, 1)

        # Color-coded verdict marker
        marker = "✅" if verdict == "GENUINE" else ("⚠️ " if verdict == "SUSPICIOUS" else "🔴")

        print(f"  {img_path.name:<30} {marker}{verdict:<10} {score:>6.3f}  {visual_score:>7.3f}  {agreement:>7}  {elapsed:>5.1f}s")

        # Show which OCR engines worked
        er = ocr.get("engine_results", {})
        engines_ok = []
        for eng in ["tesseract", "easyocr", "paddleocr"]:
            if er.get(eng, {}).get("error"):
                engines_ok.append(f"  {eng}: ❌ ({er[eng]['error'][:40]}...)")
            else:
                engines_ok.append(f"  {eng}: ✅")
        for e in engines_ok:
            print(f"    {e}")

    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"  {img_path.name:<30} {'ERROR':<12} {'—':>6}  {'—':>7}  {'—':>7}  {elapsed:>5.1f}s")
        print(f"    ⚠️  {e}")

print("  " + "─" * 72)
print()
print("  Done. Start Flask (python app.py) for full UI demo.")
print()
