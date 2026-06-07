"""
ocr/ocr_engine.py
Triple OCR Consensus Engine
Tesseract + EasyOCR + PaddleOCR — parallel execution, majority-vote entities
"""

import re
import os
import cv2
import numpy as np
import concurrent.futures
import pytesseract
from pathlib import Path

# Explicit Tesseract path for Windows
tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tess_path):
    pytesseract.tesseract_cmd = tess_path
    pytesseract.pytesseract.tesseract_cmd = tess_path

# ── Entity extraction helpers ────────────────────────────────────────────────

# Indian GST: 15-char alphanumeric (2 digit state + 10 PAN + 1 entity + Z + 1 check)
GST_PATTERN_INDIAN  = re.compile(
    r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b'
)
# Generic Tax-ID: US/CORD-style. Exclude dates (XX-XX-XXXX)
GST_PATTERN_GENERIC = re.compile(r'\b(?!\d{2}-\d{2}-\d{4})\d{2,4}-\d{2,4}-\d{4,6}\b')
# Invoice/Bill number
INVOICE_PATTERN     = re.compile(r'(?:Invoice|Bill|Receipt|No\.?|#)\s*[:\-]?\s*([A-Z0-9\-/]+)', re.IGNORECASE)
# Broad decimal pattern: handle spaces like '3 0 . 0 0'
DECIMAL_PATTERN     = re.compile(r'\d{1,8}\s*[.,]\s*\d{2}')


def _normalise(text: str) -> str:
    """Remove spaces and normalise commas→dots for number parsing."""
    # Remove interior spaces between digits that confuse patterns
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\d)\s+([.,])', r'\1\2', text)
    text = re.sub(r'([.,])\s+(\d)', r'\1\2', text)
    text = text.replace(',', '.')
    return text


def _preprocess_for_ocr(image_path: str):
    """Enhance image for OCR: Grayscale + Otsu Threshold + Closing."""
    img = cv2.imread(image_path)
    if img is None: return None
    
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Rescale (Upsample 2x)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    
    # 3. Otsu's Binarization (Better for clean backgrounds than Adaptive)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Morphological closing (Heal broken/dotted characters like '8')
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Save temp for OCR
    temp_path = str(Path(image_path).parent / f"pre_{Path(image_path).name}")
    cv2.imwrite(temp_path, processed)
    return temp_path


def _extract_entities_from_text(raw_text: str) -> dict:
    """
    Parse OCR text to extract financial entities and identifiers.
    Handles both Indian GST invoices and CORD/SROIE-style receipts.
    """
    norm = _normalise(raw_text)

    # ── GST / Tax ID ──
    gst = ""
    m = GST_PATTERN_INDIAN.search(norm.upper())
    if m:
        gst = m.group(0)
    else:
        m = GST_PATTERN_GENERIC.search(norm)
        if m:
            gst = m.group(0)

    # ── Invoice number ──
    invoice = ""
    m = INVOICE_PATTERN.search(raw_text)
    if m:
        invoice = m.group(1).strip()

    # ── Financial values from SUMMARY / TOTAL block ──
    net_subtotal = vat_percent = vat_amount = gross_total = None

    # VAT / Tax percent
    vp = re.search(r'(\d{1,2})\s*%', norm)
    if vp:
        vat_percent = float(vp.group(1))

    lines = norm.split('\n')
    found_totals = []
    
    for line in lines:
        line_upper = line.upper()
        # Avoid picking up Qty or Pax as financial totals
        if "QTY" in line_upper or "PAX" in line_upper or "PCS" in line_upper:
            continue
            
        nums = DECIMAL_PATTERN.findall(line)
        nums = [float(n.replace(',', '.')) for n in nums]
        if not nums: continue

        # 1. Broad "Total" capture
        if any(kw in line_upper for kw in ["TOTAL", "GROSS", "GRAND", "NET", "SUBTOTAL", "SUB-TOTAL", "INCLUSIVE", "EXCLUDING", "AMOUNT"]):
            found_totals.append(nums[-1])
            
            # Special case: line with 3+ numbers (Net, VAT, Total)
            if len(nums) >= 3 and ("TOTAL" in line_upper or "SUMMARY" in line_upper):
                net_subtotal = nums[-3]
                vat_amount   = nums[-2]
                gross_total  = nums[-1]

            # Contextual hints
            if any(kw in line_upper for kw in ["NET", "SUBTOTAL", "EXCLUDING"]):
                if net_subtotal is None: net_subtotal = nums[-1]
            if any(kw in line_upper for kw in ["VAT", "GST", "TAX", "SERVICE"]):
                if vat_amount is None: vat_amount = nums[-1]

    if found_totals:
        # Final gross is the last "total-like" value on the page
        if gross_total is None:
            gross_total = found_totals[-1]
        
        # If we have multiple different totals, that's a conflict
        found_gross_totals = found_totals
    
    # Fallback if no TOTAL block found (for cropped receipts)
    if gross_total is None:
        all_nums = DECIMAL_PATTERN.findall(norm)
        all_nums = [float(n.replace(',', '.')) for n in all_nums]
        if all_nums:
            # Assume the largest extracted decimal number is the Gross Total
            gross_total = max(all_nums)
            net_subtotal = gross_total # Default to 0 VAT
            if vat_percent and vat_percent > 0:
                net_subtotal = round(gross_total / (1 + vat_percent/100), 2)
                vat_amount = round(gross_total - net_subtotal, 2)

    return {
        "net_subtotal": net_subtotal,
        "vat_percent":  vat_percent,
        "vat_amount":   vat_amount,
        "gross_total":  gross_total,
        "gst_number":   gst,
        "invoice_number": invoice,
        "conflicting_totals": list(set(found_gross_totals)) if len(set(found_gross_totals)) > 1 else [],
    }


# ── OCR Engines ─────────────────────────────────────────────────────────────


# ── Global Singletons for OCR Models (Lazy Loading) ─────────────────────────
_EASYOCR_READER = None
_PADDLE_OCR = None

def _get_easyocr():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASYOCR_READER

def _get_paddleocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        import sys, os
        # Correct flag to skip Baidu connectivity check in paddlex (used by PaddleOCR v3.4+)
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        # PaddleOCR tries to parse sys.argv on init, which can crash in Flask
        _old_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            from paddleocr import PaddleOCR
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang='en')
        except Exception as e:
            _PADDLE_OCR = "FAILED"
            print(f"[FORENSIQ] PaddleOCR init failed: {e}")
        finally:
            sys.argv = _old_argv
    return _PADDLE_OCR


def _run_tesseract(image_path: str) -> dict:
    try:
        from PIL import Image as PILImage
        img   = PILImage.open(image_path)
        text  = pytesseract.image_to_string(img)
        entities = _extract_entities_from_text(text)
        entities["engine"] = "tesseract"
        entities["raw_text"] = text
        return entities
    except Exception as e:
        return {"engine": "tesseract", "error": str(e), "raw_text": ""}


def _run_easyocr(image_path: str) -> dict:
    try:
        reader = _get_easyocr()
        results = reader.readtext(image_path, detail=0, paragraph=True)
        text = "\n".join(results)
        entities = _extract_entities_from_text(text)
        entities["engine"] = "easyocr"
        entities["raw_text"] = text
        return entities
    except Exception as e:
        return {"engine": "easyocr", "error": str(e), "raw_text": ""}


def _run_paddleocr(image_path: str) -> dict:
    try:
        ocr = _get_paddleocr()
        if ocr == "FAILED":
            return {"engine": "paddleocr",
                    "error": "PaddleOCR unavailable (model init failed — offline mode)",
                    "raw_text": ""}
        result = ocr.ocr(image_path, cls=True)
        lines = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    lines.append(line[1][0])
        text = "\n".join(lines)
        entities = _extract_entities_from_text(text)
        entities["engine"] = "paddleocr"
        entities["raw_text"] = text
        return entities
    except Exception as e:
        return {"engine": "paddleocr", "error": str(e), "raw_text": ""}


# ── Majority Vote ────────────────────────────────────────────────────────────

def _majority_vote(results: list, field: str):
    """
    Given 3 engine results, return majority-voted value for a field.
    Returns (value, agreement_count).
    """
    values = [r.get(field) for r in results if r.get(field) is not None]
    if not values:
        return None, 0

    # For numeric fields: round to 2dp then vote
    if isinstance(values[0], float):
        rounded = [round(v, 1) for v in values]
        from collections import Counter
        counts = Counter(rounded)
        best_val, best_cnt = counts.most_common(1)[0]
        # Return the original un-rounded value closest to winner
        for v in values:
            if round(v, 1) == best_val:
                return round(v, 2), best_cnt
        return round(best_val, 2), best_cnt

    # For strings: exact match vote
    from collections import Counter
    counts = Counter(values)
    best_val, best_cnt = counts.most_common(1)[0]
    return best_val, best_cnt


def _compute_agreement(results: list) -> tuple:
    """
    Compute overall agreement level across the 3 engines.
    Returns (agreement: str, ocr_confidence: float)
    """
    fields = ["net_subtotal", "vat_amount", "gross_total", "gst_number"]
    agreements = []
    for f in fields:
        values = [r.get(f) for r in results if r.get(f) is not None]
        if len(values) < 2:
            agreements.append(0)
        elif len(values) == 3:
            if values[0] == values[1] == values[2]:
                agreements.append(3)
            elif values[0] == values[1] or values[1] == values[2] or values[0] == values[2]:
                agreements.append(2)
            else:
                agreements.append(1)
        else:
            if values[0] == values[1]:
                agreements.append(2)
            else:
                agreements.append(1)

    avg = sum(agreements) / len(agreements) if agreements else 1

    if avg >= 2.8:
        return "full",     1.00
    elif avg >= 1.8:
        return "majority", 0.67
    else:
        return "split",    0.33


# ── Main Entry Point ─────────────────────────────────────────────────────────

def run_triple_ocr(image_path: str) -> dict:
    """
    Run Tesseract, EasyOCR, and PaddleOCR in parallel.
    Returns majority-voted entities + agreement metadata.

    Output:
        {
            "net_subtotal":   float | None,
            "vat_percent":    float | None,
            "vat_amount":     float | None,
            "gross_total":    float | None,
            "gst_number":     str,
            "invoice_number": str,
            "agreement":      "full" | "majority" | "split",
            "ocr_confidence": float (1.0 / 0.67 / 0.33),
            "engine_results": {
                "tesseract": {...},
                "easyocr":   {...},
                "paddleocr": {...}
            }
        }
    """
    # PRE-INIT EASYOCR to avoid PyTorch threading deadlocks on Windows
    try:
        _get_easyocr()
    except Exception as e:
        print(f"[FORENSIQ] Pre-init EasyOCR failed: {e}")

    # 0. Pre-process image for better contrast/clarity
    processed_path = _preprocess_for_ocr(image_path) or image_path

    # Run all 3 engines in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "tesseract": executor.submit(_run_tesseract, processed_path),
            "easyocr":   executor.submit(_run_easyocr,   processed_path),
            "paddleocr": executor.submit(_run_paddleocr,  processed_path),
        }
        engine_results = {name: f.result() for name, f in futures.items()}

    # Cleanup temp file
    if processed_path != image_path and Path(processed_path).exists():
        try: os.remove(processed_path)
        except: pass

    results_list = [r for r in engine_results.values() if not r.get("error")]

    # Majority vote on each field
    fields = ["net_subtotal", "vat_percent", "vat_amount", "gross_total",
              "gst_number", "invoice_number"]
    merged = {}
    for f in fields:
        val, cnt = _majority_vote(results_list, f)
        merged[f] = val

    # Agreement level
    agreement, confidence = _compute_agreement(results_list)

    # Aggregated conflicting totals (any engine finding a conflict is suspicious)
    conflicting = []
    for r in results_list:
        conflicting.extend(r.get("conflicting_totals", []))
    conflicting = sorted(list(set(conflicting)))

    # SYSTEM FAILURE check: If no engine succeeded or all returned empty text
    system_failure = False
    if not results_list or all(not r.get("raw_text", "").strip() for r in results_list):
        system_failure = True

    return {
        "net_subtotal":   merged.get("net_subtotal"),
        "vat_percent":    merged.get("vat_percent"),
        "vat_amount":     merged.get("vat_amount"),
        "gross_total":    merged.get("gross_total"),
        "gst_number":     merged.get("gst_number") or "",
        "invoice_number": merged.get("invoice_number") or "",
        "agreement":      agreement,
        "ocr_confidence": confidence,
        "conflicting_totals": conflicting,
        "system_failure": system_failure,
        "engine_results": engine_results,   # individual engine outputs for UI cards
    }
