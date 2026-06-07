<div align="center">

# 🔍 FORENSIQ
### Multimodal Financial Document Forgery Detection

*A 4-stream AI pipeline for real-time fraud detection on financial bills & invoices*

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

</div>

---

## Overview

**FORENSIQ** is an end-to-end multimodal fraud detection system that authenticates financial documents (GST bills, invoices, receipts) using a 4-stream processing pipeline. It combines deep learning visual forensics, triple-engine OCR consensus, arithmetic semantic validation, and a vendor context registry to catch tampering, forgery, and document anomalies — all through a clean web dashboard.

> Built as a capstone research project. The underlying research paper is unpublished and not distributed with this repository.

---

## System Architecture

![System Architecture](./Images/Architecture.png)

*4-layer pipeline: Input → Multi-Stream Core → Semantic & Contextual Validation → Adaptive Fusion Output*

---

## Data Flow Diagram

![Data Flow Diagram](./Images/DFD.png)

---

## Key Features

| Feature | Description |
|---|---|
| **Visual Forensics** | ELA analysis, noise variance, JPEG ghost detection, copy-paste SIFT, CNN inference, edge consistency |
| **Triple OCR Consensus** | Tesseract + EasyOCR + PaddleOCR with majority-voting to maximize extraction accuracy |
| **Arithmetic Semantic Gate** | Hard rule: verifies `Net + VAT = Gross Total` — forced fraud flag on mismatch |
| **Vendor Registry** | SQLite-backed enrollment of known vendors with GST numbers, expected amount ranges & bill formats |
| **Adaptive Fusion** | Quality-aware weighted fusion (visual vs. text) producing a `fraud_score` from 0.0–1.0 |
| **Audit Dashboard** | Full history with verdicts, scores, processing times, and trend charts |
| **XAI Heatmaps** | ELA heatmap overlays for explainable AI — shows *where* tampering was detected |

---

## UI Screenshots

### Bill Verification
Upload any financial document image for instant multi-modal fraud analysis.

![Verification Upload](./Images/home_upload.png)

### Audit History Dashboard
Monitor all past verifications — verdicts, fraud scores, processing times, and trends over time.

![Audit History](./Images/audit_history_ui.png)

### Vendor Enrollment
Register known vendors with their GST numbers, expected transaction amounts, and bill formats.

![Vendor Enrollment](./Images/vendor_enrollment_ui.png)

### Validation & Metrics
Model performance metrics, ROC curves, confusion matrix, and ablation study results.

![Validation Dashboard](./Images/validation_dashboard.png)

---

## Tech Stack

### Backend
| Library | Version | Purpose |
|---|---|---|
| **Flask** | 3.x | Web framework & routing |
| **TensorFlow / Keras** | 2.x | CNN model inference |
| **OpenCV** | 4.x | Image preprocessing, ELA, SIFT, edge analysis |
| **Pillow** | Latest | Image I/O and manipulation |
| **NumPy** | Latest | Array operations & numerical computing |
| **scikit-learn** | Latest | Evaluation metrics (F1, ROC-AUC, confusion matrix) |
| **SQLite3** | Built-in | Vendor registry & bill audit log |

### OCR Engines (Triple Consensus)
| Engine | Purpose |
|---|---|
| **Tesseract** (`pytesseract`) | Primary OCR — layout-aware text extraction |
| **EasyOCR** | Secondary OCR — handles skewed/rotated text |
| **PaddleOCR** | Tertiary OCR — high-accuracy structured document parsing |

### Visualization & Reporting
| Library | Purpose |
|---|---|
| **Matplotlib** | Training curves, ROC curves, ablation plots |
| **Seaborn** | Confusion matrix heatmap |
| **Chart.js** | Frontend dashboard charts (bundled in `static/`) |

### Frontend
| Technology | Purpose |
|---|---|
| **HTML5 / CSS3** | UI templates via Jinja2 (Flask) |
| **JavaScript** | Dynamic chart rendering (Chart.js) |
| **Jinja2** | Server-side template rendering |

---

## Project Structure

```
forensiq-multimodal-financial-document-forgery-detection/
│
├── app.py                    # Flask app — main routes & pipeline orchestration
├── requirements.txt          # Python dependencies
│
├── classifier/
│   └── fraud_classifier.py   # Adaptive fusion classifier + GST validation + semantic gate
│
├── ocr/
│   └── ocr_engine.py         # Triple OCR (Tesseract + EasyOCR + PaddleOCR) with consensus
│
├── vision/
│   └── vision_model.py       # Visual forensics: ELA, noise, JPEG ghost, SIFT, CNN, edges
│
├── utils/
│   ├── vendor_db.py          # SQLite vendor registry CRUD + bill history logging
│   └── preprocessing.py      # Image preprocessing pipeline (denoise, threshold, resize)
│
├── models/
│   └── fraud_document_cnn.h5 # Trained CNN model weights
│
├── templates/                # Jinja2 HTML templates
│   ├── index.html            # Bill verification page
│   ├── vendors.html          # Vendor enrollment page
│   ├── history.html          # Audit history dashboard
│   └── metrics.html          # Model metrics & paper figures
│
├── static/
│   └── chart.umd.min.js      # Bundled Chart.js (no CDN dependency)
│
├── scripts/
│   ├── train_cnn.py          # CNN training script
│   ├── run_evaluation.py     # Full evaluation suite (accuracy, F1, ROC, confusion matrix)
│   ├── generate_paper_metrics.py  # Generates metrics.json + report figures
│   ├── consolidate_dataset.py     # Dataset preprocessing & label consolidation
│   ├── audit_dataset_labels.py    # Dataset label auditing
│   └── test_images_quick.py       # Quick smoke-test on sample images
│
├── notebooks/                # Jupyter notebooks (EDA, model experiments, OCR testing)
│   ├── 02_fraud_sanity_check.ipynb
│   ├── 08_cv_visual_features.ipynb
│   ├── 09_fraud_classifier.ipynb
│   ├── 11_cnn_model.ipynb
│   └── ... (17 notebooks total)
│
├── reports/
│   ├── metrics.json          # Evaluation results (accuracy, F1, precision, recall, AUC-ROC)
│   ├── roc_curve.png         # ROC curve figure
│   ├── confusion_matrix.png  # Confusion matrix figure
│   ├── ablation_table.png    # Ablation study table
│   └── comparison_table.png  # Comparison with baselines
│
├── Images/                   # Architecture diagrams & UI screenshots
│   ├── Architecture.png
│   ├── DFD.png
│   └── *.png
│
└── uploads/                  # Runtime upload directory (contents gitignored)
    └── heatmaps/             # Generated ELA heatmap overlays
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and added to PATH
- Git

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Gowtham-071/forensiq-multimodal-financial-document-forgery-detection.git
cd forensiq-multimodal-financial-document-forgery-detection
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

> **Note:** `paddleocr` and `paddlepaddle` can be heavy. If you're on CPU only, install `paddlepaddle` before `paddleocr`:
> ```bash
> pip install paddlepaddle
> pip install paddleocr
> ```

**4. Install Tesseract OCR**
- **Windows:** Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **macOS:** `brew install tesseract`

**5. Run the Flask application**
```bash
python app.py
```

**6. Open in browser**
```
http://127.0.0.1:5000
```

---

## Application Routes

| Route | Description |
|---|---|
| `GET /` | Bill verification — upload a document image |
| `POST /` | Processes the uploaded image through the full pipeline |
| `GET /vendors` | View enrolled vendors |
| `POST /vendors` | Enroll a new vendor with GST & amount range |
| `GET /history` | Audit history dashboard with stats & charts |
| `GET /metrics` | Model performance metrics & evaluation figures |

---

## Verdict Logic

The final `fraud_score` (0.0 → 1.0) is computed by adaptive fusion and thresholded as:

| Score Range | Verdict |
|---|---|
| `< 0.45` | ✅ **GENUINE** |
| `0.45 – 0.60` | ⚠️ **SUSPICIOUS** |
| `> 0.60` | ❌ **FRAUD** |

A hard override forces score ≥ 0.75 if arithmetic totals fail (`Net + VAT ≠ Gross`).

---

## Training the CNN (Optional)

If you have a dataset and want to retrain the CNN model:

```bash
# Consolidate and label the dataset
python scripts/consolidate_dataset.py

# Train the CNN
python scripts/train_cnn.py

# Evaluate model performance
python scripts/run_evaluation.py

# Generate paper figures (metrics.json, ROC, confusion matrix)
python scripts/generate_paper_metrics.py
```

Trained model is saved to `models/fraud_document_cnn.h5`.

---

## requirements.txt

```
flask
opencv-python
numpy
Pillow
tensorflow
pytesseract
easyocr
paddleocr
paddlepaddle
scikit-learn
matplotlib
seaborn
```

---

## License & Rights

> **© 2025 Gowtham. All Rights Reserved.**
>
> This project is made available for **viewing and reference purposes only**. No part of this codebase, research methodology, trained models, or associated materials may be reproduced, distributed, modified, used commercially, or published — in whole or in part — without explicit written permission from the author.
>
> The underlying research paper is **unpublished and proprietary** and is not included in this repository.

---

## Contact

Have questions, want to collaborate, or need permission for usage?

| Platform | Link |
|---|---|
| 💼 **LinkedIn** | [Chanchala Sai Gowtham](https://www.linkedin.com/in/chanchala-sai-gowtham-a06314322/) |
| 📧 **Email** | [saigowtham712@gmail.com](mailto:saigowtham712@gmail.com) |

---

<div align="center">
  <sub>Built with ❤️ as a capstone research project · FORENSIQ © 2025</sub>
</div>
