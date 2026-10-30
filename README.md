# PrivateMed: Differentially Private Clinical Model Training under Epsilon Budget

**Domain:** Healthcare Data Protection / Technology  
**Syllabus Mapping:** Module 4 (Privacy Vulnerabilities, Laplace Mechanism, $(\varepsilon, \delta)$-Differential Privacy)

---

## 📌 Project Overview
PrivateMed implements differential privacy mechanisms (Laplace mechanism and DP-SGD) for training clinical illness risk predictors while ensuring zero personal identification risk.

### Required Deliverables Included:
1. **Executable Python Codebase (`api/dp_engine.py`, `api/app.py`, `main.py`)**:
   - Calibrated Laplace noise injection based on global $L_1$ sensitivity bounds: $\text{Lap}(b)$ where $b = \Delta f / \varepsilon$.
   - Differentially private clinical classifier training with per-sample gradient clipping ($\|g\|_2 \le C$) and budget composition across attributes.
   - 16 clinical biomarker sensitivity calculations with sequential privacy budget composition.
2. **Pareto Utility Analysis Plot**:
   - Interactive Pareto trade-off curve tracking Diagnostic Utility (AUC-ROC & Accuracy) against Privacy Budget $\varepsilon \in [0.05, 50.0]$.
   - Visualizes the clinical "Golden Zone" ($\varepsilon = 2.0$, $\text{AUC-ROC} \approx 0.894$, $\text{MIA Risk} \approx 51.4\%$).
3. **Mathematical Derivation Appendix**:
   - Plain-English + formal derivations for $L_1$ sensitivity: $\Delta f = (\text{Max} - \text{Min}) / N$.
   - Laplace noise calibration scale: $b = \Delta f / \varepsilon$.
   - Sequential composition budget allocation across 16 vital clinical tests.
   - Theoretical Membership Inference Attack (MIA) advantage bound: $\frac{e^\varepsilon - 1}{e^\varepsilon + 1}$.

---

## 🚀 How to Run

### Option 1: Standalone Web Interface (Instant, Zero Setup)
Double-click or open `static/index.html` in any web browser (Chrome, Edge, Firefox). All simulations, interactive tabs, sliders, charts, and plain-English derivation guides work client-side out of the box!

### Option 2: Full-Stack Python Backend + UI
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch server
python main.py
```
- Open UI: [http://localhost:8000/ui](http://localhost:8000/ui)
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 3: Lightweight Node/Static Server
```bash
npx serve -s static -l 3000
```
- Open [http://localhost:3000](http://localhost:3000)

---

## 📁 Repository Structure
```
PrivateMed/
├── api/
│   ├── app.py           # FastAPI endpoints for DP training & Laplace perturbation
│   └── dp_engine.py     # Laplace mechanism, DP clinical model, & Pareto engine
├── static/
│   └── index.html       # Interactive 4-tab web application (Plain-English + Math view)
├── main.py              # Full-stack server launcher
├── requirements.txt     # Python dependencies (fastapi, uvicorn, numpy, pydantic)
└── README.md            # Documentation & project guide
```
