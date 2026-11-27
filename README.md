# Student Performance Prediction — Final Year Project

Machine Learning project that predicts a student's **final exam result (Pass/Fail)**
and **exact final score (0–20)** from their **previous data** — past exam marks (G1, G2),
study habits, family and social attributes. Runs fully on **Google Colab**.
Made Under the Guidance of Dr. Kailiash Chandra Bandhu and Ms. Yashasvi Sharma
---

## Project Structure

```
Student-Performance-Prediction/
├── SPEC.md                              # Full spec, build rules & Viva Q&A guide (Section 10)
├── README.md                            # This file — quick start
├── Student_Performance_Prediction.ipynb  # MAIN notebook (26 cells, outputs included)
├── Student_Performance_Prediction_UI.ipynb # Gradio WEB UI notebook (runs in Colab)
├── app.py                               # Streamlit WEB UI (browser interface)
├── model_core.py                        # Shared ML logic used by the web UIs
├── requirements.txt                     # Python dependencies
├── data/
│   ├── student-mat.csv                  # 395 Math students (from UCI/Kaggle)
│   └── student-por.csv                  # 649 Portuguese students
└── models/
    ├── best_classifier.joblib           # Saved best classification model
    └── best_regressor.joblib            # Saved best regression model
```

---

## How to Run on Google Colab (step by step)

**Option A — with this folder (easiest):**
1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `Student_Performance_Prediction.ipynb`
3. **Runtime → Run all** (or `Ctrl + F9`)
4. Done — charts, comparison tables, confusion matrices and final predictions appear.
   (The notebook auto-downloads the dataset from UCI — internet is on by default in Colab.)

**Option B — Kaggle dataset manually (if auto-download is blocked):**
1. Open [kaggle.com/datasets/uciml/student-performance](https://www.kaggle.com/datasets/uciml/student-performance)
2. Click **Download** → you get `student+performance.zip`
3. Unzip it → you get `student-mat.csv` and `student-por.csv`
4. In Colab: **Files panel (left) → Upload** → create folder `data` → upload both CSVs
5. Re-run the notebook (cells will detect the files and skip the download)

> Tip: this dataset is the famous **UCI Student Performance** dataset
> (Cortez et al., 2008) — same one used in thousands of Kaggle kernels, so it is
> a safe, recognizable choice for submission.

---

## 🌐 Frontend (Web UI) — two ways to show the professor

### Option A — Gradio UI inside Colab (easiest, recommended for submission)
1. Upload **`Student_Performance_Prediction_UI.ipynb`** to Colab
2. **Runtime → Run all**
3. Last cell prints a **share link** (and shows the form) → click it
4. A full form opens: fill student details (or drag G1/G2 sliders) → **PREDICT** → Pass/Fail + score appear instantly
- Self-contained: downloads data + trains models by itself (7 cells, ~40 sec)

### Option B — Streamlit website (looks like a real website)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Browser me `http://localhost:8501` khulta hai — grouped tabs
(Personal / Family / Study / Social / Previous Marks), sliders, and a
**PREDICT RESULT** button with color output, progress bar and advice.

**Frontend ↔ Backend mapping:** `app.py` / Gradio notebook = **frontend** ·
`model_core.py` + scikit-learn = **backend** · same models as the main notebook
(92.82% accuracy, R²=0.896, verified identical outputs).

---

## What Is Used (Tech Stack)

| Component | Used for |
|---|---|
| Python 3 / Google Colab | Free runtime, zero setup |
| pandas + numpy | Data loading, merging, cleaning |
| matplotlib + seaborn | 8 charts (distributions, heatmap, confusion matrices, scatter) |
| scikit-learn | All ML models, metrics, train/test split, cross-validation |
| joblib | Saving the trained models |

## Algorithms Compared (10 total)

**Classification (Pass/Fail):** Logistic Regression · Decision Tree · Random Forest ·
KNN · SVM · Gaussian Naive Bayes

**Regression (score 0–20):** Linear Regression · Decision Tree Regressor ·
Random Forest Regressor · KNN Regressor

**Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix · MAE, RMSE, R² ·
5-Fold Cross-Validation (all with `random_state = 42` → reproducible)

---

## Verified Results (test set, 20% unseen data)

**Classification — best: Logistic Regression (threshold-tuned via CV)**

| | Accuracy |
|---|---|
| Default threshold 0.50 | 91.39% |
| **Tuned threshold 0.33 (final)** | **92.82%** |
| Random Forest | 90.43% |
| Decision Tree | 89.95% |
| SVM | 86.12% |
| Naive Bayes | 80.86% |
| KNN | 80.38% |

**Regression — best: Random Forest Regressor**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest Regressor** | **0.85** | **1.24** | **0.896** |
| Linear Regression | 0.98 | 1.57 | 0.834 |
| Decision Tree Regressor | 1.07 | 2.03 | 0.722 |
| KNN Regressor | 2.20 | 3.05 | 0.372 |

**Demo output (notebook last section):**
- Strong student (G1=15, G2=16) → **Pass**, predicted **16.37 / 20** ✔
- Struggling student (G1=8, G2=7) → **Fail** (98% confidence), predicted **7.38 / 20** ✔

---

## Presentation / Viva — preparation

Full **19 likely questions with answers** are in **`SPEC.md` → Section 10**.
Quick revision list:

1. *What is the project?* → ML model predicting student Pass/Fail + exact score from previous data
2. *Dataset?* → UCI/Kaggle Student Performance, 1044 records, 33 features
3. *Why "previous data"?* → main features are G1/G2 (earlier exam marks)
4. *Classification vs regression?* → category vs continuous number (I did both)
5. *Best model?* → Logistic Regression 92.8% (clf) + Random Forest R²=0.896 (reg)
6. *What is accuracy/precision/recall/F1?* → see SPEC Q7
7. *What is RMSE/R²?* → see SPEC Q8
8. *What is a confusion matrix?* → see SPEC Q9
9. *Train/test split? CV?* → see SPEC Q10–Q11
10. *Why scaling/encoding?* → see SPEC Q12
11. *Deep learning?* → No — classical ML (scikit-learn), as required
12. *Class imbalance handled?* → stratified split + F1 + CV-based threshold tuning (Q19)

---

*Verified end-to-end: all 26 cells execute with 0 errors (~21 seconds).*
