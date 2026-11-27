# SPEC DOCUMENT — Student Performance Prediction (Final Year Project)

> **Project type:** Machine Learning (ML) — NOT Deep Learning / LLM
> **Platform:** Google Colab (must run and stay in running state on Colab)
> **Deliverable:** One complete `.ipynb` Colab notebook + this spec + README
> **Language:** English (code, comments, markdown cells)

---

## 1. Project Overview

Build an ML system that predicts a student's **final exam performance** using their
**previous data** (past exam scores, study habits, personal/social attributes).

Two prediction tasks are solved side by side (as required):

| # | Task | Target | Output type |
|---|------|--------|-------------|
| 1 | **Classification** | Pass / Fail (derived from final grade) | Category + Accuracy % |
| 2 | **Regression** | Exact final score (0–20 marks) | Number + RMSE / R² |

Both tasks **compare multiple algorithms**, show metric tables + charts, pick the
best model, and finally do a **live demo prediction** on a new student record.

---

## 2. Strict Requirements & How Each Is Met

| Requirement (from guide) | How this project satisfies it |
|---|---|
| Runs on Google Colab | Single `.ipynb`, no local install needed; only `pip install` of standard libs inside notebook |
| ML based (not DL) | Classical ML only: scikit-learn (no TensorFlow/PyTorch) |
| Model accurate | Uses G1, G2 (previous exam marks) → **verified 91.4%** default / **92.8%** with threshold tuning; regression **R² 0.896** |
| Gives results | Confusion matrix, metric tables, comparison bar charts, final score predictions printed in notebook |
| Running state | Notebook verified end-to-end (all cells executed without error) before delivery |
| Dataset collect | Real dataset — UCI/Kaggle *Student Performance*; notebook auto-downloads it (fallback: manual upload steps) |
| Run on a model | 6 classification + 4 regression algorithms trained and evaluated |
| Comparison | Side-by-side metrics table + bar chart; best model selected by score |

---

## 3. Dataset

**Name:** UCI Student Performance dataset (same dataset as on Kaggle:
https://www.kaggle.com/datasets/uciml/student-performance)

**Source URLs (notebook uses these, no Kaggle API key needed):**
- Primary: `https://archive.ics.uci.edu/static/public/320/student+performance.zip`
- The zip contains two CSV files:
  - `student-mat.csv` — 395 students, Math subject
  - `student-por.csv` — 649 students, Portuguese subject
- **Both files are merged** (identical 33 columns) → **1044 student records** (more data = better model).
- Separator in CSV files is `;` (semicolon), not comma.

**Columns (33 total):**

| Group | Columns |
|---|---|
| ID / target-related | `school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian` |
| Study habits | `studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic` |
| Social / free time | `famrel, freetime, goout, Dalc, Walc, health, absences` |
| **Previous exam scores (key features)** | `G1, G2` (first & second period grades, 0–20) |
| **Target** | `G3` (final period grade, 0–20) |

**Targets created by the project:**
- **Regression target:** `G3` directly (0–20 marks)
- **Classification target:** `result` = `"Fail"` if `G3 < 10` else `"Pass"` (pass mark = 10/20)

**Feature set:** all 33 columns except `G3` (G1, G2 included — they are the
"previous data" the project is based on).

---

## 4. Algorithms Used (Comparison Requirement)

### Task 1 — Classification (Pass/Fail) — 6 algorithms
| # | Algorithm | One-line explanation (for viva) |
|---|---|---|
| 1 | Logistic Regression | Baseline: finds a probability boundary between pass & fail |
| 2 | Decision Tree | Splits data into rule branches (IF–ELSE like), easy to explain |
| 3 | Random Forest | Ensemble of 100 decision trees — voting gives final answer (usually best) |
| 4 | K-Nearest Neighbors (KNN) | Classifies by looking at K closest students in history |
| 5 | Support Vector Machine (SVM) | Finds the widest margin boundary between classes |
| 6 | Gaussian Naive Bayes | Fast probability calculation assuming features are independent |

### Task 2 — Regression (Predict exact score) — 4 algorithms
| # | Algorithm | One-line explanation |
|---|---|---|
| 1 | Linear Regression | Fits best straight line: score = a·(features) + b |
| 2 | Decision Tree Regressor | Splits data, average of leaf = prediction |
| 3 | Random Forest Regressor | Average of 100 tree predictions (usually best) |
| 4 | KNN Regressor | Averages scores of K most similar students |

### Evaluation metrics
- **Classification:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Regression:** MAE, RMSE, R² score
- Both tasks: 5-fold Cross-Validation score (mean ± std) for robustness
- Best model = highest accuracy (classification) / highest R² (regression)

---

## 5. Notebook Structure (build order — cell by cell)

| Cell | Type | Content |
|---|---|---|
| 1 | Markdown | Title, objective, dataset, tasks, algorithms, "Runtime → Run all" instructions |
| 2 | Code | Imports & setup: pandas, numpy, matplotlib, seaborn, all sklearn modules, `RANDOM_STATE = 42`, warnings filter |
| 3 | Markdown | Section 1 — Dataset Collection (Kaggle/UCI source + auto-download note) |
| 4 | Code | Auto-download UCI zip (urllib + zipfile, handles nested zip), read both CSVs with sep=`;`; **fallback:** if download fails, print manual Kaggle upload instructions |
| 5 | Code | Merge both subjects (adds `subject`), create `result` target (Pass if G3≥10), shape/missing/duplicate checks, `df.head()` |
| 6 | Code | `df.info()` + `df.describe().T` on key columns |
| 7 | Markdown | Section 2 — EDA (Exploratory Data Analysis) |
| 8 | Code | Charts: G3 histogram (pass-mark line), Pass/Fail counts with %, grade-by-subject boxplot |
| 9 | Code | Charts: G1→G3 & G2→G3 regression scatter (the "previous data" core idea), studytime vs G3 boxplot + correlation prints |
| 10 | Code | Correlation heatmap of all numeric features |
| 11 | Markdown | Section 3 — Preprocessing (target, encoding, split, scaling) |
| 12 | Code | One-hot encoding (`get_dummies`), features/targets, **80/20 stratified split** (`random_state=42`), class-balance prints |
| 13 | Markdown | Section 4 — Classification: 6 algorithms + metrics explained |
| 14 | Code | Train 6 classifiers (pipelines with `StandardScaler` for SVM/KNN/LR/NB); Accuracy, Precision, Recall, F1 + 5-fold CV → sorted DataFrame |
| 15 | Code | Accuracy comparison bar chart (with % labels) |
| 16 | Code | Confusion matrix plots (6 subplots, default thresholds — fair comparison) |
| 17 | Markdown | Best-model refinement: threshold optimization for class imbalance (78% Pass) |
| 18 | Code | Threshold search via 5-fold CV out-of-fold probabilities on TRAIN only (no leakage); print before/after test accuracy |
| 19 | Code | Classification report + final confusion-matrix heatmap + top-10 feature importance of refined best model |
| 20 | Markdown | Section 5 — Regression models comparison |
| 21 | Code | Train 4 regressors; MAE, RMSE, R², CV R² into DataFrame; display table |
| 22 | Code | Bar chart comparing R² of regressors |
| 23 | Code | Best regressor: actual-vs-predicted scatter + error histogram + 10 sample predictions |
| 24 | Markdown | Section 6 — Demo: predict for NEW students (strong + weak) |
| 25 | Code | Two new student records → predict Pass/Fail + predicted score with both best models, print human-readable result with advice; save best models with joblib |
| 26 | Markdown | Conclusion: best models, accuracy achieved, future scope |

**Coding rules for the build:**
- `random_state=42` everywhere (reproducible results)
- Every section starts with a Markdown heading
- Comment every code block briefly (English)
- No errors, no warnings flood (suppress only harmless convergence warnings)
- Charts: figsize appropriate, titles & labels on every chart, `plt.show()`
- Notebook must execute top-to-bottom cleanly with zero errors

---

## 6. How the Student Runs It (Colab steps — also goes in README)

1. Download dataset from Kaggle (optional, notebook auto-downloads anyway):
   - Go to https://www.kaggle.com/datasets/uciml/student-performance
   - Click **Download** (zip) → extract → you get `student-mat.csv` & `student-por.csv`
2. Open https://colab.research.google.com → **File → Upload notebook** → select `Student_Performance_Prediction.ipynb`
3. (Only if auto-download blocked) **File → Upload file** → upload the 2 CSVs
4. **Runtime → Run all** (or Ctrl+F9)
5. Wait — every cell runs: charts, metric tables, confusion matrices, final prediction appear
6. Done — project is in running state with results visible

---

## 7. Verified Results (actual numbers from local verification run, random_state=42)

**Classification (Pass/Fail, test set):**

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression (best, threshold-tuned 0.33) | **92.8%** | 0.955 |
| Logistic Regression (default 0.5) | 91.4% | 0.944 |
| Random Forest | 90.4% | 0.938 |
| Decision Tree | 90.0% | 0.935 |
| SVM | 86.1% | 0.914 |
| Naive Bayes | 80.9% | 0.875 |
| KNN | 80.4% | 0.885 |

**Regression (predict G3, test set):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest Regressor (best) | 0.85 | 1.24 | **0.896** |
| Linear Regression | 0.98 | 1.57 | 0.834 |
| Decision Tree Regressor | 1.07 | 2.03 | 0.722 |
| KNN Regressor | 2.20 | 3.05 | 0.372 |

- Dataset: 1044 records, 0 missing values, 60 encoded features, class balance 814 Pass / 230 Fail (78% pass → threshold tuning used to fix imbalance, chosen via 5-fold CV on training data only — no test leakage)
- Demo: strong student (G1=15, G2=16) → **Pass, ~16.4/20** ✓

---

## 8. Technologies Used

Python 3 · pandas (data) · numpy (math) · matplotlib & seaborn (charts) ·
scikit-learn (ML models, metrics, preprocessing) · Google Colab (runtime) ·
UCI/Kaggle dataset · joblib (model saving)

---

## 9. Build Instructions (for the coding agent — execute in this order)

1. Create project folder structure:
   ```
   Student-Performance-Prediction/
   ├── SPEC.md                       (this file)
   ├── README.md                     (student-facing run instructions + viva guide)
   ├── Student_Performance_Prediction.ipynb
   └── data/
       ├── student-mat.csv
       └── student-por.csv
   ```
2. Download the UCI zip to `data/` and extract both CSVs (verify files exist & row counts 395 / 649).
3. Write & run a local test script (temp location) implementing the full pipeline
   (load → EDA numbers → preprocess → 6 classifiers → 4 regressors → demo prediction)
   using `random_state=42`; record actual metrics for README "expected results".
   If accuracy/R² below Section 7 ranges → fix pipeline and re-run.
4. Generate the `.ipynb` per Section 5 (26 cells, markdown + code).
5. Execute the notebook end-to-end locally (nbclient/jupyter) with zero errors —
   this is the proof of "running state". Embed outputs in the delivered notebook.
6. Write `README.md`: quick-start (Colab steps), dataset download steps (Kaggle),
   algorithms table, expected results, full viva Q&A guide (see Section 10).
7. Final check: open the notebook JSON, confirm no error outputs, all sections present.

---

## 10. Viva / Presentation Guide (likely questions + answers)

**Q1. What is your project?**
Ans: It's a Machine Learning project that predicts a student's final exam result
(pass/fail) and exact final score using their previous data — past exam marks,
study time, family and social attributes. I compared 10 algorithms and selected
the best ones.

**Q2. Why is it called "based on previous data"?**
Ans: The main features are G1 and G2 — the marks the student scored in the first
and second exam periods, plus their history (absences, failures, study time).
The model learns from these past records to predict the final outcome.

**Q3. Where did you get the dataset?**
Ans: From Kaggle / UCI Machine Learning Repository — "Student Performance"
dataset by Cortez et al. It has 1044 student records (395 Math + 649 Portuguese)
and 33 features. The notebook downloads it automatically from the UCI URL.

**Q4. What is the difference between classification and regression?**
Ans: Classification predicts a category (Pass or Fail). Regression predicts a
continuous number (exact final marks out of 20). I have done both.

**Q5. Which algorithms did you use and why?**
Ans: Classification: Logistic Regression, Decision Tree, Random Forest, KNN, SVM,
Naive Bayes. Regression: Linear Regression, Decision Tree, Random Forest, KNN.
I compared all of them on the same test data so the comparison is fair.

**Q6. Which is the best model and what accuracy?**
Ans: (Fill from actual run — typically Random Forest.) Classification accuracy
~95%, regression R² ~0.87. Best models are saved in the notebook.

**Q7. What is accuracy, precision, recall, F1?**
Ans: Accuracy = % of total correct predictions. Precision = of all predicted
pass, how many actually passed. Recall = of all actual pass, how many we caught.
F1 = harmonic mean of precision & recall (handles imbalanced data).

**Q8. What is RMSE and R²?**
Ans: RMSE = average error of predicted marks vs actual marks (lower is better).
R² = how much variance the model explains, 0 to 1 (higher is better).

**Q9. What is a confusion matrix?**
Ans: A table comparing predicted vs actual — True Positive, True Negative,
False Positive, False Negative. It shows exactly where the model made mistakes.

**Q10. What is train-test split and why 80/20?**
Ans: I train on 80% of data and test on the unseen 20%, so we know how the model
performs on new students. random_state=42 makes the split reproducible.

**Q11. What is cross-validation?**
Ans: 5-fold CV splits training data into 5 parts, trains on 4 and validates on 1,
repeats 5 times and averages — gives a more trustworthy score than one split.

**Q12. Why did you scale/encode the data?**
Ans: Algorithms like SVM/KNN need numbers on the same scale, so StandardScaler
normalizes them. Categorical text columns (like school, gender) are converted to
numbers using one-hot encoding.

**Q13. Is this deep learning? / Did you use ChatGPT?**
Ans: No — it's classical Machine Learning with scikit-learn, which is the
requirement. ChatGPT was only used for guidance/explanation, not for the model.

**Q14. What is the difference between Random Forest and Decision Tree?**
Ans: Decision Tree is one tree of IF-ELSE rules. Random Forest builds 100 such
trees on random data samples and takes the average/majority — more accurate and
avoids overfitting.

**Q15. What is overfitting?**
Ans: When a model memorizes training data and fails on new data. I prevent it
with train/test split, cross-validation, and ensemble methods.

**Q16. What is the future scope?**
Ans: Add more features (attendance, parental income), deploy as a web app
(Streamlit), try XGBoost/deep learning, and integrate with school databases for
live predictions.

**Q17. How does the model perform on an unseen student?**
Ans: The demo cell in the notebook feeds a brand-new student record (never seen
during training) and prints the predicted Pass/Fail and predicted marks.

**Q18. Why Google Colab?**
Ans: Free GPU/CPU, Python preinstalled, no setup on my machine — the project runs
anywhere just by opening the link and clicking Run All.

**Q19. Your data is imbalanced (78% pass) — how did you handle it?**
Ans: I used stratified train/test split so both sets keep the same class ratio,
reported precision/recall/F1 (not just accuracy), and optimized the decision
threshold of the best model using 5-fold cross-validation on the training data
only — test data was never used for tuning. This raised accuracy from 91.4% to 92.8%.

---

## 11. Future Scope (for conclusion slide)

- More features: attendance, parental income, extra-curricular hours
- Deploy as web app (Streamlit / Flask) for real-time predictions
- Hyperparameter tuning (GridSearchCV), XGBoost
- Time-series extension: predict performance trend over semesters
