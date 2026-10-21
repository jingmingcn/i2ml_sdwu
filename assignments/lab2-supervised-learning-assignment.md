# Lab 2 Assignment — Supervised Learning

**Course:** Introduction to Machine Learning (with Python & scikit-learn)
**Chapter:** 2 — Supervised Learning
**Topic:** Comparing Classifiers on the Breast Cancer Dataset
**Weight:** 10% of the final grade
**Due:** _2026.11.11_  **Submit via:** _xuexitong_

---

## 1. Overview

In this lab you train and compare several supervised classifiers on the Breast
Cancer Wisconsin dataset, observe **overfitting**, read a random forest's
**feature importances**, see why **SVMs need feature scaling**, and obtain
**probability estimates**. You then **write up your work as a lab report** using
one of the provided templates.

## 2. What to do

1. **Complete the notebook** `notebooks/02-supervised-learning-code.ipynb`.
   Fill in every cell marked `# TODO` and make the **Check** cells pass. Cover all
   six exercises:
   1. Load the Breast Cancer dataset and create a train/test split (`random_state=0`).
   2. Train `LogisticRegression(max_iter=5000)`; report train/test accuracy.
   3. Train a `DecisionTreeClassifier` with no depth limit **and** with
      `max_depth=4`; compare train vs. test accuracy (which overfits?).
   4. Train `RandomForestClassifier(n_estimators=100, random_state=0)`; report
      test accuracy and the **5 most important features**.
   5. Train `SVC()` on **raw** vs. `StandardScaler`-**scaled** data; compare.
   6. Print `predict_proba` for the first 5 test samples.

2. **Write the lab report** using **one** of the templates (choose the format you prefer):
   - **Word:** `docx/lab2-supervised-learning.docx`
   - **LaTeX:** `tex/lab2-supervised-learning.tex`

   Fill in the title block (your name, student ID, date) and replace every
   `TODO` / placeholder with **your own** results, tables, and analysis.

## 3. Deliverables

Submit the following:

- [x] Your lab report as a **PDF** (exported from the Word or LaTeX template).

**File naming:** `Lab2_<StudentID>_<Name>.pdf`.

## 4. Report requirements

Your report must include (the template already has these sections):

- **Objectives** and a short **Background** in your own words.
- **Dataset** description.
- **Methodology** — the models and parameters you used.
- **Results** — fill the tables for Exercises 1–6 with **your** measured values
  (model comparison, tree over-fitting, top-5 features, SVM scaling, probabilities).
- **Discussion** — answer the prompts (how large is the over-fitting gap? which
  model generalised best? how much did scaling help the SVM? do the top features
  make sense? are any predictions low-confidence?).
- **Conclusion** and **References**.

## 5. Notes

- You may use either template, but the **content and sections are the same** — pick
  Word or LaTeX based on your toolchain.
- Work must be your **own**. You may discuss ideas, but copying another student's
  report or results is not permitted.
- Late submissions: _______________ (per course policy).
