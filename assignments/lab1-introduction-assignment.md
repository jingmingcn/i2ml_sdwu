# Lab 1 Assignment — Introduction

**Course:** Introduction to Machine Learning (with Python & scikit-learn)
**Chapter:** 1 — Introduction
**Topic:** Classifying Iris Species with k-Nearest Neighbors
**Weight:** 10% of the final grade
**Due:** _2026.11.11_  **Submit via:** _xuexitong_

---

## 1. Overview

In this lab you rebuild the course's first machine learning application: you load
the Iris dataset, split it, train a **k-Nearest Neighbors** classifier, make a
prediction, and evaluate it. You then **write up your work as a lab report** using
one of the provided templates.

## 2. What to do

1. **Complete the notebook** `notebooks/01-introduction-code.ipynb`.
   Fill in every cell marked `# TODO` and make the **Check** cells pass. Cover all
   seven exercises:
   1. Load the Iris dataset.
   2. Explore it (samples, features, per-class counts).
   3. Create a train/test split with `random_state=0`.
   4. Build and train `KNeighborsClassifier(n_neighbors=1)`.
   5. Predict the species of a new flower `[5.0, 2.9, 1.0, 0.2]`.
   6. Evaluate the test-set accuracy.
   7. **(Bonus)** Vary `k` from 1 to 10 and compare the accuracy.

2. **Write the lab report** using **one** of the templates (choose the format you prefer):
   - **Word:** `docx/lab1-introduction.docx`
   - **LaTeX:** `tex/lab1-introduction.tex`

   Fill in the title block (your name, student ID, date) and replace every
   `TODO` / placeholder with **your own** results, figures, and analysis.

## 3. Deliverables

Submit the following:

- [x] Your lab report as a **PDF** (exported from the Word or LaTeX template).

**File naming:** `Lab1_<StudentID>_<Name>.pdf`.

## 4. Report requirements

Your report must include (the template already has these sections):

- **Objectives** and a short **Background** in your own words.
- **Dataset** description.
- **Methodology** — what you did and why.
- **Results** — the measured values for Exercises 1–7 (not just the "expected" hints).
- **Discussion** — answer the prompts (why hold out a test set? how does `k`
  relate to model complexity? drawbacks of k-NN?).
- **Conclusion** and **References**.

## 5. Notes

- You may use either template, but the **content and sections are the same** — pick
  Word or LaTeX based on your toolchain.
- Work must be your **own**. You may discuss ideas, but copying another student's
  report or results is not permitted.
- Late submissions: _______________ (per course policy).
