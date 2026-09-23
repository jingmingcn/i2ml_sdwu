---
marp: true
theme: course
paginate: true
footer: 'Introduction to Machine Learning · Ch.5 Exercises'
---

<!-- _class: lead -->

# Chapter 5 — Exercises

## Multiple-Choice Questions

Model evaluation & improvement

---

## How to Use These Slides

- **12 multiple-choice** questions on Chapter 5 concepts
- Pick **one** answer (A–D) for each question
- The **answer key** with brief explanations is at the end
- Try to answer before revealing the solutions

> Topics: cross-validation, grid search, imbalanced data, confusion matrix, precision/recall/f1, thresholds, and ROC/AUC.

---

## Q1. The main benefit of cross-validation over a single split is:

A. a more stable, reliable estimate of generalization

B. faster training

C. fewer features

D. guaranteed higher accuracy

---

## Q2. Stratified k-fold cross-validation:

A. shuffles the target labels

B. keeps each class's proportion the same in every fold

C. uses one sample per fold

D. is only used for regression

---

## Q3. Why shouldn't you tune parameters on the **test set**?

A. It is too slow

B. The test set is too small

C. It leaks the test set and gives over-optimistic scores

D. It changes the labels

---

## Q4. `GridSearchCV.best_score_` reports:

A. the test-set accuracy

B. the training error

C. the number of parameter combinations

D. the mean cross-validation score on the training data

---

## Q5. On a 99%-negative imbalanced dataset, accuracy is:

A. misleading — "always predict negative" already scores 99%

B. always zero

C. the best possible metric

D. equal to recall

---

## Q6. Precision is defined as:

A. TP / (TP + FN)

B. TP / (TP + FP)

C. (TP + TN) / all samples

D. FP / (FP + TN)

---

## Q7. Recall is high when the model:

A. makes very few positive predictions

B. has few false positives

C. catches most of the actual positives

D. predicts everything as negative

---

## Q8. The f1-score is:

A. the arithmetic mean of accuracy and recall

B. the same as accuracy

C. always higher than precision

D. the harmonic mean of precision and recall

---

## Q9. Lowering the decision threshold generally:

A. increases recall but lowers precision

B. increases both precision and recall

C. has no effect

D. increases precision but lowers recall

---

## Q10. The AUC of a purely random classifier is:

A. 1.0

B. 0.5

C. 0.0

D. dependent on class balance

---

## Q11. To make grid search optimize for AUC, set:

A. `cv="roc_auc"`

B. `metric="auc"`

C. `scoring="roc_auc"`

D. `average="auc"`

---

## Q12. In a confusion matrix, the **diagonal** entries are:

A. the false positives

B. the false negatives

C. all of the errors

D. the correct predictions

---

<!-- _class: lead -->

## Answer Key

# Solutions & Explanations

---

## Answers 1–6

| Q | Ans | Why |
|---|:---:|-----|
| 1 | **A** | CV averages several splits → a **stable** estimate |
| 2 | **B** | Stratified folds preserve **class proportions** |
| 3 | **C** | Tuning on the test set **leaks** it → optimistic scores |
| 4 | **D** | `best_score_` = mean **CV score on training** data |
| 5 | **A** | "Always negative" already scores 99% → accuracy misleads |
| 6 | **B** | Precision = **TP / (TP + FP)** |

---

## Answers 7–12

| Q | Ans | Why |
|---|:---:|-----|
| 7 | **C** | Recall = fraction of **actual positives** caught |
| 8 | **D** | f1 = **harmonic mean** of precision and recall |
| 9 | **A** | Lower threshold → more positives → ↑recall, ↓precision |
| 10 | **B** | Random classifier → **AUC 0.5**, regardless of balance |
| 11 | **C** | Set **`scoring="roc_auc"`** in grid search / CV |
| 12 | **D** | Diagonal entries = **correct** predictions (TP, TN) |

---

<!-- _class: lead -->

# Well done!

### Review Chapter 5 for any you missed
