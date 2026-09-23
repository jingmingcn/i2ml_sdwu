---
marp: true
theme: course
paginate: true
footer: 'Introduction to Machine Learning · Ch.6 Exercises'
---

<!-- _class: lead -->

# Chapter 6 — Exercises

## Multiple-Choice Questions

Algorithm chains & pipelines

---

## How to Use These Slides

- **12 multiple-choice** questions on Chapter 6 concepts
- Pick **one** answer (A–D) for each question
- The **answer key** with brief explanations is at the end
- Try to answer before revealing the solutions

> Topics: data leakage, the `Pipeline` class, parameter naming, `make_pipeline`, `named_steps`, and grid-searching workflows.

---

## Q1. Scaling **all** training data before cross-validation causes:

A. data leakage and over-optimistic scores

B. faster convergence

C. underfitting

D. no problem at all

---

## Q2. Any step that learns from data must be fit on:

A. the whole dataset

B. the test folds

C. the training folds only

D. a random subset of features

---

## Q3. A `Pipeline` chains steps into:

A. several independent models

B. a single estimator with `fit` / `predict` / `score`

C. a plotting utility

D. a cross-validation splitter

---

## Q4. In a pipeline, every step **except the last** must implement:

A. `predict`

B. `score`

C. `fit_proba`

D. `transform`

---

## Q5. To tune the `C` of an SVM step named `svm`, the grid key is:

A. `svm__C`

B. `svm.C`

C. `C__svm`

D. `svm-C`

---

## Q6. `make_pipeline` names each step:

A. with numbers only

B. automatically, using the lowercase class name

C. after its parameters

D. randomly

---

## Q7. To inspect a fitted step inside a pipeline, use:

A. `best_params_`

B. `steps_only`

C. `named_steps`

D. `get_dummies`

---

## Q8. Compared to scaling leakage, feature-selection leakage is:

A. identical in effect

B. always harmless

C. easier to spot

D. far worse — it can make random data look predictive

---

## Q9. To grid-search **which model** to use, you pass:

A. a list of parameter-grid dictionaries

B. a single float

C. two separate test sets

D. no `param_grid` at all

---

## Q10. Setting a pipeline step to `None` in a grid means:

A. delete the pipeline

B. skip that step

C. reset all parameters

D. use the default scaler

---

## Q11. Inside cross-validation, a pipeline refits its scaler:

A. once on all the data

B. never

C. in every fold

D. only on the test set

---

## Q12. Adding more parameters to a grid search makes the number of fits:

A. stay the same

B. decrease

C. grow linearly

D. grow exponentially

---

<!-- _class: lead -->

## Answer Key

# Solutions & Explanations

---

## Answers 1–6

| Q | Ans | Why |
|---|:---:|-----|
| 1 | **A** | The scaler sees validation folds → **leakage** |
| 2 | **C** | Learn preprocessing on **training folds only** |
| 3 | **B** | `Pipeline` = one estimator (`fit`/`predict`/`score`) |
| 4 | **D** | All but the last step need a **`transform`** method |
| 5 | **A** | Use **`svm__C`** (step name + `__` + parameter) |
| 6 | **B** | `make_pipeline` uses the **lowercase class name** |

---

## Answers 7–12

| Q | Ans | Why |
|---|:---:|-----|
| 7 | **C** | **`named_steps`** maps step names → fitted estimators |
| 8 | **D** | Feature-selection leakage is **far worse** than scaling |
| 9 | **A** | Pass a **list of dicts** to swap estimators |
| 10 | **B** | `None` means **skip** that step (e.g. no scaling) |
| 11 | **C** | The scaler is refit **per fold** → no leakage |
| 12 | **D** | Combinations multiply → **exponential** growth |

---

<!-- _class: lead -->

# Well done!

### Review Chapter 6 for any you missed
