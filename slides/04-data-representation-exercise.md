---
marp: true
theme: course
paginate: true
footer: 'Introduction to Machine Learning · Ch.4 Exercises'
---

<!-- _class: lead -->

# Chapter 4 — Exercises

## Multiple-Choice Questions

Representing data & engineering features

---

## How to Use These Slides

- **12 multiple-choice** questions on Chapter 4 concepts
- Pick **one** answer (A–D) for each question
- The **answer key** with brief explanations is at the end
- Try to answer before revealing the solutions

> Topics: categorical encoding, binning, interactions & polynomials, transformations, feature selection, and expert knowledge.

---

## Q1. Why not encode categories as integers 0, 1, 2, …?

A. It implies a false order and spacing between categories

B. Integers take too much memory

C. scikit-learn rejects integer columns

D. It always causes overfitting

---

## Q2. One-hot encoding represents a categorical feature as:

A. a single integer column

B. one 0/1 column per category

C. the mean of the target per category

D. a floating-point average

---

## Q3. By default, `pd.get_dummies` encodes:

A. all numeric columns

B. only the target column

C. object / string (categorical) columns

D. nothing until the data is scaled

---

## Q4. Binning a continuous feature mainly benefits:

A. linear models (it adds flexibility)

B. decision trees

C. random forests

D. all models equally

---

## Q5. An interaction feature is typically:

A. the logarithm of a feature

B. a feature minus its mean

C. a one-hot column

D. the product of two features

---

## Q6. A log transform is especially helpful for:

A. tree-based models

B. skewed count data used with linear models

C. already-Gaussian features

D. categorical features

---

## Q7. Univariate feature selection can miss features that are:

A. strongly correlated with the target

B. high-variance

C. useful only in combination with other features

D. numeric

---

## Q8. Model-based selection (`SelectFromModel`) uses:

A. a model's feature importances or coefficients

B. a single statistical test per feature

C. random sampling of features

D. the mean of the target

---

## Q9. Recursive Feature Elimination (RFE):

A. is the cheapest selection method

B. selects features without any model

C. only works with linear models

D. iteratively fits and drops the weakest feature

---

## Q10. In the Citi Bike case, a random forest using only POSIX time:

A. achieved near-perfect accuracy

B. failed because trees can't extrapolate beyond the training range

C. needed no features at all

D. worked only after scaling

---

## Q11. Which features dramatically improved the Citi Bike model?

A. The raw timestamp squared

B. Random noise features

C. Hour of day and day of week

D. The POSIX time in milliseconds

---

## Q12. For a `LinearRegression` on hour/day, the best encoding was:

A. one-hot encoding (plus interactions)

B. raw integers

C. no encoding at all

D. standardized integers

---

<!-- _class: lead -->

## Answer Key

# Solutions & Explanations

---

## Answers 1–6

| Q | Ans | Why |
|---|:---:|-----|
| 1 | **A** | Integer codes imply a **false order** the model would use |
| 2 | **B** | One-hot = one **0/1 column per category** |
| 3 | **C** | `get_dummies` encodes **string/object** columns by default |
| 4 | **A** | Binning lets **linear models** fit nonlinear effects |
| 5 | **D** | An interaction feature is a **product** of features |
| 6 | **B** | Log tames **skewed count** data for linear models |

---

## Answers 7–12

| Q | Ans | Why |
|---|:---:|-----|
| 7 | **C** | Univariate tests miss features useful **in combination** |
| 8 | **A** | `SelectFromModel` uses **importances / coefficients** |
| 9 | **D** | RFE **iteratively drops** the weakest feature (expensive) |
| 10 | **B** | Trees **can't extrapolate** → POSIX time fails on new dates |
| 11 | **C** | **Hour of day** and **day of week** captured the pattern |
| 12 | **A** | **One-hot (+ interactions)** matched the random forest |

---

<!-- _class: lead -->

# Well done!

### Review Chapter 4 for any you missed
