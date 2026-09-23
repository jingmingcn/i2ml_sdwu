---
marp: true
theme: course
paginate: true
footer: 'Introduction to Machine Learning · Ch.2 Exercises'
---

<!-- _class: lead -->

# Chapter 2 — Exercises

## Multiple-Choice Questions

Supervised learning algorithms

---

## How to Use These Slides

- **12 multiple-choice** questions on Chapter 2 concepts
- Pick **one** answer (A–D) for each question
- The **answer key** with brief explanations is at the end
- Try to answer before revealing the solutions

> Topics: classification vs regression, overfitting, k-NN, linear models, trees & ensembles, SVMs, neural nets, and uncertainty.

---

## Q1. Which task is a *regression* problem?

A. Deciding whether an email is spam or not

B. Classifying the species of an iris

C. Predicting the sale price of a house

D. Detecting whether a transaction is fraud

---

## Q2. "Overfitting" means a model is:

A. too complex and memorizes training noise, generalizing poorly

B. too simple to capture the real pattern

C. always bad on the training data

D. built with too few features

---

## Q3. With k-NN, a very small `k` (e.g. 1) tends to give:

A. a very smooth, simple decision boundary

B. a complex, jagged boundary that can overfit

C. identical predictions everywhere

D. a strictly linear boundary

---

## Q4. In Ridge regression, increasing `alpha`:

A. removes the intercept term

B. always improves the test score

C. makes coefficients larger

D. shrinks coefficients toward zero (a simpler model)

---

## Q5. A distinctive property of Lasso (L1) is that it:

A. never changes the coefficients

B. sets some coefficients exactly to zero (feature selection)

C. requires the target to be scaled

D. only works for classification

---

## Q6. Despite its name, `LogisticRegression` is:

A. a clustering method

B. a regression model for continuous output

C. a classification algorithm

D. a preprocessing transformer

---

## Q7. For `LogisticRegression` / `LinearSVC`, a **high** value of `C`:

A. fits the training data harder (less regularization)

B. forces all coefficients to zero

C. has no effect on the model

D. always causes underfitting

---

## Q8. Which models are **sensitive to feature scaling**?

A. Decision trees

B. Random forests

C. Gradient boosting

D. SVMs, k-NN, and neural networks

---

## Q9. An unrestricted decision tree on the training data typically:

A. underfits badly

B. reaches 100% training accuracy and overfits

C. cannot be constructed

D. ignores the target values

---

## Q10. Random forests reduce overfitting mainly by:

A. using a single very deep tree

B. removing all features

C. averaging many trees trained on random subsets

D. scaling the data first

---

## Q11. Gradient boosted trees are built:

A. serially, each tree correcting the previous ones' errors

B. all at once in parallel, then averaged

C. with no depth limit by default

D. only for regression tasks

---

## Q12. `predict_proba` returns:

A. the single predicted class label

B. a probability per class, summing to 1

C. the distance to the decision boundary

D. the number of support vectors

---

<!-- _class: lead -->

## Answer Key

# Solutions & Explanations

---

## Answers 1–6

| Q | Ans | Why |
|---|:---:|-----|
| 1 | **C** | House price is a **continuous** value → regression |
| 2 | **A** | Overfitting = too complex, memorizes noise, poor generalization |
| 3 | **B** | Small `k` → complex, jagged boundary (high variance) |
| 4 | **D** | Higher `alpha` → more L2 regularization → smaller coefficients |
| 5 | **B** | L1 drives some coefficients to **zero** → feature selection |
| 6 | **C** | `LogisticRegression` is a **classifier**, not a regressor |

---

## Answers 7–12

| Q | Ans | Why |
|---|:---:|-----|
| 7 | **A** | High `C` → less regularization → fits training data harder |
| 8 | **D** | SVMs, k-NN, and neural nets need **scaled** features |
| 9 | **B** | A full tree splits until pure → 100% train, overfits |
| 10 | **C** | Averaging many **randomized** trees cancels overfitting |
| 11 | **A** | Boosting builds trees **serially**, fixing prior errors |
| 12 | **B** | `predict_proba` → per-class probabilities summing to 1 |

---

<!-- _class: lead -->

# Well done!

### Review Chapter 2 for any you missed
