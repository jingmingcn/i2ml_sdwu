---
marp: true
theme: course
paginate: true
footer: 'Introduction to Machine Learning · Ch.8 Exercises'
---

<!-- _class: lead -->

# Chapter 8 — Exercises

## Multiple-Choice Questions

Wrapping up: the bigger picture

---

## How to Use These Slides

- **12 multiple-choice** questions on Chapter 8 concepts
- Pick **one** answer (A–D) for each question
- The **answer key** with brief explanations is at the end
- Try to answer before revealing the solutions

> Topics: approaching a problem, humans in the loop, production, custom estimators, scaling, and where to go next.

---

## Q1. Before running an algorithm, you should first:

A. frame the problem and decide how to measure success

B. pick the most complex model available

C. collect as many features as possible

D. deploy to production

---

## Q2. The "what if I built the perfect model?" question helps assess:

A. which algorithm is fastest

B. whether the problem is worth solving (its business impact)

C. the number of features to use

D. the learning rate

---

## Q3. "Humans in the loop" typically means:

A. humans label every single prediction

B. removing all automation

C. routing uncertain or complex cases to a person

D. training only on human-written data

---

## Q4. A/B testing is a form of:

A. offline evaluation on historical data

B. cross-validation

C. feature selection

D. online (live) evaluation with real users

---

## Q5. To build a scikit-learn-compatible transformer, inherit from:

A. `BaseEstimator` and `TransformerMixin`

B. `Pipeline` and `GridSearchCV`

C. `numpy.ndarray`

D. `StandardScaler`

---

## Q6. Out-of-core learning is used when:

A. the data has no labels

B. the data doesn't fit in RAM, so it's streamed in chunks

C. you need a GPU

D. the model is a decision tree

---

## Q7. For distributed learning across a cluster, a common tool is:

A. matplotlib

B. pandas

C. spark with MLlib

D. Jupyter

---

## Q8. Probabilistic programming languages include:

A. NumPy and SciPy

B. R and MATLAB

C. Go and Scala

D. PyMC and Stan

---

## Q9. For statistical modeling and inference in Python, consider:

A. statsmodels

B. matplotlib

C. Pillow

D. requests

---

## Q10. Ranking is the kind of learning behind:

A. image classification

B. search engines

C. clustering

D. data scaling

---

## Q11. Kaggle and OpenML are mainly useful for:

A. deploying production systems

B. writing documentation

C. practicing on datasets and competitions

D. scaling to clusters

---

## Q12. Moving a prototype to production often involves:

A. never changing the code

B. using only Jupyter notebooks

C. ignoring runtime and memory

D. reimplementing it in a high-performance language (Go/Scala/C++/Java)

---

<!-- _class: lead -->

## Answer Key

# Solutions & Explanations

---

## Answers 1–6

| Q | Ans | Why |
|---|:---:|-----|
| 1 | **A** | Start by **framing the problem** and defining success |
| 2 | **B** | It gauges the **business impact** — is it worth building? |
| 3 | **C** | Automate simple cases, **route hard ones to humans** |
| 4 | **D** | A/B testing is **online / live** evaluation |
| 5 | **A** | Inherit **`BaseEstimator` + `TransformerMixin`** |
| 6 | **B** | Out-of-core **streams data in chunks** (larger than RAM) |

---

## Answers 7–12

| Q | Ans | Why |
|---|:---:|-----|
| 7 | **C** | **spark + MLlib** distributes learning over a cluster |
| 8 | **D** | **PyMC** and **Stan** are probabilistic programming tools |
| 9 | **A** | **statsmodels** targets statistical modeling & inference |
| 10 | **B** | **Ranking** powers **search engines** |
| 11 | **C** | Kaggle / OpenML → **practice datasets & competitions** |
| 12 | **D** | Production often means **reimplementing** in a fast language |

---

<!-- _class: lead -->

# Well done!

### You've completed the course exercises!
