---
marp: true
theme: course
paginate: true
footer: 'Ch.1 · Introduction'
---

<!-- _class: lead -->

# Chapter 1
## Introduction

Why machine learning, the Python stack,
and your first classifier

---

## Learning Objectives

By the end of this chapter you can:

- Explain what machine learning is and when to use it
- Distinguish **supervised** from **unsupervised** learning
- Set up the Python data-science stack
- Build, train, and evaluate a first model with scikit-learn

---

## Why Machine Learning?

- Early intelligent systems were **hand-coded rules** — expensive and brittle
- Rules fail when:
  - the logic is domain-specific and hard to enumerate
  - the task changes over time
  - humans can't articulate *how* they decide (e.g. recognizing faces)
- **ML learns the decision logic from data** instead of us writing it

---

## Two Main Kinds of ML

<div class="cols">
<div>

### Supervised
- Learn from **input → known output** pairs
- Model predicts outputs for new inputs
- Examples: spam detection, tumor diagnosis, fraud detection

</div>
<div>

### Unsupervised
- Only inputs, **no labeled outputs**
- Find structure in data
- Examples: customer segmentation, topic discovery, anomaly detection

</div>
</div>

---

## Know Your Task and Your Data

Ask before modeling:

- What question am I trying to answer? Do I have the data to answer it?
- What is the **best phrasing** as an ML problem?
- Have I collected **enough** and **relevant** data?
- What **features** will I extract? Will they enable good predictions?
- How will I **measure success** in my application?

> The data you feed the model matters more than the algorithm you pick.

---

## The Python Stack

| Tool | Role |
|------|------|
| **scikit-learn** | ML algorithms & workflow |
| **NumPy** | arrays, linear algebra |
| **SciPy** | scientific computing, sparse matrices |
| **pandas** | tabular data (DataFrames) |
| **matplotlib** | plotting & visualization |
| **Jupyter** | interactive notebooks |

Install: `pip install numpy scipy scikit-learn matplotlib pandas`

---

## First Application: Classifying Iris Species

Goal: predict the species of an iris flower from measurements.

- **Features:** sepal length/width, petal length/width (cm)
- **Target:** species — *setosa, versicolor, virginica*
- A **supervised, multiclass classification** problem

```python
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape)      # (150, 4)
print(iris.target_names)    # ['setosa' 'versicolor' 'virginica']
```

---

## Training and Testing Data

Never evaluate on the data you trained on — split it first.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
```

- **Training set** → fit the model
- **Test set** → estimate how it performs on *unseen* data
- `random_state` makes the split reproducible

---

## Build Your First Model: k-Nearest Neighbors

Predict by looking at the *k* closest training points.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)          # train
```

The scikit-learn pattern you'll reuse everywhere:
**`fit`** → **`predict`** → **`score`**

---

## Predict and Evaluate

```python
# predict a brand-new flower
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
knn.predict(X_new)                 # -> setosa

# accuracy on the held-out test set
knn.score(X_test, y_test)          # ~0.97
```

**97% accuracy** — the model gets ~97 of every 100 unseen flowers right.

---

## Chapter Summary

- ML learns rules from **data**, not hand-coding
- **Supervised** = labeled data; **unsupervised** = find structure
- Split data into **train** / **test** to measure real performance
- Every scikit-learn model follows **`fit` / `predict` / `score`**
- Success depends on good data and clear problem framing

**Next:** Chapter 2 — Supervised Learning in depth
