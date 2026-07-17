---
marp: true
theme: course
paginate: true
footer: 'Ch.6 · Algorithm Chains & Pipelines'
---

<!-- _class: lead -->

# Chapter 6
## Algorithm Chains & Pipelines

Chaining preprocessing and models
without leaking data

---

## Learning Objectives

- See why preprocessing **before** cross-validation leaks data
- Build a **`Pipeline`** to chain steps safely
- Grid-search over preprocessing **and** model parameters together
- Use the convenient `make_pipeline` interface

---

## The Problem: Preprocessing Leakage

Scaling on **all** data before CV lets test info sneak into training.

```python
# WRONG: scaler saw the whole dataset
X_scaled = StandardScaler().fit_transform(X)
cross_val_score(SVC(), X_scaled, y, cv=5)   # over-optimistic!
```

In cross-validation, the scaler must be re-fit **inside each fold**.

---

## The Solution: Pipeline

Bundle preprocessing + model into one estimator.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([("scaler", StandardScaler()),
                 ("svm", SVC())])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

`fit` runs each step's `fit`/`transform` in order — no manual bookkeeping.

---

## Pipelines in Cross-Validation

Now scaling happens **within** each fold — correctly.

```python
from sklearn.model_selection import cross_val_score
cross_val_score(pipe, X, y, cv=5)   # scaler re-fit per fold
```

The whole chain is treated as a single model → no leakage.

---

## Using Pipelines in Grid Searches

Tune model (and preprocessing) params through the pipeline.
Reference a step's parameter as `stepname__param`.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"svm__C": [0.1, 1, 10],
              "svm__gamma": [0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
grid.best_params_
```

Double underscore `__` separates the step name from its parameter.

---

## Convenient Creation: make_pipeline

Auto-names steps after their classes.

```python
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), SVC(C=100))
# step names: 'standardscaler', 'svc'
pipe.steps                      # inspect the named steps
```

---

## Accessing Step Attributes

Reach inside a fitted pipeline via `named_steps`.

```python
pipe.named_steps["standardscaler"].mean_       # learned means
grid.best_estimator_.named_steps["svc"]         # tuned model
```

Works even after grid search, through `best_estimator_`.

---

## Grid-Searching Preprocessing *and* Model

Tune the whole workflow at once — even the number of PCA components.

```python
pipe = make_pipeline(StandardScaler(), PCA(), Ridge())
param_grid = {"pca__n_components": [5, 10, 20],
              "ridge__alpha": [0.1, 1, 10]}
GridSearchCV(pipe, param_grid, cv=5).fit(X_train, y_train)
```

---

## Grid-Searching *Which Model* To Use

A grid entry can even swap the estimator itself.

```python
param_grid = [
    {"classifier": [SVC()], "preprocessing": [StandardScaler()],
     "classifier__C": [1, 10]},
    {"classifier": [RandomForestClassifier()],
     "preprocessing": [None]},
]
```

Compare entirely different pipelines in **one** search.

---

## Chapter Summary

- Preprocessing outside CV **leaks** test information
- **`Pipeline`** chains steps and applies them correctly inside CV
- Grid-search params with `stepname__param`
- `make_pipeline` auto-names steps; `named_steps` inspects them
- You can tune preprocessing, model, and even model **choice** together

**Next:** Chapter 7 — Working with Text Data
