---
marp: true
theme: course
paginate: true
footer: 'Ch.5 · Model Evaluation & Improvement'
---

<!-- _class: lead -->

# Chapter 5
## Model Evaluation & Improvement

Cross-validation, grid search,
and choosing the right metric

---

## Learning Objectives

- Evaluate models robustly with **cross-validation**
- Tune hyperparameters with **grid search**
- Avoid overfitting the parameters to the test set
- Pick **metrics** that match the business goal

---

## Beyond a Single Train/Test Split

- One split → the score depends on *which* points landed in the test set
- **Cross-validation** repeats the split many ways for a stable estimate

---

## k-Fold Cross-Validation

- Split data into **k folds**; each fold serves as test once
- Report the **mean** (and spread) of the k scores

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
scores.mean()
```

- **Stratified** k-fold keeps class ratios balanced (default for classification)

---

## Benefits of Cross-Validation

- More **reliable** performance estimate than one split
- Reveals how **sensitive** the model is to the data split (variance)
- Uses data efficiently — every point is used for both train and test
- ✘ Costs ~k× more compute

---

## Grid Search for Hyperparameters

- Systematically try combinations of parameters, keep the best
- ⚠️ Never tune on the test set — that leaks and inflates scores

```python
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
```

Split into **three**: train (fit) · validation (tune) · test (final report).

---

## Grid Search *with* Cross-Validation

Best practice — `GridSearchCV` combines tuning + CV.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

grid.best_params_                 # best combination
grid.score(X_test, y_test)        # honest final estimate
```

Test set is touched **once**, at the very end.

---

## Evaluation Metrics — Keep the Goal in Mind

- **Accuracy can mislead**, especially with **imbalanced classes**
- A 99%-negative dataset → a "always negative" model scores 99%
- Choose a metric tied to the real-world cost of each error type

---

## Metrics for Binary Classification

- **Confusion matrix:** TP, FP, TN, FN — the full picture
- **Precision** = TP / (TP + FP) — how many predicted positives are right
- **Recall** = TP / (TP + FN) — how many actual positives we caught
- **f1-score** = harmonic mean of precision & recall

```python
from sklearn.metrics import classification_report, confusion_matrix
```

---

## Precision–Recall & ROC Curves

- Models output scores → the **threshold** trades precision vs. recall
- **Precision–Recall curve** — good for imbalanced data
- **ROC curve** + **AUC** — threshold-independent summary (1.0 = perfect)

```python
from sklearn.metrics import roc_auc_score, average_precision_score
roc_auc_score(y_test, model.decision_function(X_test))
```

---

## Multiclass & Regression Metrics

<div class="cols">
<div>

### Multiclass
- Extend precision/recall via **macro / weighted** averaging
- Multiclass confusion matrix

</div>
<div>

### Regression
- **R²** (default `.score`)
- **MSE / MAE**
- Pick based on how you weigh large vs. small errors

</div>
</div>

---

## Using Metrics in Model Selection

Optimize grid search for the metric you actually care about:

```python
GridSearchCV(SVC(), param_grid, scoring='roc_auc', cv=5)
```

Common `scoring` values: `accuracy`, `f1`, `roc_auc`,
`average_precision`, `r2`, `neg_mean_squared_error`.

---

## Chapter Summary

- Use **cross-validation** for stable performance estimates
- Tune with **GridSearchCV**; keep the test set untouched until the end
- **Accuracy misleads** on imbalanced data
- Choose metrics (**precision, recall, f1, AUC, R²**) to match the goal
- Optimize grid search for that metric via `scoring`

**Next:** Chapter 6 — Algorithm Chains & Pipelines
