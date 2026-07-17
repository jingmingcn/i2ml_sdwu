---
marp: true
theme: course
paginate: true
footer: 'Ch.2 · Supervised Learning'
---

<!-- _class: lead -->

# Chapter 2
## Supervised Learning

Classification, regression,
and the major algorithm families

---

## Learning Objectives

- Tell **classification** from **regression**
- Understand **overfitting**, **underfitting**, and generalization
- Know the strengths/weaknesses of the main algorithm families
- Read model **uncertainty** estimates

---

## Classification vs. Regression

<div class="cols">
<div>

### Classification
- Predict a **class label** (discrete)
- *Binary*: yes/no, spam/not-spam
- *Multiclass*: which iris species

</div>
<div>

### Regression
- Predict a **continuous number**
- Income, temperature, house price
- Ordered output — "close" predictions are good predictions

</div>
</div>

Quick test: is there continuity between outputs? → regression.

---

## Generalization, Overfitting, Underfitting

- **Generalization:** performing well on *unseen* data — the whole goal
- **Overfitting:** model too complex → memorizes training noise
- **Underfitting:** model too simple → misses real patterns

> The sweet spot balances model complexity against the amount of data.
> More data → you can afford a more complex model.

---

## Model Complexity vs. Dataset Size

- With **more data**, more complex models generalize better
- With **little data**, prefer simpler models
- Collecting more/varied data often beats tuning the algorithm

---

## k-Nearest Neighbors (k-NN)

- Prediction = vote (classification) or average (regression) of *k* nearest points
- **Small k** → complex, noisy boundary (risk of overfitting)
- **Large k** → smooth boundary (risk of underfitting)

```python
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
```

✔ Easy to understand  ✘ Slow on large/high-dimensional data

---

## Linear Models

Prediction is a weighted sum of features: `ŷ = w·x + b`

- **Linear/Ridge/Lasso** regression — for continuous targets
- **Logistic Regression, Linear SVM** — for classification
- **Regularization** controls complexity:
  - *Ridge (L2)* — shrinks weights
  - *Lasso (L1)* — shrinks some weights to **zero** (feature selection)

```python
from sklearn.linear_model import Ridge
Ridge(alpha=1.0).fit(X_train, y_train)   # higher alpha = simpler
```

Fast, scalable, strong on high-dimensional / sparse data.

---

## Naive Bayes Classifiers

- Very fast to train — great baselines, especially for **text**
- Assume features are independent given the class ("naive")
- Variants: `GaussianNB`, `BernoulliNB`, `MultinomialNB`

✔ Blazing fast, works with high dimensions
✘ Predicted probabilities are not well-calibrated

---

## Decision Trees

- Learn a hierarchy of **if/else** questions
- Highly interpretable; no scaling needed
- **Unrestricted trees overfit** → limit with `max_depth`, `min_samples_leaf` (pre-pruning)

```python
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
```

Gives **feature importances** — how much each feature drives splits.

---

## Ensembles of Decision Trees

<div class="cols">
<div>

### Random Forests
- Many trees on random subsets of data & features
- Averaging reduces overfitting
- Strong, robust default

</div>
<div>

### Gradient Boosting
- Trees built **sequentially**, each fixing prior errors
- Often top accuracy; needs tuning
- Key knobs: `learning_rate`, `n_estimators`, `max_depth`

</div>
</div>

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
```

---

## Kernelized Support Vector Machines (SVM)

- Extend linear SVMs to **nonlinear** boundaries via kernels (e.g. RBF)
- Powerful on medium-sized datasets
- **Sensitive to scaling** — always preprocess features
- Key parameters: `C` (regularization) and `gamma` (kernel width)

```python
from sklearn.svm import SVC
SVC(kernel='rbf', C=10, gamma=0.1).fit(X_train_scaled, y_train)
```

---

## Neural Networks (Deep Learning)

- Stacked layers of weighted sums + nonlinearities
- Can build very complex models given enough data & compute
- scikit-learn: `MLPClassifier` / `MLPRegressor`
- **Scale inputs**; tune `hidden_layer_sizes`, `alpha`, `activation`

```python
from sklearn.neural_network import MLPClassifier
MLPClassifier(hidden_layer_sizes=[100, 100], alpha=0.01).fit(Xs, y)
```

For large-scale deep learning, use dedicated libraries (Keras, PyTorch).

---

## Uncertainty Estimates

Beyond the predicted class, ask *how confident?*

```python
model.decision_function(X_test)   # signed distance to boundary
model.predict_proba(X_test)       # calibrated-ish probabilities [0,1]
```

- Crucial when the **cost of errors is asymmetric** (e.g. medical tests)
- Lets you set custom decision thresholds

---

## Choosing an Algorithm — Rules of Thumb

- **Start simple:** linear models / Naive Bayes → fast baseline
- **Tabular, need accuracy:** random forest or gradient boosting
- **Small–medium, scaled features:** SVM
- **Lots of data / complex patterns:** neural networks
- Always **scale features** for k-NN, SVM, and neural nets

---

## Chapter Summary

- Supervised = classification (labels) or regression (numbers)
- Manage the **complexity ↔ generalization** trade-off
- Each family has trade-offs in speed, accuracy, interpretability, scaling
- Tree ensembles are strong general-purpose defaults
- Use **uncertainty estimates** when error costs differ

**Next:** Chapter 3 — Unsupervised Learning & Preprocessing
