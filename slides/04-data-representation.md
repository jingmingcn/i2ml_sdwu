---
marp: true
theme: course
paginate: true
footer: 'Ch.4 · Representing Data & Engineering Features'
---

<!-- _class: lead -->

# Chapter 4
## Representing Data &
## Engineering Features

Turning raw data into features
models can use

---

## Learning Objectives

- Encode **categorical** variables correctly
- Use **binning**, **interactions**, and **polynomial** features
- Apply **nonlinear transformations**
- **Select** the most useful features automatically

> How you represent your data often matters more than which model you use.

---

## Categorical Variables

- Models need numbers, but categories aren't numeric
- ✘ Encoding categories as 1,2,3… implies a false ordering
- ✔ Use **one-hot encoding** (dummy variables): one 0/1 column per category

```python
import pandas as pd
pd.get_dummies(df)                       # pandas
# or
from sklearn.preprocessing import OneHotEncoder
```

---

## Numbers Can Encode Categoricals

- A column of integers may really be **categorical** (e.g. `workclass=0..4`)
- pandas `get_dummies` only encodes object/string columns by default
- Cast to string or specify columns so codes get one-hot encoded

```python
df['category'] = df['category'].astype(str)
pd.get_dummies(df, columns=['category'])
```

---

## Binning (Discretization)

- Split a continuous feature into **bins** → categorical
- Lets **linear models** capture nonlinear effects
- Trees gain little from binning (they can split anywhere already)

```python
import numpy as np
bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)
```

---

## Interactions & Polynomial Features

- **Interactions:** products of features (e.g. `x1 * x2`)
- **Polynomials:** `x, x², x³, …` → linear model fits smooth curves

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10).fit(X)
X_poly = poly.transform(X)
```

Powerful with linear models; less useful with tree ensembles.

---

## Univariate Nonlinear Transformations

- Functions like **log**, **exp**, **sqrt** reshape distributions
- Especially helpful for **skewed count data** with linear models & Naive Bayes
- Goal: make features look more **Gaussian**

```python
X_log = np.log(X + 1)
```

Tree-based models rarely need these transforms.

---

## Automatic Feature Selection

Fewer, better features → simpler, faster, more generalizable models.

Three strategies:

1. **Univariate statistics** — keep features most related to the target
2. **Model-based selection** — use a model's importances/coefficients
3. **Iterative selection** — add/remove features step by step (RFE)

---

## 1 · Univariate Statistics

- Test each feature's relationship to the target **independently**
- Fast; misses features useful only **in combination**

```python
from sklearn.feature_selection import SelectPercentile
sel = SelectPercentile(percentile=50).fit(X_train, y_train)
X_train_sel = sel.transform(X_train)
```

---

## 2 · Model-Based Selection

- Fit a model that scores feature importance, keep the top ones
- Considers features **together**

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
sel = SelectFromModel(RandomForestClassifier(n_estimators=100),
                      threshold='median').fit(X_train, y_train)
```

---

## 3 · Iterative Selection (RFE)

- Repeatedly fit and drop the weakest feature (or add the best)
- Most expensive; often the best-performing

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(RandomForestClassifier(n_estimators=100),
          n_features_to_select=40).fit(X_train, y_train)
```

---

## Utilizing Expert Knowledge

- Domain knowledge suggests features the data alone won't reveal
- Example: add **holiday** / **day-of-week** flags to forecast demand
- Encode known structure (seasonality, physical limits) as features
- The best features often come from **understanding the problem**

---

## Chapter Summary

- **One-hot encode** categoricals; watch out for integer-coded categories
- **Binning, interactions, polynomials** help linear models fit nonlinearities
- **Log/other transforms** tame skewed features
- **Select features** with univariate, model-based, or iterative methods
- **Expert knowledge** creates features data can't

**Next:** Chapter 5 — Model Evaluation & Improvement
