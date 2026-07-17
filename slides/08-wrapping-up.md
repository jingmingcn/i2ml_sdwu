---
marp: true
theme: course
paginate: true
footer: 'Ch.8 · Wrapping Up'
---

<!-- _class: lead -->

# Chapter 8
## Wrapping Up

From prototype to production,
and where to go next

---

## Learning Objectives

- Approach a real ML problem end-to-end
- Understand the human role in ML systems
- Move from prototype to production responsibly
- Know where to deepen your skills

---

## Approaching a Machine Learning Problem

1. **Frame** the question as an ML task
2. **Gather & understand** the data
3. **Build features** and a baseline model
4. **Evaluate** with the right metric & cross-validation
5. **Iterate** — improve data, features, model
6. **Deploy** and monitor

Most effort is in framing, data, and features — not the algorithm.

---

## Humans in the Loop

- ML augments human decisions; it rarely replaces judgment
- Keep humans reviewing **high-stakes** or **low-confidence** predictions
- Watch for **bias** in data → biased models
- Interpretability builds trust and catches errors

---

## From Prototype to Production

- Prototype notebook ≠ production system
- Production needs: reliability, latency, monitoring, retraining
- Consider engineering constraints (memory, speed) early
- The best model on paper isn't always the best in production

---

## Testing Production Systems

- **Offline evaluation** — historical/held-out data (what we've done)
- **Online evaluation** — live testing (e.g. **A/B tests**) with real users
- Monitor for **data drift** — inputs change → model degrades over time
- Plan for periodic **retraining**

---

## Building Your Own Estimator

- Implement the scikit-learn interface: `fit`, `predict`/`transform`
- Inherit from `BaseEstimator` + a mixin (`ClassifierMixin`, …)
- Then it plugs into **pipelines** and **grid search** for free

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): ...
```

---

## Where to Go From Here

- **Theory:** *The Elements of Statistical Learning* (Hastie et al.)
- **Other frameworks:** statsmodels, TensorFlow / Keras, PyTorch
- **More topics:** ranking, recommender systems, probabilistic programming
- **Neural networks / deep learning:** for images, audio, language
- **Scaling:** out-of-core & distributed learning for big data

---

## Honing Your Skills

- Practice on real datasets (Kaggle, OpenML, UCI)
- Enter competitions to benchmark yourself
- Build end-to-end projects, not just models
- Read code and papers; contribute to open source

> The best way to learn ML is to keep applying it.

---

## Course Summary

- ML = learn patterns from **data**; frame the problem well
- Master **supervised** & **unsupervised** methods and their trade-offs
- **Represent data** thoughtfully — features drive performance
- **Evaluate** honestly with CV, grid search, and the right metric
- Chain steps in **pipelines**; guard against **data leakage**
- Keep learning — this is a foundation, not the finish line

---

<!-- _class: lead -->

# Thank you!

### Questions?

*Introduction to Machine Learning with Python*
Andreas C. Müller & Sarah Guido
