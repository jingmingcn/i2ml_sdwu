---
marp: true
theme: course
paginate: true
footer: 'Introduction to Machine Learning · Ch.3 Exercises'
---

<!-- _class: lead -->

# Chapter 3 — Exercises

## Multiple-Choice Questions

Unsupervised learning & preprocessing

---

## How to Use These Slides

- **12 multiple-choice** questions on Chapter 3 concepts
- Pick **one** answer (A–D) for each question
- The **answer key** with brief explanations is at the end
- Try to answer before revealing the solutions

> Topics: scaling, PCA / NMF / t-SNE, k-Means, agglomerative clustering, DBSCAN, and evaluating clusters.

---

## Q1. The main challenge of unsupervised learning is that it:

A. always needs a GPU

B. is hard to evaluate because there are no labels

C. only works on image data

D. requires more features than samples

---

## Q2. `StandardScaler` transforms each feature to have:

A. mean 0 and variance 1

B. values exactly in [0, 1]

C. unit length per sample

D. only integer values

---

## Q3. A scaler should be fit on:

A. the test data

B. the whole dataset at once

C. the training data only

D. each fold's test portion

---

## Q4. PCA orders its principal components by:

A. alphabetical order

B. the number of samples

C. correlation with the labels

D. the amount of variance they capture

---

## Q5. t-SNE is primarily used for:

A. visualizing high-dimensional data in 2-D

B. transforming new data for a model

C. supervised classification

D. feature scaling

---

## Q6. `MinMaxScaler` shifts the data into the range:

A. mean 0, variance 1

B. [-1, 1]

C. [0, 1]

D. unbounded

---

## Q7. k-Means clustering requires you to:

A. label some of the data first

B. specify the number of clusters in advance

C. scale the target variable

D. build a dendrogram

---

## Q8. An advantage of DBSCAN is that it:

A. always needs the number of clusters set

B. only finds spherical clusters

C. cannot detect outliers

D. finds arbitrary shapes and flags noise, without setting a cluster count

---

## Q9. Compared to PCA, NMF produces components that are:

A. non-negative and additive ("parts")

B. always orthogonal

C. ordered by variance

D. computed from the labels

---

## Q10. To evaluate clustering against **known labels**, use:

A. `accuracy_score`

B. `r2_score`

C. `adjusted_rand_score` (ARI)

D. `mean_squared_error`

---

## Q11. Why is `accuracy_score` wrong for evaluating clusters?

A. It is too slow to compute

B. Cluster label numbers are arbitrary

C. It needs probability estimates

D. It only works for regression

---

## Q12. Agglomerative clustering works by:

A. randomly assigning points to k groups

B. iteratively updating cluster centers

C. expanding dense regions

D. bottom-up merging of the closest clusters

---

<!-- _class: lead -->

## Answer Key

# Solutions & Explanations

---

## Answers 1–6

| Q | Ans | Why |
|---|:---:|-----|
| 1 | **B** | No labels → no ground truth → hard to evaluate |
| 2 | **A** | `StandardScaler` → mean 0, variance 1 per feature |
| 3 | **C** | Fit on **training only**; fitting on test = leakage |
| 4 | **D** | Components are ordered by **variance** captured |
| 5 | **A** | t-SNE is a **visualization** method (no `transform`) |
| 6 | **C** | `MinMaxScaler` squeezes data into **[0, 1]** |

---

## Answers 7–12

| Q | Ans | Why |
|---|:---:|-----|
| 7 | **B** | k-Means needs the **number of clusters** up front |
| 8 | **D** | DBSCAN finds any shape, flags **noise**, no `k` needed |
| 9 | **A** | NMF components are **non-negative** and additive |
| 10 | **C** | Use **ARI** / NMI with ground truth (never accuracy) |
| 11 | **B** | Cluster label **numbers are arbitrary** |
| 12 | **D** | Agglomerative = **bottom-up merging** → dendrogram |

---

<!-- _class: lead -->

# Well done!

### Review Chapter 3 for any you missed
