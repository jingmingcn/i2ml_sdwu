---
marp: true
theme: course
paginate: true
footer: 'Introduction to Machine Learning · Ch.1 Exercises'
---

<!-- _class: lead -->

# Chapter 1 — Exercises

## Multiple-Choice Questions

Test your understanding of the Introduction

---


## Q1. What is machine learning best described as?

A. Storing large amounts of data efficiently

B. Extracting knowledge and patterns from data

C. Writing explicit rules for every possible situation

D. Manually labeling every data point by hand

---

## Q1. What is machine learning best described as?

A. Storing large amounts of data efficiently

__B. Extracting knowledge and patterns from data__

C. Writing explicit rules for every possible situation

D. Manually labeling every data point by hand

Answer: **B** — Machine learning is about extracting knowledge and patterns from data, rather than hand-coding rules.

---

## Q2. Which of these is a *supervised* learning task?

A. Grouping customers into segments not known in advance

B. Discovering unknown topics in a set of blog posts

C. Predicting whether a tumor is benign or malignant from labeled images

D. Flagging abnormal website access with no examples of "abnormal"

---

## Q2. Which of these is a *supervised* learning task?

A. Grouping customers into segments not known in advance

B. Discovering unknown topics in a set of blog posts

__C. Predicting whether a tumor is benign or malignant from labeled images__

D. Flagging abnormal website access with no examples of "abnormal"

Answer: **C** — Supervised learning requires labeled input/output pairs, such as images labeled as benign or malignant.

---

## Q3. Why do handcoded rules fail for tasks like face detection?

A. They run too slowly on modern hardware

B. They require a GPU to execute

C. They can only be written in Python

D. Humans can't express the decision as explicit rules over pixels

---

## Q3. Why do handcoded rules fail for tasks like face detection?

A. They run too slowly on modern hardware

B. They require a GPU to execute

C. They can only be written in Python

__D. Humans can't express the decision as explicit rules over pixels__

Answer: **D** — Humans cannot write explicit rules for pixel-level decisions like face detection, so we rely on machine learning to learn from examples.

---

## Q4. In a data table, a **row** and a **column** represent:

A. a feature and a sample, respectively

B. a sample and a feature, respectively

C. a label and a model, respectively

D. a parameter and a target, respectively

---

## Q4. In a data table, a **row** and a **column** represent:

A. a feature and a sample, respectively

__B. a sample and a feature, respectively__

C. a label and a model, respectively

D. a parameter and a target, respectively

Answer: **B** — In a data table, rows correspond to samples (data points), and columns correspond to features (attributes of the data).

---

## Q5. Why do we split data into training and test sets?

A. To estimate how well the model generalizes to unseen data

B. To make the model train faster

C. To reduce the number of features

D. To guarantee the classes are perfectly balanced

---

## Q5. Why do we split data into training and test sets?

__A. To estimate how well the model generalizes to unseen data__

B. To make the model train faster

C. To reduce the number of features

D. To guarantee the classes are perfectly balanced

Answer: **A** — The test set is used to evaluate the model's performance on unseen data, providing an estimate of its generalization ability.

---

## Q6. Classifying an iris as setosa / versicolor / virginica is:

A. a regression problem

B. a binary classification problem

C. a multiclass classification problem

D. an unsupervised clustering problem

---

## Q6. Classifying an iris as setosa / versicolor / virginica is:

A. a regression problem

B. a binary classification problem

__C. a multiclass classification problem__

D. an unsupervised clustering problem

Answer: **C** — Since there are three possible classes (setosa, versicolor, virginica), this is a multiclass classification problem.

---

## Q7. The common scikit-learn workflow for a model is:

A. load → save → print

B. fit → predict → score

C. map → reduce → filter

D. compile → run → debug

---

## Q7. The common scikit-learn workflow for a model is:

A. load → save → print

__B. fit → predict → score__

C. map → reduce → filter

D. compile → run → debug

Answer: **B** — The typical workflow in scikit-learn involves fitting the model to training data, making predictions on new data, and scoring the model's performance.

---

## Q8. With `n_neighbors=1`, k-NN predicts a new point as:

A. the class of the single nearest training point

B. the average of all training labels

C. the majority class of the whole dataset

D. a randomly chosen class

---

## Q8. With `n_neighbors=1`, k-NN predicts a new point as:

__A. the class of the single nearest training point__

B. the average of all training labels

C. the majority class of the whole dataset

D. a randomly chosen class

Answer: **A** — In 1-NN (k-nearest neighbors with k=1), the predicted class for a new point is simply the class of its closest training point.

---

## Q9. Which library provides the **DataFrame** for tabular data?

A. NumPy

B. SciPy

C. pandas

D. matplotlib

---

## Q9. Which library provides the **DataFrame** for tabular data?

A. NumPy

B. SciPy

__C. pandas__

D. matplotlib

Answer: **C** — The pandas library provides the DataFrame structure, which is ideal for handling tabular data, while NumPy provides arrays.


---

## Q10. A defining feature of *unsupervised* learning is:

A. it always achieves higher accuracy

B. it needs labeled input/output pairs

C. it requires a separate test set in order to train

D. it has no known outputs and finds structure in the data

---

## Q10. A defining feature of *unsupervised* learning is:

A. it always achieves higher accuracy

B. it needs labeled input/output pairs

C. it requires a separate test set in order to train

__D. it has no known outputs and finds structure in the data__

Answer: **D** — Unsupervised learning does not rely on labeled outputs; instead, it seeks to find patterns or structure in the input data.

---

## Q11. `load_iris()` returns the dataset as:

A. a pandas DataFrame

B. a `Bunch` object (dictionary-like)

C. a plain Python list

D. a CSV file on disk

---

## Q11. `load_iris()` returns the dataset as:

A. a pandas DataFrame

__B. a `Bunch` object (dictionary-like)__

C. a plain Python list

D. a CSV file on disk

Answer: **B** — The `load_iris()` function from scikit-learn returns the dataset as a `Bunch` object, which is similar to a dictionary and contains the data, target labels, and other metadata.

---

## Q12. For a classifier, `knn.score(X_test, y_test)` returns:

A. the number of neighbors used

B. the predicted class labels

C. the accuracy on the test set

D. the training-set error

---

## Q12. For a classifier, `knn.score(X_test, y_test)` returns:

A. the number of neighbors used

B. the predicted class labels

__C. the accuracy on the test set__

D. the training-set error

Answer: **C** — The `score` method for a classifier in scikit-learn computes the accuracy of the model on the provided test data and labels.

---

<!-- _class: lead -->

## Answer Key

# Solutions & Explanations

---

## Answers 1–6

| Q | Ans | Why |
|---|:---:|-----|
| 1 | **B** | ML *extracts knowledge from data* instead of hand-coding rules |
| 2 | **C** | Labeled input→output pairs make it supervised |
| 3 | **D** | No one can write pixel-level rules for a face → learn from examples |
| 4 | **B** | Data is **samples × features** — rows are samples, columns features |
| 5 | **A** | The test set measures **generalization** to unseen data |
| 6 | **C** | Three possible classes → **multiclass** classification |

---

## Answers 7–12

| Q | Ans | Why |
|---|:---:|-----|
| 7 | **B** | Every estimator follows **`fit` → `predict` → `score`** |
| 8 | **A** | 1-NN copies the label of the **single closest** training point |
| 9 | **C** | **pandas** provides `DataFrame`; NumPy provides arrays |
| 10 | **D** | Unsupervised learning has **no labels** — it finds structure |
| 11 | **B** | `load_iris()` returns a **`Bunch`** (dict-like) with `data`, `target`, … |
| 12 | **C** | For a classifier, `score` returns **accuracy** on the given data |

---

<!-- _class: lead -->

# Well done!

### Review Chapter 1 for any you missed
