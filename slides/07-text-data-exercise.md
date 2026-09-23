---
marp: true
theme: course
paginate: true
footer: 'Introduction to Machine Learning · Ch.7 Exercises'
---

<!-- _class: lead -->

# Chapter 7 — Exercises

## Multiple-Choice Questions

Working with text data

---

## How to Use These Slides

- **12 multiple-choice** questions on Chapter 7 concepts
- Pick **one** answer (A–D) for each question
- The **answer key** with brief explanations is at the end
- Try to answer before revealing the solutions

> Topics: bag-of-words, `CountVectorizer`, `min_df` & stopwords, tf-idf, n-grams, stemming/lemmatization, and LDA.

---

## Q1. The bag-of-words representation discards:

A. the vocabulary

B. word order

C. the word counts

D. the documents themselves

---

## Q2. `CountVectorizer` produces:

A. a sparse matrix of word counts

B. a dense image

C. a single integer per document

D. a list of sentences

---

## Q3. Setting `min_df=5` means a token is kept only if it:

A. appears at most 5 times total

B. is exactly 5 characters long

C. appears in at least 5 documents

D. is among the top 5 words

---

## Q4. Stopwords are:

A. rare, highly informative words

B. words unique to one document

C. misspelled words

D. very frequent words that carry little signal

---

## Q5. tf-idf gives a **high** weight to a word that is:

A. frequent in one document but rare across the corpus

B. frequent in every document

C. absent from the corpus

D. always a stopword

---

## Q6. Using n-grams (e.g. bigrams) lets the model capture:

A. document length only

B. some word order and context

C. fewer features

D. the target labels directly

---

## Q7. The feature "not worth" is an example of:

A. a stopword

B. a stemmed token

C. a bigram that captures context

D. a topic

---

## Q8. Compared to stemming, lemmatization:

A. is always faster

B. ignores the dictionary

C. only trims common suffixes

D. uses a dictionary and the word's role for an accurate base form

---

## Q9. Latent Dirichlet Allocation (LDA) is used for:

A. unsupervised topic modeling

B. supervised classification

C. feature scaling

D. cross-validation

---

## Q10. The words tf-idf marks as "important" are:

A. always predictive of the label

B. distinctive, but not necessarily predictive of the label

C. the rarest single-document words

D. exactly the stopwords

---

## Q11. For high-dimensional sparse text features, a good model is:

A. a deep decision tree

B. k-nearest neighbors

C. a linear model like logistic regression

D. DBSCAN

---

## Q12. Adding trigrams to the vocabulary tends to cause:

A. fewer features

B. a guaranteed drop in accuracy

C. no change at all

D. an explosion in the number of features

---

<!-- _class: lead -->

## Answer Key

# Solutions & Explanations

---

## Answers 1–6

| Q | Ans | Why |
|---|:---:|-----|
| 1 | **B** | Bag-of-words keeps counts but **discards word order** |
| 2 | **A** | Output is a **sparse count matrix** |
| 3 | **C** | `min_df=5` → token must appear in **≥5 documents** |
| 4 | **D** | Stopwords = **very frequent**, low-signal words |
| 5 | **A** | High tf-idf: **frequent in a doc, rare in the corpus** |
| 6 | **B** | n-grams recover **word order / context** |

---

## Answers 7–12

| Q | Ans | Why |
|---|:---:|-----|
| 7 | **C** | "not worth" is a **bigram** that captures context |
| 8 | **D** | Lemmatization uses a **dictionary + word role** |
| 9 | **A** | LDA is **unsupervised topic modeling** |
| 10 | **B** | tf-idf is unsupervised → **distinctive ≠ predictive** |
| 11 | **C** | High-dim sparse text → **linear models** work best |
| 12 | **D** | More n-grams → **feature explosion** |

---

<!-- _class: lead -->

# Well done!

### Review Chapter 7 for any you missed
