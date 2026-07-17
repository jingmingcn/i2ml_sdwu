---
marp: true
theme: course
paginate: true
footer: 'Ch.7 · Working with Text Data'
---

<!-- _class: lead -->

# Chapter 7
## Working with Text Data

Bag-of-words, tf-idf,
and topic modeling

---

## Learning Objectives

- Represent text numerically with **bag-of-words**
- Improve it with **stopwords**, **tf-idf**, and **n-grams**
- Interpret model coefficients on text
- Discover topics with **LDA**

---

## Types of Data as Strings

- **Categorical** — fixed set (e.g. colors)
- **Free strings that map to categories** — messy but finite
- **Structured strings** — addresses, dates
- **Text** — full sentences/documents (our focus)

A dataset of documents is a **corpus**; each document is a data point.

---

## Example: Sentiment Analysis of Movie Reviews

- Task: classify reviews as **positive** or **negative**
- Classic **binary text classification** problem
- Need to turn variable-length text into fixed-length feature vectors

---

## Bag-of-Words Representation

Three steps:

1. **Tokenize** — split text into words
2. **Build vocabulary** — collect all words across the corpus
3. **Encode** — count how often each vocab word appears per document

Word **order is discarded** — only counts remain.

---

## Bag-of-Words in scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)   # sparse count matrix
len(vect.vocabulary_)                  # size of vocabulary
```

Output is a **sparse matrix** (mostly zeros) → memory-efficient.

---

## Stopwords

- Very common words (*the, and, is*) carry little signal
- Remove them to shrink the vocabulary and cut noise

```python
CountVectorizer(stop_words="english")   # built-in list
# or set min_df to drop rare words
CountVectorizer(min_df=5)
```

---

## Rescaling with tf-idf

**Term frequency–inverse document frequency** weights words by how
informative they are: frequent in a doc but rare across the corpus → high.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
TfidfVectorizer(min_df=5).fit(text_train)
```

Downweights words common everywhere; highlights distinctive terms.

---

## Investigating Model Coefficients

- A linear model over BoW has one coefficient per word
- Largest **positive/negative** coefficients = most influential words
- Great for **interpretability** — see what drives predictions

```python
# inspect coef_ against vect.get_feature_names()
```

---

## Bag-of-Words with n-Grams

- Single words (unigrams) lose context: *"not good"* → *not*, *good*
- **n-grams** = sequences of n adjacent tokens capture some order

```python
CountVectorizer(ngram_range=(1, 2))   # unigrams + bigrams
```

Bigrams/trigrams add context but **explode** the feature count.

---

## Advanced Tokenization: Stemming & Lemmatization

- Normalize word forms: *replace, replaced, replacing* → one token
- **Stemming** — chop to a root (fast, crude)
- **Lemmatization** — dictionary base form (accurate, needs NLP library, e.g. spaCy)

Reduces vocabulary size and merges related words.

---

## Topic Modeling with LDA

- **Latent Dirichlet Allocation** — unsupervised discovery of **topics**
- Each document = a mixture of topics; each topic = a distribution of words
- Useful for organizing and exploring large corpora

```python
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10, learning_method="batch")
lda.fit(X)
```

---

## Chapter Summary

- **Bag-of-words** turns text into count vectors (order discarded)
- Trim with **stopwords** / `min_df`; reweight with **tf-idf**
- **n-grams** recover some word order at a size cost
- **Stemming/lemmatization** normalize word forms
- **LDA** discovers latent topics unsupervised

**Next:** Chapter 8 — Wrapping Up
