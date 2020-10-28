# Imbalanced data stream classification with the changing prior probabilities

Master thesis implementation 

## Setup - experiment 3a

Comparison between HDWE method and selected state-of-the-art method with base classifiers SVC. 

### Data

In general, numbers of data streams: 84

1. Generated from stream-learn 

   * Number of class: 2
   * Number of concept drifts: 5
   * Types of concept drifts: sudden, incremental
   * Stationary imbalance ratio: 1%, 3%, 5%, 10%, 15%, 20%, 25%
   * Dynamically imbalance ratio: 1%, 3%, 5%, 10%, 15%, 20%, 25%
   * Number of samples: 10000 (where number of chunks: 200 and chunk size: 500)
   * Number of features: 20 (where informative: 15 and redundant: 5)
   * Random state: 1111, 1234, 1567

### Methods

1. HDWE
2. SEA
3. AWE
4. Learn++.CDS
5. Learn++.NIE
6. OUSE
7. REA

## Setup - experiment 3b

Comparison between HDWE method and selected state-of-the-art method with base classifiers HDDT. 

### Data

In general, numbers of data streams: 84

1. Generated from stream-learn 

   * Number of class: 2
   * Number of concept drifts: 5
   * Types of concept drifts: sudden, incremental
   * Stationary imbalance ratio: 1%, 3%, 5%, 10%, 15%, 20%, 25%
   * Dynamically imbalance ratio: 1%, 3%, 5%, 10%, 15%, 20%, 25%
   * Number of samples: 10000 (where number of chunks: 200 and chunk size: 500)
   * Number of features: 20 (where informative: 15 and redundant: 5)
   * Random state: 1111, 1234, 1567

### Methods

1. HDWE
2. SEA
3. AWE
4. Learn++.CDS
5. Learn++.NIE
6. OUSE
7. REA

## Setup - experiment 2

Different base classifier in the HDWE (Hellinger Distance Weighted Ensemble) method. Check influence of the type of base classifier on the ensemble.

### Data

In general, numbers of data streams: 84

1. Generated from stream-learn 

   * Number of class: 2
   * Number of concept drifts: 5
   * Types of concept drifts: sudden, incremental
   * Stationary imbalance ratio: 1%, 3%, 5%, 10%, 15%, 20%, 25%
   * Dynamically imbalance ratio: 1%, 3%, 5%, 10%, 15%, 20%, 25%
   * Number of samples: 10000 (where number of chunks: 200 and chunk size: 500)
   * Number of features: 20 (where informative: 15 and redundant: 5)
   * Random state: 1111, 1234, 1567

### Methods

1. HDWE(GNB)
2. HDWE(MLP)
3. HDWE(CART)
4. HDWE(HDDT)
5. HDWE(KNN)
6. HDWE(SVC)

### Evaluation

1. Test Then Train

### Metrics

1. Specificity
2. Recall
3. Precision
4. F1-score
5. Balanced accuracy score 
6. Geometric-mean

 

## Setup - experiment 1

Compare 2 methods HDWE and AWE with different imbalance ratio.

### Data

1. Generated from stream-learn 

   * Number of class: 2
   * Number of streams: 10 (random state has changed in range 1000 - 1550)
   * Number of concept drifts: 5
   * Types of concept drifts: sudden, incremental
   * Stationary imbalance ratio: 10%, 20%, 30%, 40%, 50%
   * Dynamically imbalance ratio: 10%, 20%, 30%, 40%
   * Number of samples: 10000 (where number of chunks: 200 and chunk size: 500)
   * Number of features: 20 (where informative: 15 and redundant: 5)

### Methods

1. HDWE (Hellinger Distance Weighted Ensemble) - own contribution
2. HDDT (Hellinger Distance Decision Tree) - own implementation 
2. AWE (Accuracy-Weighted Ensemble)
3. Gaussian Naive Bayes - base classifier

### Evaluation

1. Test Then Train

### Metrics

1. Balanced accuracy score
2. F1 score
3. Recall
4. Specificity

