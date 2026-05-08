# Activation Functions

Convert raw model scores into probabilities for classification tasks.

## Overview

Machine learning models often produce raw numerical scores — called **logits** — that need to be converted into probabilities before they can be interpreted as predictions. Activation functions perform this conversion. Quiver provides two standard activation functions: `softMax` for multi-class problems and `sigmoid` for binary classification.

### Softmax

The softmax function converts a vector of raw scores into a probability distribution. Each output value falls between 0 and 1, and the outputs sum to 1.0. This makes softmax the standard final step in multi-class classification — and the core operation inside transformer attention mechanisms:

```swift
import Quiver

let logits = [2.0, 1.0, 0.1]
let probs = logits.softMax()
// [0.659, 0.242, 0.099] — sums to 1.0
```

> Experiment: **The Quiver Notebook** is the right place to see how softMax redistributes mass. Push one logit from 1.0 to 3.0 and watch the corresponding probability jump while the others shrink — the output still sums to 1.0, which is what makes softMax a probability distribution rather than a score. See <doc:Quiver-Notebook>.

Quiver uses the numerically stable variant, which subtracts the maximum value before exponentiation. This prevents overflow when working with large scores — for example, `[1000, 1001, 1002]` computes correctly instead of producing infinity:

```swift
import Quiver

// These large values would overflow naive exp(), but softMax handles them
let scores = [1000.0, 1001.0, 1002.0]
let probs = scores.softMax()
// [0.090, 0.245, 0.665] — still sums to 1.0
```

### Sigmoid

The sigmoid function squashes each value into the range (0, 1) independently. It is the standard activation function for binary classification — a single output representing the probability that a sample belongs to the positive class:

```swift
import Quiver

let logits = [-2.0, 0.0, 2.0, 5.0]
let probs = logits.sigmoid()
// [0.119, 0.5, 0.881, 0.993]
```

### Choosing between softmax and sigmoid

Sigmoid and softMax serve different purposes. Sigmoid operates element-wise — each output depends only on its own input — making it the right choice for binary classification and multi-label problems (where multiple labels can be true simultaneously). SoftMax produces a distribution where outputs sum to 1.0, making it the right choice for multi-class problems (where exactly one label is correct):

```swift
import Quiver

// Binary: "is this email spam?" — one score, sigmoid
let spamScore = [1.8].sigmoid()  // [0.858] → 85.8% probability of spam

// Multi-class: "which category?" — one score per class, softMax
let categoryScores = [2.0, 1.0, 0.1].softMax()  // [0.659, 0.242, 0.099]
```

A useful mathematical property: σ(x) + σ(−x) = 1.0. This symmetry means the sigmoid of a positive score and the sigmoid of its negation always sum to 1, which is why a single sigmoid output captures both P(positive) and P(negative) without needing two outputs.

### Where activation functions fit

Quiver's built-in models handle probability conversion internally, so calling `softMax` or `sigmoid` on their output is not necessary. These functions are most useful when working with external model output — many models return raw logits, and converting them with `softMax` or `sigmoid` produces human-readable probabilities that pair naturally with Quiver's visualization tools.

Activation functions also work well for custom scoring systems. Any time we have raw numerical scores — user ratings, feature importance weights, similarity rankings — these functions convert them into a normalized probability scale for comparison and visualization.

## Topics

### Functions
- ``Swift/Array/softMax()->[Double]``
- ``Swift/Array/sigmoid()->[Double]``
