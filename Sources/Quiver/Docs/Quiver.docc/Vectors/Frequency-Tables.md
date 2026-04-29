# Frequency Tables

Counting how often values appear and turning counts into probabilities.

## Overview

Frequency is the bridge between raw observations and probability. Before any model fits, before any inference test runs, we usually want to know a simpler thing: how often does each value show up in the data? A frequency table answers that question by listing every distinct value alongside the number of times it appears, or — once we divide by the total — alongside the fraction of the dataset it represents. This is the most elementary form of statistical reduction, and it is the building block underneath class priors, histograms, and any analysis that treats categories as data.

A short list of quiz outcomes makes the idea concrete. Suppose six students took a three-tier rubric and we recorded each result as `1`, `2`, or `3`:

```swift
let outcomes = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]
```

The dataset is small enough to read by eye, but the same operations scale to thousands of class [labels](<doc:Machine-Learning-Primer>) in a training set. We will return to this array throughout the article.

### Counting unique values

The first question to ask of any categorical dataset is which values appear at all, and after that, how many times each one shows up. Quiver answers both with a pair of methods. The `distinct()` method returns the unique values in ascending order, and `distinctCounts()` returns those same values paired with their integer counts.

The ascending-order guarantee matters more than it sounds. Calling `distinct()` twice on the same input always produces the same array, which makes tests, snapshots, and tutorial output reproducible from one run to the next. The companion method, `distinctCounts()`, returns labeled tuples — `(value:, count:)` — so call sites read clearly without remembering tuple positions.

```swift
import Quiver

let outcomes = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]

outcomes.distinct()
// [1.0, 2.0, 3.0]

outcomes.distinctCounts()
// [(value: 1.0, count: 3), (value: 2.0, count: 2), (value: 3.0, count: 1)]
```

Both methods work on any `Array` whose `Element` is `Hashable & Comparable`, so they apply equally well to `[Int]`, `[String]`, or any custom value type that conforms to both protocols.

### Going beyond standard collections

A reasonable first instinct is that the language already handles uniqueness. Wrapping a list in a `Set` collapses duplicates, and a small loop can count occurrences. So why introduce dedicated methods for something the standard library nearly gives us for free?

The answer is **order**. A `Set` knows what is unique, but it does not know how to present those values consistently. The same input can come back in a different arrangement on a different run or a different platform — fine if all we need is membership, but a problem the moment we want to read the result, compare it to a previous run, or print it in a tutorial. A frequency table is something we look at, not just something we compute, and a table that reshuffles itself between runs is not really a table.

`distinct()` and `distinctCounts()` add the missing piece: a guaranteed ascending order. Once values are unique *and* ordered, the output becomes reproducible, easy to scan, and stable enough to test against. That is the small but meaningful step beyond what `Set` alone provides.

### From counts to probabilities

Counts answer the question "how many?" Probabilities answer "what fraction?" The conversion is one division — divide each count by the total number of observations. The result is a number between `0` and `1` that represents the empirical probability of drawing that value if we sampled uniformly at random from the array.

The `frequencyDistribution()` method performs the conversion in one call. It returns a dictionary mapping each unique value to its relative frequency, and the resulting frequencies sum to `1.0` within floating-point tolerance — a valid empirical probability distribution by construction. When we only need a single value's frequency rather than the whole table, `probability(of:)` does the same arithmetic for one value at a time.

```swift
import Quiver

let outcomes = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]

outcomes.probability(of: 1.0)    // 0.5         (3 / 6)
outcomes.probability(of: 2.0)    // 0.333...    (2 / 6)
outcomes.probability(of: 3.0)    // 0.166...    (1 / 6)

outcomes.frequencyDistribution()
// [1.0: 0.5, 2.0: 0.333..., 3.0: 0.166...]
// 0.5 + 0.333... + 0.166... = 1.0
```

The same number wears two names depending on what we are claiming. As a description of the data we already have, `0.5` is the **frequency** of the value `1.0` — three out of six observations matched. As a forecast about what we would draw next from this same distribution, the same `0.5` is a **probability**. Frequency is the empirical fact; probability is the inference we draw from it. Both methods compute counts as `count / total` with no Bessel correction, because we are summarizing the observations in hand rather than estimating an underlying parameter.

### Building the priors a classifier needs

The same machinery that summarizes quiz scores produces the **class priors** a Naive Bayes classifier needs. A prior is the answer to "how common is each class in the training data?" — and that is exactly what `frequencyDistribution()` returns when the input is an array of class labels.

Suppose we are training a flower classifier with three species encoded as `0`, `1`, and `2`. The training set has six examples:

```swift
import Quiver

// Class labels from training data: 0 = setosa, 1 = versicolor, 2 = virginica
let labels = [0.0, 0.0, 1.0, 1.0, 1.0, 2.0]

labels.probability(of: 0.0)
// 0.333... — prior for setosa

labels.frequencyDistribution()
// [0.0: 0.333..., 1.0: 0.5, 2.0: 0.166...]
```

The dictionary above is the complete prior table for this training set. The same number appears as `prior:` in the per-class statistics a fitted Gaussian Naive Bayes model reports. Computing it directly with `frequencyDistribution()` lets us inspect class balance before fitting — a `0.166...` prior on virginica means it is the smaller class, which is information worth knowing before deciding whether to stratify a split or rebalance the data. See <doc:Naive-Bayes> for how the same priors flow into prediction.

## Topics

### Counting unique values
- ``Swift/Array/distinct()``
- ``Swift/Array/distinctCounts()``

### From counts to probabilities
- ``Swift/Array/probability(of:)``
- ``Swift/Array/frequencyDistribution()``

### Related articles
- <doc:Statistics-Primer>
- <doc:Naive-Bayes>
