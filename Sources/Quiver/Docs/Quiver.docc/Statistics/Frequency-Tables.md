# Frequency Tables

Counting how often values appear and turning counts into probabilities.

## Overview

Frequency is the bridge between raw observations and probability. Before any model fits, we usually want to know how often each value shows up in the data. A **frequency table** answers this question by listing every distinct value alongside the number of times it appears, or (once we divide by the total) alongside the fraction of the dataset it represents. This is the most elementary form of statistical reduction, and it is the building block underneath class priors, histograms, and any analysis that treats categories as data.

A short list of quiz outcomes makes the idea concrete. Suppose six students took a three-tier rubric and we recorded each result as `1`, `2`, or `3`:

```swift
let outcomes = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]
```

The dataset is small enough to read by eye, but the same operations scale to thousands of class [labels](<doc:Machine-Learning-Primer>) in a training set. We will return to this array throughout the article.

### Counting unique values

The first question to ask of any categorical dataset is which values appear at all, and after that, how many times each one shows up. Quiver answers both with a pair of methods. The `distinct()` method returns the unique values in ascending order, and `distinctCounts()` returns those same values paired with their integer counts.

The ascending-order guarantee matters more than it sounds. Calling `distinct()` twice on the same input always produces the same array, which makes tests, snapshots, and tutorial output reproducible from one run to the next. The companion method, `distinctCounts()`, returns labeled tuples, `(value:, count:)`, so call sites read clearly without remembering tuple positions.

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

The answer is **order**. A `Set` knows what is unique, but it does not know how to present those values consistently. The same input can come back in a different arrangement on a different run or a different platform. That is fine if all we need is membership, but a problem the moment we want to read the result, compare it to a previous run, or print it in a tutorial. A frequency table is something we look at, not just something we compute, and a table that reshuffles itself between runs is not really a table.

`distinct()` and `distinctCounts()` add the missing piece: a guaranteed ascending order. Once values are unique *and* ordered, the output becomes reproducible, easy to scan, and stable enough to test against. That is the small but meaningful step beyond what `Set` alone provides.

### The most common value

Once we have counts, the next question often asked of a frequency table is which value appears the most often. This is the **mode**, and it is the natural center for categorical data the same way the mean is the natural center for numeric data. Quiver exposes it as `mode()` on `Array where Element: Hashable`, returning an array of all values tied for highest frequency:

```swift
import Quiver

let outcomes = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]
outcomes.mode()              // [1.0] — the most common outcome

let bimodal = [4, 5, 4, 3, 5, 4, 5]
bimodal.mode()               // [4, 5] — two values tied for first

let strings = ["yes", "no", "yes", "maybe"]
strings.mode()               // ["yes"]
```

Returning an array rather than a single value is deliberate. A bimodal distribution (two categories tied for the highest frequency) is a real fact about the data, and Quiver surfaces it rather than picking one tie-breaker silently. When every value appears the same number of times, every value is a mode and the full array comes back unchanged. The mode pairs naturally with `distinctCounts()`: counts answer "how often does each value appear," and mode answers "which appears the most."

### From counts to probabilities

Counts answer the question "how many?" Probabilities answer "what fraction?" The conversion is one division: divide each count by the total number of observations. The result is a number between `0` and `1` that represents the empirical probability of drawing that value if we sampled uniformly at random from the array.

The `frequencyDistribution()` method performs the conversion in one call. The method returns a dictionary mapping each unique value to its relative frequency, and the resulting frequencies sum to `1.0` within floating-point tolerance: a valid empirical probability distribution by construction. When we only need a single value's frequency rather than the whole table, `probability(of:)` does the same arithmetic for one value at a time.

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

> Experiment: **The Quiver Notebook** is the right place to see priors form from data. Swap the outcomes array for one with different class proportions and re-run. The probabilities recompute and the prior distribution shifts. The same mechanism feeds class priors in Naive Bayes. See <doc:Quiver-Notebook>.

The same number wears two names depending on what we are claiming. As a description of the data we already have, `0.5` is the **frequency** of the value `1.0`: three out of six observations matched. As a forecast about what we would draw next from this same distribution, the same `0.5` is a **probability**. Frequency is the empirical fact; probability is the inference we draw from it. Both methods compute counts as `count / total` with no small-sample correction, because we are summarizing the observations in hand rather than estimating a population parameter.

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

The dictionary above is the complete prior table for this training set. When we later train a classifier, the same number appears as `prior:` in the per-class statistics a fitted Gaussian Naive Bayes model reports. Computing it directly with `frequencyDistribution()` lets us inspect class balance before fitting: a `0.166...` prior on virginica means it is the smaller class, which is information worth knowing before deciding whether to stratify a split or rebalance the data. See <doc:Naive-Bayes> for how the same priors flow into prediction.

### Are the counts what we expected

A frequency table tells us what the data *did*. A natural follow-up question asks whether what the data did is consistent with what we expected: whether a six-sided die is fair, whether website visitors are uniformly distributed across four landing pages, whether a survey's response rates match the population they sampled. The **chi-squared goodness-of-fit test** answers this. The test compares observed category counts to expected counts under a null hypothesis, summarizes the disagreement with a single number, and reads a p-value off the chi-squared distribution.

The test statistic is the sum of `(observed − expected)² / expected` across every category. Larger values mean larger disagreement. Under the null hypothesis that the observed counts came from the expected distribution, the statistic follows the chi-squared distribution with `k − 1` degrees of freedom, where `k` is the number of categories. Quiver provides the reference distribution as `Distributions.chiSquared.cdf`:

```swift
import Quiver
import Foundation

// 60 rolls of a six-sided die, counts for faces 1 through 6
let observed = [12.0, 8.0, 11.0, 9.0, 13.0, 7.0]

// Under a fair die, every face should appear 60 / 6 = 10 times
let total = observed.sum()
let expected = total / Double(observed.count)  // 10.0

// Chi-squared statistic: sum of (O − E)² / E across categories
var chiSquared = 0.0
for o in observed {
    chiSquared += pow(o - expected, 2) / expected
}
// chiSquared ≈ 2.8

let df = Double(observed.count - 1)  // 5 categories → df = 5
if let cdf = Distributions.chiSquared.cdf(x: chiSquared, df: df) {
    let pValue = 1 - cdf  // ≈ 0.731
    print("chi-squared: \(chiSquared), p: \(pValue)")
}
```

The statistic of `2.8` and the p-value of `0.731` together say the observed counts are well within the range a fair die would produce by chance. We do not reject the null hypothesis of fairness. The variation across categories (twelve ones, seven sixes, thirteen fives) looks unusual to the eye but is not statistically unusual at all.

Compare that to a clearly loaded die where six comes up 35 times in 60 rolls:

```swift
let loaded = [5.0, 5.0, 5.0, 5.0, 5.0, 35.0]
var chiSqLoaded = 0.0
for o in loaded {
    chiSqLoaded += pow(o - 10.0, 2) / 10.0
}
// chiSqLoaded ≈ 75.0
// p-value ≈ 10⁻¹⁵ — overwhelming evidence against fairness
```

A statistic of `75` with df=5 produces a p-value far below any reasonable significance level. The data is incompatible with a fair die, and the chi-squared test makes the gap between "looks weird" and "is statistically weird" quantitative.

> Note: The chi-squared approximation requires that every expected count be reasonably large: a common rule of thumb is at least five observations expected per category. For small samples or rare categories, the approximation breaks down and the test loses calibration. For binary outcomes specifically, a binomial-based test is the more honest choice.

## Topics

### Counting unique values
- ``Swift/Array/distinct()``
- ``Swift/Array/distinctCounts()``
- ``Swift/Array/mode()``

### From counts to probabilities
- ``Swift/Array/probability(of:)``
- ``Swift/Array/frequencyDistribution()``

### Inferential follow-up
- ``Distributions/chiSquared/cdf(x:df:)``

### Related articles
- <doc:Statistics-Primer>
- <doc:Naive-Bayes>
- <doc:Working-With-Distributions>
