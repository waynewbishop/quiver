# Working with Distributions

Evaluating probability densities, cumulative probabilities, and quantiles for the normal, Student's t, chi-squared, Poisson, and binomial distributions.

## Overview

A **probability distribution** assigns a probability to each possible value of a random quantity. Whether the data represents test scores, sensor noise, or heights, we analyze these distributions using three functions: the probability density function (PDF), the cumulative distribution function (CDF), and the quantile function.

The `Distributions` namespace organizes these functions. Quiver provides the normal distribution, the Student's t-distribution, the chi-squared distribution, the Poisson distribution, and the binomial distribution. Every distribution provides `pdf`, `cdf`, and `quantile` methods, ensuring a consistent API.

```swift
import Quiver

// pdf: height of the bell curve at the mean
Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: 1)          // ≈ 0.3989

// cdf: 97.5% of the distribution sits at or below 1.96
Distributions.normal.cdf(x: 1.96, mean: 0, standardDeviation: 1)       // ≈ 0.975

// quantile: 1.96 is the value that puts 97.5% of the distribution below it
Distributions.normal.quantile(p: 0.975, mean: 0, standardDeviation: 1) // ≈ 1.96
```

### From counts to probability

Empirical probabilities are calculated from observed event frequencies. For a list of events, the probability of an outcome is the fraction of observations matching it.

```swift
import Quiver

// Past message labels: 1.0 = spam, 0.0 = legitimate.
let labels = [/* 420 observed messages */]

let pSpam = labels.probability(of: 1.0)         // 0.20
```

The `frequencyDistribution()` returns a dictionary mapping distinct outcomes to their probabilities, summing to 1.0. This empirical distribution is the building block for priors in Bayesian reasoning. See <doc:Statistics-Primer> for the descriptive vocabulary, and <doc:Frequency-Tables> for categorical counts.

### The normal distribution

The normal distribution models phenomena that cluster around an average with symmetric falloff, such as measurement error, test scores, and adult heights. The Central Limit Theorem guarantees that sample means of large enough populations are approximately normal, making this distribution central to inferential statistics.

Quiver provides `pdf` (density), `logPDF` (log-density), `cdf` (cumulative probability), and `quantile` (inverse CDF) for the normal distribution at `Distributions.normal`.

```swift
import Quiver

// Peak density at the mean
let peak = Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: 1)  // ≈ 0.3989

// Log-density remains representable far into the tail
let lp = Distributions.normal.logPDF(x: 4, mean: 0, standardDeviation: 1)  // ≈ -8.919
```

The CDF gives the probability that a value falls at or below `x`. For test scores with mean 75 and standard deviation 10, the fraction scoring below 90 is:

```swift
let belowNinety = Distributions.normal.cdf(x: 90, mean: 75, standardDeviation: 10)  // ≈ 0.9332
```

The quantile function is the inverse CDF, determining the cutoff value for a desired probability. The 90th percentile of those test scores is:

```swift
let cutoff = Distributions.normal.quantile(p: 0.90, mean: 75, standardDeviation: 10)  // ≈ 87.82
```

> Experiment: The **Quiver Notebook** is the right place to sweep `mean` and `standardDeviation` to visualize how these parameters alter the bell curve's shape. See <doc:Quiver-Notebook>.

### The Student's t-distribution

The t-distribution models the distribution of sample means when the population standard deviation is unknown and the sample size is small. It features heavier tails than the normal distribution to account for the added uncertainty of estimating parameters from small samples. As degrees of freedom (`df`) increase, the t-distribution converges to the standard normal.

Use `Distributions.t` for `cdf(x:df:)` and `quantile(p:df:)` to compute confidence intervals and perform hypothesis tests.

```swift
import Quiver

// Sample of size 11 → df = 10
let tCrit = Distributions.t.quantile(p: 0.975, df: 10)               // ≈ 2.2281
let zCrit = Distributions.normal.quantile(p: 0.975, mean: 0,
                                          standardDeviation: 1)      // ≈ 1.9600
```

### The chi-squared distribution

The chi-squared distribution models the sum of squared independent standard-normal values. It is primarily used for variance inference and goodness-of-fit testing. The mean equals the degrees of freedom (`df`).

Use `Distributions.chiSquared.cdf(x:df:)` to perform these tests. For a fair-die test (observed `[12, 8, 11, 9, 13, 7]`, expected `10`), the chi-squared statistic compares the observed frequency to the expected frequency across six categories.

```swift
import Quiver

let observed: [Double] = [12, 8, 11, 9, 13, 7]
let expected = 10.0

var chiSquaredStat = 0.0
for count in observed {
    let diff = count - expected
    chiSquaredStat += (diff * diff) / expected
}

// Six categories minus one estimated parameter (total) → df = 5
let upperTailProbability = 1.0 - (Distributions.chiSquared.cdf(x: chiSquaredStat, df: 5) ?? 0)
// upperTailProbability ≈ 0.73
```

### The Poisson distribution

The Poisson distribution models the count of independent events occurring in a fixed window, given an average rate `λ` per window (e.g., calls per hour, earthquakes per year). Mean and variance both equal `λ`. Quiver uses `pmf` (probability mass) for discrete distributions, along with `cdf` and `quantile`.

```swift
import Quiver

// probability of exactly 2 calls when 3.5 are expected
Distributions.poisson.pmf(k: 2, lambda: 3.5)        // ≈ 0.1850

// smallest count where cumulative probability reaches 0.95
Distributions.poisson.quantile(p: 0.95, lambda: 3.5) // 7
```

### The binomial distribution

The binomial distribution models the count of successes in a fixed number of independent binary trials (`n` trials, each with success probability `p`). Mean is `n·p`, variance is `n·p·(1-p)`.

```swift
import Quiver

// exactly 3 successes in 10 trials at p = 0.4
Distributions.binomial.pmf(k: 3, n: 10, p: 0.4)   // ≈ 0.2150

// smallest success count where cumulative probability reaches 0.95
Distributions.binomial.quantile(p: 0.95, n: 10, probability: 0.4) // 7
```

Classifier evaluation uses the binomial to estimate confidence intervals for accuracy, treating each correct prediction as a Bernoulli trial.

### Numerical safety

Every function in `Distributions` returns `Double?`. Invalid inputs—such as non-positive standard deviation, negative degrees of freedom, probabilities outside `(0, 1)`, or non-finite results—return `nil` rather than throwing or propagating `NaN`. This forces explicit handling at the call site. See <doc:Numerical-Literacy> for the handling of missing data (`nil`) versus undefined math (`NaN`).

## Topics

### Distributions
- ``Distributions/normal``
- ``Distributions/t``
- ``Distributions/chiSquared``
- ``Distributions/poisson``
- ``Distributions/binomial``

### Density and mass
- ``Distributions/normal/pdf(x:mean:standardDeviation:)``
- ``Distributions/normal/logPDF(x:mean:standardDeviation:)``
- ``Distributions/poisson/pmf(k:lambda:)``
- ``Distributions/poisson/logPMF(k:lambda:)``
- ``Distributions/binomial/pmf(k:n:p:)``
- ``Distributions/binomial/logPMF(k:n:p:)``

### Cumulative probability and quantiles
- ``Distributions/normal/cdf(x:mean:standardDeviation:)``
- ``Distributions/normal/quantile(p:mean:standardDeviation:)``
- ``Distributions/t/cdf(x:df:)``
- ``Distributions/t/quantile(p:df:)``
- ``Distributions/chiSquared/cdf(x:df:)``
- ``Distributions/poisson/cdf(k:lambda:)``
- ``Distributions/poisson/quantile(p:lambda:)``
- ``Distributions/binomial/cdf(k:n:p:)``
- ``Distributions/binomial/quantile(p:n:probability:)``

### Mean and variance
- ``Distributions/poisson/mean(lambda:)``
- ``Distributions/poisson/variance(lambda:)``
- ``Distributions/binomial/mean(n:p:)``
- ``Distributions/binomial/variance(n:p:)``
