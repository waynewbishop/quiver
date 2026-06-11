# Working with Distributions

Evaluating probability densities, cumulative probabilities, and quantiles for the normal, Student's t, chi-squared, Poisson, and binomial distributions.

## Overview

A probability distribution describes how likely each possible value of a random quantity is. Test scores cluster around an average and taper off in both directions. Sensor noise sits near zero, with most samples small and a few large. Heights of adults fall in a bell-shaped band. The same mathematical machinery describes all three: a probability density function, a cumulative distribution function, and a quantile function.

The `Distributions` namespace groups these functions by distribution name. Quiver ships the normal distribution at `Distributions.normal`, the Student's t-distribution at `Distributions.t`, the chi-squared distribution at `Distributions.chiSquared`, the Poisson distribution at `Distributions.poisson`, and the binomial distribution at `Distributions.binomial`. Each call passes the distribution parameters directly. There is no fitted-distribution object to construct, no shared state to mis-configure, and every call site is self-documenting.

Three functions answer three questions about a distribution. The `pdf` function gives the height of the curve at a point, indicating where probability is concentrated. The `cdf` function gives the probability of falling at or below a value, useful for "what fraction of the distribution sits below this?" The `quantile` function inverts `cdf` to find the cutoff for a given probability, useful for "what value marks the 97.5th percentile?" The three calls below demonstrate each function on a standard normal:

```swift
import Quiver

// pdf: height of the bell curve at the mean
Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: 1)          // ≈ 0.3989

// cdf: 97.5% of the distribution sits at or below 1.96
Distributions.normal.cdf(x: 1.96, mean: 0, standardDeviation: 1)       // ≈ 0.975

// quantile: 1.96 is the value that puts 97.5% of the distribution below it
Distributions.normal.quantile(p: 0.975, mean: 0, standardDeviation: 1) // ≈ 1.96
```

### The normal distribution

The normal distribution is the bell-shaped curve that shows up everywhere in nature and engineering. Test scores, sensor noise, measurement error, and adult heights all cluster around an average with a symmetric falloff on either side. The Central Limit Theorem guarantees that the sample mean of almost any underlying population is approximately normal once the sample is large enough, which is why the normal sits at the center of inferential statistics. Quiver exposes the normal at `Distributions.normal` with all four functions: `pdf` for the density, `logPDF` for the log-density, `cdf` for cumulative probability, and `quantile` for the inverse CDF.

The probability density function (PDF) gives the relative likelihood of observing a specific value. The density is positive everywhere on the real line, peaks at the mean, and integrates to `1.0` across its full support. For a standard normal with mean `0` and standard deviation `1`, the density at the mean is approximately `0.3989`, which equals `1 / √(2π)`.

```swift
import Quiver

// Peak density of a standard normal sits at the mean
let peak = Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: 1)  // ≈ 0.3989

// One standard deviation out, density drops by about 39%
let oneSigma = Distributions.normal.pdf(x: 1, mean: 0, standardDeviation: 1)  // ≈ 0.2420
```

The log-density returns the natural log of the same quantity. Working in log-space is the standard tactic for numerical work that combines many density values together. Products become sums, and densities far in the tail that would round to zero in linear space stay representable. The `GaussianNaiveBayes` model calls `Distributions.normal.logPDF` directly during prediction, the same implementation we expose publicly. Any classifier, kernel density estimator, or probabilistic model we write next can use the same well-tested function without reimplementing the math.

```swift
import Quiver

// Log-density 4σ from the mean, finite even though the linear density is tiny
let lp = Distributions.normal.logPDF(x: 4, mean: 0, standardDeviation: 1)  // ≈ -8.919
```

The cumulative distribution function (CDF) gives the probability that a normally distributed value falls at or below `x`. The curve rises monotonically from `0` at negative infinity to `1` at positive infinity, equals `0.5` at the mean, and at `x = 1.96` returns the canonical value used to construct 95% confidence intervals.

```swift
import Quiver

// Halfway up the distribution at the mean
Distributions.normal.cdf(x: 0, mean: 0, standardDeviation: 1)  // = 0.5

// 1.96 standard deviations above the mean, the 97.5th percentile
Distributions.normal.cdf(x: 1.96, mean: 0, standardDeviation: 1)  // ≈ 0.975
```

A worked example makes the value concrete. Consider a class where test scores are approximately normal with mean `75` and standard deviation `10`. What fraction of the class scored below `90`?

```swift
import Quiver

// Test scores: mean = 75, std = 10
let belowNinety = Distributions.normal.cdf(x: 90, mean: 75, standardDeviation: 10)  // ≈ 0.9332
```

About 93% of the class scored below `90`. The CDF turns "where does this value sit in the distribution" into a concrete probability.

The quantile function is the inverse of the CDF. The quantile answers the question "what value of `x` puts probability `p` below it?" For a standard normal, the quantile at `p = 0.975` is approximately `1.96`. This is the cutoff that puts 2.5% of the distribution in each tail and bounds a 95% confidence interval.

```swift
import Quiver

// 95% critical value, the canonical confidence-interval cutoff
Distributions.normal.quantile(p: 0.975, mean: 0, standardDeviation: 1)  // ≈ 1.96

// And the median sits at the mean for a symmetric distribution
Distributions.normal.quantile(p: 0.5, mean: 0, standardDeviation: 1)    // = 0
```

The same function works on non-standard normals by passing `mean` and `standardDeviation`. Continuing the test-score example, what cutoff score does 90% of the class fall below?

```swift
import Quiver

// 90th percentile of test scores: mean = 75, std = 10
let cutoff = Distributions.normal.quantile(p: 0.90, mean: 75, standardDeviation: 10)  // ≈ 87.82
```

Roughly 90% of the class scored below `87.8`. The quantile turns "what percentile do I want" into a concrete score on the original measurement scale.

> Experiment: **The Quiver Notebook** is the right place to sweep `mean` and `standardDeviation`. Re-evaluating `pdf` and `cdf` with shifted parameters is how the shape of the bell curve moves from a memorized formula to a parameterized object. See <doc:Quiver-Notebook>.

### The Student's t-distribution

The t-distribution is the small-sample sibling of the normal. The normal describes how a sample mean behaves when the sample is large enough for the Central Limit Theorem to apply, with the usual rule of thumb being `n` of `30` or more. Below that threshold the sample mean has a wider distribution than the normal predicts, because we are estimating the standard deviation from the same small sample we are using to estimate the mean. That extra uncertainty needs a wider reference curve to honor it. The t-distribution is that wider curve.

The curve looks like a standard normal that has been stretched in the tails: centered at zero and symmetric, but with a spread that depends on a parameter called the degrees of freedom. Small `df` values produce visibly heavier tails. As `df` grows, the curve narrows toward the standard normal. At `df` near `1000` the two are visually indistinguishable. The classic rule for a sample of size `n` is `df = n - 1` for a one-sample inference. The `-1` accounts for the one quantity (the sample mean) we already estimated from the data before estimating the standard deviation.

Quiver exposes the t-distribution at `Distributions.t` with `cdf(x:df:)` and `quantile(p:df:)`. There is no PDF — practical inference uses the CDF and the quantile, and `LinearRegression.summary` calls both internally to compute p-values and confidence intervals for fitted coefficients.

```swift
import Quiver

// CDF: probability that a t-distributed value with df = 10 falls at or below 1.96
Distributions.t.cdf(x: 1.96, df: 10)    // ≈ 0.9608

// Quantile: the value with cumulative probability 0.975 under the same curve
Distributions.t.quantile(p: 0.975, df: 10)  // ≈ 2.2281
```

The numbers are concrete. Consider an experiment with a sample of size `11`, so `df = 10`. A 95% confidence interval splits the remaining 5% into two tails of 2.5% each, so the critical cutoff sits at `p = 0.975`. The t-critical value at `df = 10` is approximately `2.2281`. The corresponding normal critical value at the same `p` is approximately `1.9600`. The t-critical is `0.27` units wider, and that gap is the price we pay for not knowing the population standard deviation and having to estimate it from a sample of only eleven values.

```swift
import Quiver

// Sample of size 11 → df = 10
let tCrit = Distributions.t.quantile(p: 0.975, df: 10)               // ≈ 2.2281
let zCrit = Distributions.normal.quantile(p: 0.975, mean: 0,
                                          standardDeviation: 1)      // ≈ 1.9600
```

As `df` grows, the gap closes. At `df = 30` the t-critical is approximately `2.0423`. At `df = 1000` it is approximately `1.9623`, two decimals away from the standard normal value of `1.9600` and effectively indistinguishable for practical inference. The convergence is the mathematical statement of the rule of thumb that the t-distribution is unnecessary for samples large enough for the Central Limit Theorem to dominate.

### The chi-squared distribution

The chi-squared distribution describes the sum of squared independent standard-normal values. Square `df` independent draws from `N(0, 1)`, add them up, and the resulting random quantity follows the chi-squared distribution with `df` degrees of freedom. The distribution sits entirely on `[0, ∞)`, since squares are never negative, and the mean equals `df`. A chi-squared with `df = 5` has a typical value near `5`, and one with `df = 30` has a typical value near `30`.

Two practical questions reach for chi-squared. The first is variance inference. When we draw a sample of size `n` from a normal population, the sample variance scaled by `(n - 1) / σ²` follows a chi-squared distribution with `df = n - 1`, which gives confidence intervals and hypothesis tests for the population variance. The second is goodness of fit. A chi-squared statistic compares observed counts to expected counts under a hypothesized distribution and asks whether the gap is larger than chance would predict.

Quiver exposes the chi-squared distribution at `Distributions.chiSquared` with `cdf(x:df:)`. The CDF is the only function the common use cases need; goodness-of-fit testing and variance inference both work by computing a test statistic and reading the upper-tail probability off the CDF.

```swift
import Quiver

// CDF: probability that a chi-squared(df = 5) value falls at or below 11.07
Distributions.chiSquared.cdf(x: 11.07, df: 5)   // ≈ 0.9500

// At the canonical 0.95 upper-tail cutoff for df = 10
Distributions.chiSquared.cdf(x: 18.307, df: 10) // ≈ 0.9500
```

The value `11.07` is the standard 0.95 critical value for `df = 5`. The value `18.31` is the standard 0.95 critical value for `df = 10`. Both are reproduced here as cumulative probabilities. A worked example makes the goodness-of-fit use case concrete. Suppose a six-sided die is rolled 60 times and the observed counts in faces 1 through 6 are `[12, 8, 11, 9, 13, 7]`. Under a fair die, every face should appear approximately `10` times. The chi-squared statistic sums `(observed - expected)² / expected` across the six categories.

```swift
import Quiver

let observed: [Double] = [12, 8, 11, 9, 13, 7]
let expected = 10.0

var chiSquaredStat = 0.0
for count in observed {
    let diff = count - expected
    chiSquaredStat += (diff * diff) / expected
}
// chiSquaredStat = 2.8

// Six categories minus one estimated parameter (the total) gives df = 5
let upperTailProbability = 1.0 - (Distributions.chiSquared.cdf(x: chiSquaredStat, df: 5) ?? 0)
// upperTailProbability ≈ 0.73
```

A chi-squared statistic of `2.8` produces an upper-tail probability around `0.73`. Under a fair die, a gap that large or larger would happen roughly seven times in ten, well above the conventional `0.05` cutoff. The data is consistent with a fair die. We have no evidence to reject the null.

### The Poisson distribution

The Poisson distribution describes the count of independent events that occur in a fixed window when the events arrive at an average rate of `λ` per window. The classic examples are calls into a help line per hour, edits to a Wikipedia article per day, and earthquakes above a magnitude threshold per year. The distribution lives on the non-negative integers, and its single parameter `λ` controls both the mean and the variance: both equal `λ`. That mean-equals-variance property is also the quickest field test for whether observed count data is plausibly Poisson at all.

A Poisson is discrete, so the density function is a probability mass function rather than a probability density: each value `k` carries an actual probability `P(K = k)`, not a density that integrates to a probability. Quiver names this `pmf` to keep the distinction visible in the API. Use `pmf(k:lambda:)` for a single mass, `cdf(k:lambda:)` for a cumulative tail probability, and `quantile(p:lambda:)` for the smallest count whose cumulative probability reaches a chosen level.

```swift
import Quiver

// PMF: probability of exactly 2 calls when 3.5 per window are expected
Distributions.poisson.pmf(k: 2, lambda: 3.5)        // ≈ 0.1850

// CDF: probability of 5 or fewer calls
Distributions.poisson.cdf(k: 5, lambda: 3.5)        // ≈ 0.8576

// Quantile: the smallest count whose cumulative probability reaches 0.95
Distributions.poisson.quantile(p: 0.95, lambda: 3.5) // 7
```

A worked example anchors the use case. A web service receives an average of `4.2` requests per second from a single client. We want a guardrail count `k` such that the probability of seeing more than `k` requests in any given second from a legitimate client stays below `0.01`. The 99th percentile of `Poisson(λ = 4.2)` is the value we are after.

```swift
import Quiver

let lambda = 4.2

// The smallest k such that P(K <= k) >= 0.99
if let cutoff = Distributions.poisson.quantile(p: 0.99, lambda: lambda) {
    // cutoff = 10
}

// Sanity check the tail probability at the cutoff
if let cdfAtCutoff = Distributions.poisson.cdf(k: 10, lambda: lambda) {
    // cdfAtCutoff ≈ 0.9959, so P(K > 10) ≈ 0.0041
}
```

A guardrail at `k = 10` keeps the legitimate-traffic false-positive rate well below one percent per second under the Poisson model. The `logPMF` companion is the function to reach for when multiplying many of these probabilities together; the products become sums, and tail values that would round to zero in linear space stay representable. The `GaussianNaiveBayes` model uses the same log-space pattern with the normal density.

### The binomial distribution

The binomial distribution describes the count of successes in a fixed number of independent yes-or-no trials, each succeeding with the same probability `p`. Ten coin flips with a fair coin gives a binomial with `n = 10` and `p = 0.5`. Twenty-five A/B test exposures with a `0.04` conversion rate gives a binomial with `n = 25` and `p = 0.04`. The distribution lives on `{0, 1, ..., n}`, the mean is `n·p`, and the variance is `n·p·(1-p)`.

Quiver exposes the binomial at `Distributions.binomial` with the same four functions as the Poisson: `pmf(k:n:p:)`, `logPMF(k:n:p:)`, `cdf(k:n:p:)`, and `quantile(p:n:probability:)`. The `quantile` parameter name `probability:` avoids colliding with the cumulative probability `p:`; everywhere else `p` is the per-trial success probability.

```swift
import Quiver

// PMF: exactly 3 successes in 10 trials at p = 0.4
Distributions.binomial.pmf(k: 3, n: 10, p: 0.4)   // ≈ 0.2150

// CDF: at most 3 successes in 10 trials
Distributions.binomial.cdf(k: 3, n: 10, p: 0.4)   // ≈ 0.3823

// Quantile: the smallest success count whose cumulative probability reaches 0.95
Distributions.binomial.quantile(p: 0.95, n: 10, probability: 0.4) // 7
```

Classifier evaluation is the most common reason to reach for the binomial. A model that achieved `82` correct predictions out of `100` test examples has an empirical accuracy of `0.82`, but the true accuracy could be anywhere in a confidence interval around that value. Treating each prediction as an independent Bernoulli trial, the count of correct predictions follows a binomial. The CDF then gives the probability of seeing at least the observed correct count under a hypothesized true accuracy, which is the building block for accuracy confidence intervals and for the comparison of two classifiers.

```swift
import Quiver

// 82 correct out of 100 under a hypothesized true accuracy of 0.80
let n = 100
let p = 0.80

if let cdfAt82 = Distributions.binomial.cdf(k: 82, n: n, p: p) {
    // cdfAt82 ≈ 0.7287 — so P(correct >= 83) ≈ 0.2713
}
```

The chance of seeing `82` or more correct predictions when the true accuracy is `0.80` is about `0.27`. That is well within the range where the observed `82` is consistent with a true rate of `0.80`; we have no reason from this one experiment to claim the model is actually better than `0.80`. The narrative is the same shape as the chi-squared goodness-of-fit example above: a test statistic, a reference distribution, an upper-tail probability, a verdict.

### Why the optional return

Every function in `Distributions` returns `Double?`. The optional makes out-of-domain input a `nil` rather than a runtime trap or a silently propagating `NaN`. The conditions are distribution-specific but consistent: a non-positive standard deviation for the normal, non-positive degrees of freedom for t and chi-squared, a probability outside `(0, 1)` for any `quantile` call, and any computation whose result is non-finite. This matches the pattern used by `mean`, `median`, and other Quiver statistics. Invalid input is handled at the call site with `if let` or `guard let`, not buried inside the result. See <doc:Numerical-Literacy> for the broader distinction between `nil` (no data) and `NaN` (math undefined) across Quiver.

```swift
import Quiver

// std must be positive, nil falls out cleanly
let bad = Distributions.normal.cdf(x: 1.0, mean: 0, standardDeviation: -1)  // nil

// p must lie strictly between 0 and 1
let edge = Distributions.normal.quantile(p: 1.0, mean: 0, standardDeviation: 1)  // nil
```

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
