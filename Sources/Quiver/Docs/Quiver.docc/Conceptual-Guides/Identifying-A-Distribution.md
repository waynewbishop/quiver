# Identifying a Distribution

Working from raw data to a named distribution family using histograms, summary statistics, and comparison to known candidates.

## Overview

Data collected from real-world sources—whether support tickets, user heights, or test scores—rarely arrives with a label. Identifying the underlying distribution family simplifies our analytical path, unlocking the correct statistical tools for modeling, testing, or reporting. We focus on three families: the **normal** distribution for continuous values clustering around a center; the **Poisson** distribution for independent events in a fixed window; and the **binomial** distribution for success counts in a fixed number of trials.

Identifying a distribution is a workflow. We build a histogram to visualize the shape, summarize it using quartiles and asymmetry measures, and compare the result to known candidates. Quiver provides the tools for each step.

### Looking at the shape

A histogram is our first diagnostic. It bins values and counts their frequency, tracing the empirical distribution. A symmetric bell often points to the normal family; a right-skewed shape with a wall at zero suggests Poisson; and a discrete cluster with a clear upper bound hints at the binomial family.

The bin count influences the story the histogram tells. Quiver provides three rules for choosing a bin count directly from the data, replacing manual guessing:

```swift
import Quiver

let times = [3.2, 2.8, 3.5, 3.1, 2.9, 3.3, 3.4, 3.0, 3.2, 3.1,
             3.6, 2.7, 3.3, 3.1, 3.2, 3.4, 2.9, 3.0, 3.3, 3.2]

let bars = times.histogram(rule: .freedmanDiaconis)
// 5 bars, the middle three carry most of the mass, suggesting a roughly symmetric shape
```

> Tip: Running multiple binning rules helps assess the trade-off. For categorical data where every value already carries a count, <doc:Frequency-Tables> is the appropriate starting point instead of a histogram.

### Selecting a binning strategy

Choose a binning strategy based on the data's characteristics. The square-root rule (`.squareRoot`) provides a simple, predictable bin count (`⌈√n⌉`), making it suitable for quick exploration on small datasets. The Sturges rule (`.sturges`) serves as a classical default for roughly symmetric, normal-like data. The Freedman-Diaconis rule (`.freedmanDiaconis`) is the robust modern default; it uses the interquartile range (IQR) to determine bin width, offering resistance to outliers that would otherwise distort the histogram. For specific requirements, the explicit `histogram(bins: Int)` overload allows manual bin count override.

### Measuring the shape

A histogram provides visual intuition, while summary statistics provide quantitative verification. We derive measures of center (median), spread (interquartile range/IQR), and asymmetry (the gap between `q3 - median` and `median - q1`) directly from <doc:Statistics-Primer>.

```swift
import Quiver

if let q = times.quartiles() {
    // q.median  ≈ 3.20  (center)
    // q.iqr     ≈ 0.30  (spread of middle 50%)
}
```

If `q3 - median` exceeds `median - q1`, the distribution is right-skewed. `skewnessReport()` quantifies this: it computes both a moment-based skew coefficient (sensitive to extremes) and the robust Bowley coefficient (based on quartiles). When these coefficients align, we can confidently confirm the distribution's asymmetry.

```swift
if let report = salaries.skewnessReport() {
    print(report)
}
```

Disagreement between these coefficients signals that extreme values may be distorting the distribution's tail, warranting further investigation rather than reliance on a single metric.

### Naming candidate families

Three questions help distinguish the main families:

1.  **Continuous or integer?** Continuous values suggest the normal family; integer counts suggest Poisson or binomial.
2.  **Is the count bounded?** A count with a fixed upper limit (e.g., correct quiz answers) suggests binomial. A count with no natural ceiling (e.g., tickets per hour) suggests Poisson.
3.  **Does mean roughly equal variance?** Mean-equals-variance is the defining diagnostic for Poisson; normal and binomial families decouple mean and variance.

Answering these questions rules out candidates. If the mean and variance differ by an order of magnitude, the Poisson family is wrong.

```swift
import Quiver

// Continuous response times
if let mean = times.mean(), let variance = times.variance() {
    // Normal family is the candidate here.
}

// Integer event counts per minute
let arrivals = [3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0, 4.0, 3.0, 4.0, 5.0]
if let mean = arrivals.mean(), let variance = arrivals.variance() {
    // Poisson is the candidate (λ ≈ 3.87).
}
```

Perfect agreement is unlikely; real counts carry sampling noise. The diagnostic aims to rule out families, not confirm them absolutely.

### Comparing data to a candidate

The cleanest comparison plots the empirical cumulative distribution function (CDF) against the candidate’s theoretical CDF. Close tracking indicates a plausible match, while systematic divergence suggests the family is incorrect.

Quiver’s `cdf` functions compute the theoretical curve. For the response-time sample (normal, mean=3.16, std=0.25):

```swift
import Quiver

// What fraction of the candidate distribution sits at or below 3.0?
let theoreticalAt3 = Distributions.normal.cdf(x: 3.0, mean: 3.16, standardDeviation: 0.25)

// What fraction of the sample sits at or below 3.0?
let empiricalAt3 = times.percentileRank(of: 3.0) / 100.0
```

Small gaps support the candidate. Systematic divergence—visible in a quantile-quantile plot—points to a failure in fit.

### Checking standard-deviation bands

Normal distributions promise predictable mass: ~68% within one standard deviation (σ), ~95% within 2σ, and ~99.7% within 3σ. The `empiricalRule()` method measures how closely data adheres to this promise.

```swift
import Quiver

let samples = [10.0, 11, 12, 11, 10, 9, 11, 10, 12, 30]

if let check = samples.empiricalRule() {
    print("Within 1σ: \(check.within1Sigma) (expected \(check.expected1Sigma))")
}
```

The observed fractions sit beside the normal predictions, and the signed difference highlights deviations. Checking all three bands reveals departures; a set of bands close to expected indicates consistency with a normal distribution.

### Testing the fit

Visual comparison suffices for exploratory work. When statistical confirmation is required, perform a goodness-of-fit test. The chi-squared test compares observed counts against expected counts under the candidate distribution, converting the difference into an upper-tail probability (p-value). A p-value below 0.05 generally indicates the candidate is a poor fit.

`Distributions.chiSquared.cdf(x:df:)` computes the cumulative probability for a chi-squared statistic, providing the foundation for these tests.

### From identifying to modeling

Identifying a distribution enables downstream modeling. Normal data feeds linear regression; Poisson data fits count-regression models; binomial data informs accuracy metrics and logistic evaluation. Recognizing that no standard family fits is equally valuable, as it signals the need for non-parametric approaches or mixture modeling.

Throughout this workflow, distribution functions return `Double?` rather than trapping on invalid input—see <doc:Numerical-Literacy> for the `nil` vs `NaN` contract.

> Experiment: The **Quiver Notebook** is the right place to walk this chain. Start from a sample, then run the workflow: `histogram(rule: .freedmanDiaconis)` for shape, `skewnessReport()` for the asymmetry, `mean()` and `variance()` for candidate diagnostic, `Distributions.normal.cdf(...)` for comparison, and `empiricalRule()` for the bands. Add an outlier, spread the values wider, or introduce a second cluster, and watch how the diagnostics reveal the change. See <doc:Quiver-Notebook>.

## Topics

### Visual inspection
- ``Swift/Array/histogram(bins:)``
- ``Swift/Array/histogram(rule:)``
- ``BinRule``

### Shape characterization
- ``Swift/Array/quartiles()``
- ``Swift/Array/skewnessReport()``
- ``SkewnessReport``

### Comparison
- ``Distributions/normal/cdf(x:mean:standardDeviation:)``
- ``Distributions/poisson/cdf(k:lambda:)``
- ``Distributions/binomial/cdf(k:n:p:)``
- ``Swift/Array/empiricalRule()``
- ``EmpiricalRule``

