# Identifying a Distribution

Working from raw data to a named distribution family using histograms, summary statistics, and comparison to known candidates.

## Overview

A dataset arrives. Maybe it is the number of support tickets opened per hour, the heights of users in a fitness app, or the count of correct answers on a quiz. Before we can model the data, we want to answer a question: what kind of distribution is this? Knowing the answer unlocks the right downstream tools. A normal distribution lets us reason about averages with the central limit theorem, a Poisson distribution gives us a single parameter to estimate, and a binomial distribution turns counts into proportions with a known variance.

Identifying a distribution is a workflow, not a single function call. We start with a histogram to see the shape, summarize that shape with quartiles and a measure of asymmetry, then compare what we see to the candidate families we know. Quiver provides the tools for each step; this primer walks the workflow end-to-end. The three families it covers are the ones every practitioner reaches for first: the **normal** for continuous values that cluster around a center, the **Poisson** for counts of independent events in a fixed window, and the **binomial** for counts of successes in a fixed number of yes-or-no trials. Together they cover most of the data a Quiver caller will encounter, and the workflow that distinguishes among them scales to the broader catalog.

### Looking at the shape

The first step is always to see the data. A histogram divides the values into bins and counts how many fall into each. The resulting bars trace out the empirical distribution, and a glance at the shape suggests where to start. A symmetric bell points at the normal family. A right-leaning shape with a wall at zero points at the Poisson family. A discrete cluster of bars with a clear ceiling points at the binomial family.

The bin count matters. Too few bins smooth away real features. Too many bins surface random fluctuations as structure. Quiver exposes three rules for choosing a bin count from the data itself, freeing the caller from guessing. The next section, "Choosing a bin rule," covers the trade-off behind each choice; for now `.freedmanDiaconis` is a safe default for unfamiliar data.

```swift
import Quiver

let times = [3.2, 2.8, 3.5, 3.1, 2.9, 3.3, 3.4, 3.0, 3.2, 3.1,
             3.6, 2.7, 3.3, 3.1, 3.2, 3.4, 2.9, 3.0, 3.3, 3.2]

let bars = times.histogram(rule: .freedmanDiaconis)
// 5 bars, the middle three carry most of the mass, suggesting a roughly symmetric shape
```

Running all three rules on the same sample makes the trade-off concrete. Each rule returns a histogram; reading the bin count is the quick way to compare them without rendering the bars.

```swift
times.histogram(rule: .squareRoot).count        // 5
times.histogram(rule: .sturges).count           // 6
times.histogram(rule: .freedmanDiaconis).count  // 5
```

For a small sample like this one, all three rules produce a similar picture. The differences become obvious on heavier-tailed or larger datasets, where the choice of rule begins to influence the visual story the histogram tells. The bin count itself is sometimes worth printing alongside the bars, because a histogram that looks bell-shaped at five bins can look spiky at twenty. For discrete data where every value already carries a count, <doc:Frequency-Tables> is the right starting point instead of a histogram.

### Choosing a bin rule

Each of the three rules answers a different question about the data, and the choice between them is a choice about which assumption to make.

The **square-root rule** is the simplest. The bin count depends only on the sample size: `k = ⌈√n⌉`. A dataset of 100 values gets 10 bins. A dataset of 10,000 values gets 100 bins. The rule says nothing about the shape of the data, which makes it fast to compute and predictable in its behavior. Use it when the goal is a quick first look and the dataset is small enough that any sensible bin count will reveal the rough shape.

The **Sturges rule** is the classical default in introductory statistics. Herbert Sturges derived it in 1926 by treating each bin as a Bernoulli trial under the assumption that the data follows a normal distribution. The formula `k = ⌈log₂(n) + 1⌉` grows slowly: 8 bins at `n = 100`, 11 bins at `n = 1,000`, only 15 bins at `n = 10,000`. The rule works well for the small, roughly symmetric samples it was designed for and remains a common default in introductory statistical software.

The **Freedman-Diaconis rule** is the modern robust choice. Published by David Freedman and Persi Diaconis in 1981, the rule sets a bin width from the data rather than picking a bin count directly: `width = 2 · IQR / n^(1/3)`. The bin count then falls out from the data's range. The cube root in the denominator is the optimal scaling for the L2 risk of a histogram density estimator, and using the interquartile range instead of the standard deviation makes the bin width resistant to outliers; a few unusual values cannot blow up the bin width the way they can with standard-deviation-based rules. The rule is the default in most modern statistical libraries and the working data scientist's first choice for unfamiliar data.

When none of the three rules is right for a specific case, the explicit overload `histogram(bins: Int)` accepts a bin count directly. A teaching example with the round number `20`, a regulated reporting workflow that requires a specific bin count, or a chart that needs to align with a downstream visualization all have reasons to override the rule-driven choice. The two overloads coexist by parameter label and type, so the compiler picks the right one without ambiguity.

```swift
let exploration = scores.histogram(rule: .freedmanDiaconis)  // rule-driven
let reporting   = scores.histogram(bins: 20)                  // explicit
```

### Measuring the shape

A glance at the histogram suggests a shape, but a more honest reading uses a few summary statistics drawn from <doc:Statistics-Primer>. We want a measure of center, a measure of spread, and a measure of asymmetry. Quartiles give us all three at once: the median is the center, the interquartile range is the spread, and the gap between `q3 - median` and `median - q1` tells us whether the right tail is heavier than the left.

```swift
import Quiver

if let q = times.quartiles() {
    // q.median  ≈ 3.20      center
    // q.iqr     ≈ 0.30      spread of the middle 50%
    // q.q3 - q.median  ≈ 0.10
    // q.median - q.q1  ≈ 0.20
}
```

A symmetric distribution has matching upper and lower halves of the IQR. A right-skewed distribution has `q3 - median` larger than `median - q1`. The sample above shows the reverse, with the lower half twice as wide as the upper half, which is a mild left skew. The gap is small enough that a normal family is still a plausible candidate, and we turn to the candidate families to decide.

Comparing the two halves of the IQR by hand is a fair first read, and `skewnessReport()` turns it into a number we can trust. The report pairs two measures of asymmetry — a moment coefficient that weighs every value, and a robust quartile coefficient that listens only to the middle of the data — and tells us whether they agree. Skewness is the average cubed deviation from the mean, divided by the cubed standard deviation; cubing keeps the sign of each deviation, so a positive coefficient means a long tail toward high values and a negative one a long tail toward low values. The robust measure is the **Bowley coefficient**, named for the British statistician Arthur Lyon Bowley (1869–1957), who introduced it in the early twentieth century as a measure of asymmetry that holds up when the data carries extreme values. The coefficient is the same upper-minus-lower gap we just measured by hand, divided by the IQR to land it on a fixed scale.

```swift
import Quiver

// The team-salary sample from the Statistics primer, where the mean-median gap first flagged the skew.
let salaries = [50.0, 55.0, 58.0, 60.0, 62.0, 180.0]

if let report = salaries.skewnessReport() {
    print(report)
    // shape:       long tail toward high values   (right-skewed)
    // coefficient: 2.4109
    // cross-check: -0.1304   ⚠ the robust measure leans the other way — a few extreme values may be distorting this; check your outliers
}
```

The moment coefficient reads strongly right-skewed because the lone value at 180 dominates a sum of cubed distances, while the robust measure, reading only the quartiles, sees a nearly symmetric middle and leans slightly the other way. The two coefficients carry opposite signs, and the report says so plainly rather than collapsing them into a single verdict. The disagreement is the signal: the asymmetry lives in the tail, not the central body, and it points us back to the outlier before we trust either number alone. Where the two measures agree the skew is corroborated; where they part on strength or direction the right move is to investigate, not to read a cutoff.

### Naming candidate families

Three questions tell the big three apart. **Is the data continuous or integer-valued?** Continuous values point at the normal family; integer counts point at Poisson or binomial. **Is the count bounded?** A count with a fixed upper limit, like correct answers on a twenty-question quiz, points at the binomial family. A count with no natural ceiling, like support tickets per hour, points at the Poisson family. **Does the mean roughly equal the variance?** Mean-equals-variance is the defining diagnostic of the Poisson distribution; the normal and binomial families both decouple mean and variance.

Answering these three questions in order narrows the candidate to one family per branch.

```swift
import Quiver

// A continuous sample of response times
if let mean = times.mean(),
   let variance = times.variance() {
    // mean     ≈ 3.16
    // variance ≈ 0.054
    // Mean and variance are far apart, and the values are continuous;
    // the normal family is the working candidate.
}

// A small count of events per minute
let arrivals = [3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0, 4.0, 3.0, 4.0, 5.0]
if let mean = arrivals.mean(),
   let variance = arrivals.variance() {
    // mean     ≈ 3.87
    // variance ≈ 1.41
    // Mean and variance are similar in magnitude and the data is integer-valued;
    // the Poisson family is the working candidate, with λ ≈ 3.87.
}
```

The Poisson example here is close to but not exactly mean-equals-variance, which is what we expect from real data. Perfect agreement would actually be suspicious; real counts always carry some sampling noise. The point of the diagnostic is to rule out families, not to prove one. If the mean and variance differ by an order of magnitude, the Poisson family is wrong and we look elsewhere.

### Comparing data to a candidate

Once a candidate family is named, the next step is comparing the data to a sample from that family. The cleanest comparison is between the empirical cumulative distribution function (the sorted data) and the theoretical CDF of the candidate. If they track each other closely, the family is a plausible match. If they diverge in a systematic way, the family is wrong.

Quiver's `cdf` functions compute the theoretical curve for any candidate. For the response-time sample above, the normal candidate with `mean = 3.16` and `standardDeviation = 0.25` predicts the CDF at any point:

```swift
import Quiver

let candidateMean = 3.16
let candidateStd = 0.25

// What fraction of the candidate distribution sits at or below 3.0?
let theoreticalAt3 = Distributions.normal.cdf(x: 3.0, mean: candidateMean, standardDeviation: candidateStd)
// ≈ 0.2611

// What fraction of the sample sits at or below 3.0?
let empiricalAt3 = times.percentileRank(of: 3.0) / 100.0
// ≈ 0.30
```

A theoretical value of `0.2611` and an empirical value of `0.30` are within sampling noise of each other for a sample of twenty values. Across a grid of points the gap stays small, which supports the normal candidate. For the Poisson candidate on the arrivals data, the same comparison uses `Distributions.poisson.cdf` and gives the per-count tail probabilities. The visual version of this comparison is the quantile-quantile plot, which Quiver renders downstream once `Swift Charts` is wired in.

### Checking the standard-deviation bands

The candidate named above is normal, and a normal distribution makes a specific promise about where its mass sits: roughly 68 percent within one standard deviation of the mean, 95 percent within two, and 99.7 percent within three. The `empiricalRule()` method reads how closely the data keeps that promise, band by band. The method does not test for normality; it measures the gap between observed and expected at each band and hands us the numbers to read.

We start by printing the whole result. The `EmpiricalRule` value lays the three bands side by side, each with its observed fraction, the fraction a normal distribution predicts, and the signed difference between them. Here a few values cluster near eleven, with one outlier at thirty:

```swift
import Quiver

let samples = [10.0, 11, 12, 11, 10, 9, 11, 10, 12, 30]

if let check = samples.empiricalRule() {
    print(check)
    // Empirical rule check (n = 10)
    //               actual    expected    diff
    //   within 1σ:  0.900     0.683       +0.217
    //   within 2σ:  0.900     0.955       -0.054
    //   within 3σ:  1.000     0.997       +0.003
}
```

The observed fractions sit beside what a normal distribution predicts, and the signed difference makes each gap readable at a glance. Checking all three bands rather than one is the point: a single band can look fine while another reveals a departure.

To act on the result in code, we read the fields directly. The observed fractions are `within1Sigma`, `within2Sigma`, and `within3Sigma`; the matching expected fractions are `expected1Sigma`, `expected2Sigma`, and `expected3Sigma`:

```swift
import Quiver

if let check = samples.empiricalRule() {
    let gap1 = check.within1Sigma - check.expected1Sigma   // +0.217
    let gap2 = check.within2Sigma - check.expected2Sigma   // -0.054
    let gap3 = check.within3Sigma - check.expected3Sigma   // +0.003
    // The gap is three numbers, not one.
}
```

The gap is a vector, not a scalar. What we do next follows from *where* the largest gap sits, not from whether any gap crosses a cutoff — there is no threshold at which a fraction "becomes" non-normal. The action is to investigate, not to judge: go back to the histogram, render a quantile-quantile plot, or run the goodness-of-fit test the next section sets up.

The bands are nested — every value inside one standard deviation is also inside two and three — so the three rows are three readings of one sample, not three independent checks. The 3σ band reaching `1.000` is nearly automatic for well-behaved data and carries little information; the 1σ and 2σ bands do the real work. A set of bands close to expected means the data is *consistent with* normal in those bands, never that it *is* normal: many non-normal shapes place roughly normal-looking mass in the central bands and depart only in the tails. The bands are drawn at the sample standard deviation (`ddof: 1`).

### Testing the fit

Visual comparison is enough for most exploratory work. When we need a number that says whether the fit is close enough, we reach for a goodness-of-fit test. The chi-squared test compares observed counts to expected counts under the candidate distribution and turns the gap into an upper-tail probability. A small p-value means the candidate is unlikely; a large p-value means the candidate is consistent with the data.

The full goodness-of-fit machinery ships with the next release. The building blocks are already here. `Distributions.chiSquared.cdf(x:df:)` returns the cumulative probability of a chi-squared statistic, which is the operation a goodness-of-fit test reduces to. The arrivals data with a Poisson candidate of `λ = 3.87` would bucket the counts, compute expected frequencies from `Distributions.poisson.pmf`, sum the chi-squared statistic, and read the upper-tail probability off the CDF. The narrative is identical to the fair-die example in <doc:Working-With-Distributions>; only the candidate distribution changes.

### From identifying to modeling

A named distribution is not the goal; it is a tool. Once the data is identified as normal, Poisson, or binomial, the downstream modeling work changes shape. Normal data flows into linear regression, principal components, and the central limit theorem. Poisson data flows into rate-comparison tests and into the count-regression family of models. Binomial data flows into accuracy confidence intervals, A/B test comparisons, and into logistic-regression evaluation.

The chain that gets us from raw data to a named family also tells us when no named family fits well. A histogram with two clear peaks is bimodal, and no single-peak distribution will describe it honestly. Heavy tails that the candidate cannot reproduce point at a different family or at a mixture. Recognizing that the named families do not fit is its own diagnostic, and it is often the moment when domain knowledge gets pulled into the modeling decision. Throughout the workflow, the candidate functions return `Double?` rather than trapping on out-of-domain input; see <doc:Numerical-Literacy> for the `nil` versus `NaN` contract that governs every `Distributions` call.

> Experiment: **The Quiver Notebook** is the right place to walk this chain on data we shape ourselves. Start from a short array, then run the five steps: `histogram(rule: .freedmanDiaconis)` for the shape, `skewnessReport()` for the asymmetry, `mean()` and `variance()` for the candidate diagnostic, `Distributions.normal.cdf(x:mean:standardDeviation:)` for the comparison, and `empiricalRule()` for the standard-deviation bands. Add an outlier, spread the values wider, or introduce a second cluster, and watch the bands surface a defect a single glance would miss. Watching the chain react to a change we made by hand is the fastest way to internalize what identifying a distribution actually means. See <doc:Quiver-Notebook>.

## Topics

### Visual inspection
- ``Swift/Array/histogram(bins:)``
- ``Swift/Array/histogram(rule:)``
- ``BinRule``

### Shape characterization
- ``Swift/Array/mean()``
- ``Swift/Array/variance(ddof:)``
- ``Swift/Array/standardDeviation(ddof:)``
- ``Swift/Array/quartiles()``
- ``Swift/Array/skewnessReport()``
- ``SkewnessReport``

### Candidate distributions
- ``Distributions/normal``
- ``Distributions/poisson``
- ``Distributions/binomial``

### Comparison
- ``Distributions/normal/cdf(x:mean:standardDeviation:)``
- ``Distributions/poisson/cdf(k:lambda:)``
- ``Distributions/binomial/cdf(k:n:p:)``
- ``Distributions/chiSquared/cdf(x:df:)``
- ``Swift/Array/empiricalRule()``
- ``EmpiricalRule``
