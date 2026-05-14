# Inferential Statistics Primer

Reason from a sample to the population it came from, with hypothesis tests, confidence intervals, and resampling.

## Overview

Descriptive statistics summarize a dataset we already have. Inferential statistics treats that dataset as a sample of something larger and asks what the sample lets us say about the population behind it.

> Note: This article assumes familiarity with mean, standard deviation, and z-scores. See <doc:Statistics-Primer> for the descriptive vocabulary the methods here build on.

### Samples and populations

A **population** is the full set of values we care about while a **sample** is the subset we actually observe. Inferential statistics asks what the sample lets us say about the population, and how confident we can be in saying it.

The catch is that any single sample is one of many possible samples we could have drawn. A different week of users would have produced a slightly different mean session time. A different watch session would have produced a slightly different step rhythm. Inferential statistics is built around that variability.

A small sample makes the distinction concrete:

```swift
import Quiver

// 10 users from a one-week A/B test variant
let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

sessionSeconds.mean()         // 250.0 — the sample mean
sessionSeconds.standardDeviation()   // ~6.91 — the sample standard deviation
```

The sample mean of `250.0` is a fact about these ten users. Whether the population mean is also near `250.0`, or whether the gap from a known control baseline is real, is the question inferential statistics answers.

### The sampling distribution of the mean

Imagine drawing the same-sized sample from the population over and over again. Each draw produces a slightly different sample mean. The collection of all those possible sample means has its own distribution, the **sampling distribution of the mean**. It describes how much the sample mean wobbles from one draw to the next.

The remarkable fact about this distribution has a name. The **Central Limit Theorem** says that when we average many independent observations, the distribution of the sample mean approaches a normal (bell-shaped) distribution regardless of the population's shape, as long as the population has finite variance. Skewed populations, bimodal populations, populations with strange tails: once we average enough of them, the sample mean is approximately normal. Almost every method in this primer rests on this single fact. It means we do not need to know the shape of the population; we only need a sample large enough for the theorem to apply, and the math we use on the sample mean is allowed to assume the bell-shaped behavior of a normal distribution.

The theorem is easier to trust once we have seen it work. The next snippet builds a heavily skewed population from an exponential distribution, draws a thousand small samples from it, and records the mean of each sample. The population is obviously not bell-shaped — most values cluster near zero, with a long right tail — but the distribution of sample means is:

```swift
import Quiver

// Build a skewed population with rate = 0.5, so the population mean is 1/0.5 = 2.0.
// Exponential is asymmetric — most values are small, with a long right tail.
let population = [Double].randomExponential(10_000, rate: 0.5)

// Draw 1,000 samples of size 50 and record the mean of each.
let sampleMeans = population.samplingDistributionOfMean(
    sampleSize: 50,
    iterations: 1000,
    seed: 42
)

// The sampling distribution is approximately normal, centered on the population mean.
sampleMeans.mean()              // ≈ 2.05  — recovers the population mean
sampleMeans.standardDeviation() // ≈ 0.29  — the standard error of the mean
```

The empirical standard error of about `0.29` matches the theoretical prediction: the population standard deviation is `1 / 0.5 = 2.0`, and dividing by `√50` gives `≈ 0.283`. A skewed population produces a bell-shaped sampling distribution, and the math we use from this point on rests on that fact.

> Note: A common rule of thumb is that samples of size 30 or more are usually large enough for the Central Limit Theorem to give a good approximation. Smaller samples can still be analyzed, but the t-distribution is the right tool when we cannot rely on a large sample.

### The standard error

The sampling distribution has its own spread. That spread has a name, the **standard error** of the mean, and it is the quantity that tells us how precisely the sample mean estimates the population mean. A small standard error means the sample mean changes very little from one hypothetical sample to the next; a large standard error means the sample mean is unstable and a single value should not be trusted on its own.

The standard error equals the sample standard deviation divided by the square root of the sample size. Larger samples produce smaller standard errors, which is the mathematical statement of "more data, more confidence." Quiver ships this as `standardError()` on any `[Double]`:

```swift
import Quiver

let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

if let se = sessionSeconds.standardError() {
    print(se)  // ≈ 2.19
}
```

> Note: Standard error uses the sample standard deviation, which divides by `n - 1`. Quiver's `standardDeviation()` and `standardError()` default to this — `ddof: 1` is the sample formula every inferential calculation in this primer assumes. For population statistics on a complete dataset, pass `ddof: 0` explicitly.

### Small samples and the t-distribution

The Central Limit Theorem promises normality for the sample mean when `n ≥ 30`. Below that threshold the sampling distribution of the mean has heavier tails than the bell curve, because we are estimating both the population mean and the population standard deviation from the same small sample. The **t-distribution** corrects for that uncertainty by widening the reference distribution as the sample shrinks. Quiver gives us the building blocks to use it directly through `Distributions.t.quantile` and `Distributions.t.cdf`:

```swift
import Quiver

// 95% two-tailed critical value for a sample of size 10 — df = n − 1 = 9
let tCritical = Distributions.t.quantile(p: 0.975, df: 9)  // ≈ 2.262

// The normal counterpart for comparison
let zCritical = Distributions.normal.quantile(p: 0.975, mean: 0, standardDeviation: 1)  // ≈ 1.96
```

The t-critical of about `2.262` is noticeably wider than the normal z-critical of `1.96`. That gap is the penalty for not knowing the population standard deviation in advance. As `df` grows, the t-distribution converges to the normal — at `df = 30` the t-critical is already `2.04`, and by `df = 100` the two are barely distinguishable. A confidence interval built with the t-critical is wider than the same interval built with the normal critical, and the extra width is the honest accounting of small-sample uncertainty.

> Note: For the full t-distribution and chi-squared surface, see <doc:Working-With-Distributions>.

### Hypothesis testing

A **hypothesis test** is a structured way to decide whether the data we observed is consistent with a specific claim about the population. The procedure has a fixed shape. State a **null hypothesis**, the conservative claim, usually that nothing has changed or that two groups are the same. Pair it with an **alternative hypothesis**, the claim we would accept if the null is rejected. Pick a significance level, called **alpha** (typically `0.05`), which is the probability of mistakenly rejecting the null when it is actually true. Then compute a single number from the sample, called a **test statistic**, and compare it against what we would expect if the null were true.

For an A/B test on session times, the null hypothesis is "the variant's population mean equals the control baseline of 240 seconds." The alternative is "the variant's population mean differs from 240 seconds." Our sample mean is `250.0`. The question is whether a gap of 10 seconds is large enough, given the standard error, to be evidence that the populations actually differ, or whether a gap that size could plausibly arise by chance from random sampling alone.

With only ten observations, the small-sample t-distribution is the appropriate reference — the normal approximation rests on `n ≥ 30`. The t-statistic itself is computed the same way as a z-statistic — the difference is which distribution we compare it against:

```swift
import Quiver

let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

let baseline = 240.0
let alpha = 0.05

if let sampleMean = sessionSeconds.mean(),
   let se = sessionSeconds.standardError() {

    // Test statistic: how many standard errors does the sample mean
    // sit from the hypothesized population mean?
    let t = (sampleMean - baseline) / se    // ≈ 4.575

    // Two-tailed p-value from the t-distribution with df = n − 1 = 9
    let df = Double(sessionSeconds.count - 1)
    if let cdf = Distributions.t.cdf(x: abs(t), df: df) {
        let pValue = 2 * (1 - cdf)          // ≈ 0.0013
        let reject = pValue < alpha
        print("t: \(t), p: \(pValue), reject null: \(reject)")
    }
}
```

The t-statistic of about `4.575` means the sample mean of 250 sits more than four standard errors above the hypothesized population mean of 240. Under the null, a gap that large would happen roughly thirteen times in ten thousand. The p-value is far below alpha, so we reject the null — the data is not consistent with a population mean of 240.

> Note: The same t-statistic compared against the normal distribution would produce a p-value near `0.000005` — roughly 270× smaller than the honest small-sample value. The normal reference is overconfident at this sample size. The conclusion does not change at α = 0.05, but the normal p-value would overstate the strength of the evidence by orders of magnitude. For small samples, always pair the t-statistic with the t-distribution.

#### Type I and Type II errors

Hypothesis tests can produce two kinds of mistakes. A **Type I error** is rejecting the null when it is true: a false alarm. A **Type II error** is failing to reject the null when the alternative is true: a missed detection. Setting alpha to `0.05` caps the Type I error rate at five percent. The Type II error rate depends on sample size, true effect size, and how strict alpha is.

### Interpreting the p-value

The output of a hypothesis test is a number called the **p-value**. It is the probability of observing data at least as extreme as our sample if the null hypothesis were true. A small p-value means the data would be surprising under the null, which is taken as evidence against the null. By convention, when the p-value falls below alpha, we reject the null.

The p-value is also the most misinterpreted number in applied statistics. The misinterpretations show up constantly in product reviews, dashboards, and engineering postmortems, so the precise statement of what the p-value is matters.

> Important: A p-value of `0.03` does not mean "there is a 3% probability the null hypothesis is true." It does not mean "there is a 97% probability the alternative is true." It means: "if the null were true, we would see data this extreme or more extreme only 3% of the time." The probability statement is about the data given the hypothesis, not the hypothesis given the data. These are different conditional probabilities, and conflating them is the most common mistake in applied hypothesis testing.

The mistake is easy to make because the misinterpretation feels intuitive. It is wrong for the same reason that "the probability a person with a positive medical test has the disease" is not the same as "the probability of a positive test given the disease." Translating between the two requires knowing the prior probability of the disease in the population, and the p-value has no access to a prior. The p-value alone cannot tell us how likely the null is to be true. It can only tell us how surprising the data would be under the null.

A second common mistake is treating `p < 0.05` as an on-off switch: true or false, ship it or kill it. A p-value just below `0.05` and a p-value just above are barely distinguishable as evidence. The threshold is a convention for taking action, not a sharp break in the underlying truth. Effect size and practical significance, covered later in this primer, exist precisely to keep us from treating a borderline p-value as a yes-or-no verdict.

### Resampling

Inferential statistics has two broad strategies. One is **parametric**: assume a reference distribution like the normal or the t, plug in a closed-form formula, and read off a p-value or interval. The other is **resampling**: let the data itself describe the variability of the estimate, by drawing many resamples and watching how much the statistic moves from one resample to the next. Quiver exposes the resampling tool as `resampled(iterations:seed:statistic:)` on `[Double]`.

> Note: In the statistics literature this technique is known as the *bootstrap*.

The idea is direct. Given a sample of size `n`, draw a resample of the same size *with replacement*, meaning the same value can appear multiple times in the resample, and some original values will be missing. Compute the statistic of interest on that resample. Repeat thousands of times. The collection of resampled statistics approximates the sampling distribution we cannot observe directly.

```swift
import Quiver

// Variant's session times in seconds
let sample = [245.0, 252.0, 238.0, 261.0, 247.0,
              255.0, 249.0, 258.0, 244.0, 251.0]

// 1,000 resampled means — the resampled distribution of the mean
let resampledMeans = sample.resampled(iterations: 1000, seed: 42) { resample in
    resample.mean() ?? 0.0
}
```

The closure receives a fresh resample on each iteration. Returning `mean` makes the resampled distribution reflect the variability of the sample mean. Returning `median` would give the resampled distribution of the median instead. Any statistic the closure can compute on a `[Double]` is fair game: a quartile, a difference of group means, a ratio. The resampling framework does not need to know the math behind the statistic.

> Experiment: **The Quiver Notebook** is the right surface for resampling. Run the snippet, change the seed, and re-run. The resampled distribution shifts, the percentile interval moves slightly, and the variability the math describes becomes visible in the output pane. The `seed` parameter pins the randomness, so the same seed produces identical resamples every time. See <doc:Quiver-Notebook>.

### Confidence intervals from resampling

A resampled distribution turns into a **confidence interval** by reading off its percentiles. A 95% percentile confidence interval for the mean runs from the 2.5th percentile of the resampled distribution to the 97.5th, the bounds that contain the middle 95% of the resampled values. Quiver exposes this as `percentileCI(level:)`, which works on any `[Double]` and was designed to compose with `resampled`:

```swift
import Quiver

let sample = [245.0, 252.0, 238.0, 261.0, 247.0,
              255.0, 249.0, 258.0, 244.0, 251.0]

// Resample the mean, then take its percentile interval
let resampledMeans = sample.resampled(iterations: 1000, seed: 42) { resample in
    resample.mean() ?? 0.0
}
if let ci = resampledMeans.percentileCI(level: 0.95) {
    // ci.lower ≈ 246, ci.upper ≈ 254 — a plausible range for the population mean
}
```

The interval `[~246, ~254]` answers the same question a parametric confidence interval would: given this sample of ten session times, the population mean is plausibly somewhere in that range. The control baseline of `240` sits clearly outside the interval, which is the resampling counterpart of rejecting the null. The width of the interval communicates how precise the estimate is. A narrower interval means the sample is informative; a wider interval means the data leaves the population mean less constrained.

> Important: A 95% confidence interval is not "a 95% probability the population mean lies in this interval." The population mean is fixed; the 95% describes the *procedure* — repeated many times, about 95% of the intervals it produces would contain the true mean. In practice, say "we estimate the mean is around 250, plausibly between 246 and 254," not "there is a 95% chance the true mean is between 246 and 254."

The percentile interval is the simplest of the resampling-based interval methods. It is appropriate when the resampled distribution looks roughly symmetric around the original sample statistic, which is the common case for sample means and medians on data without extreme skew. More elaborate constructions, like bias-corrected and accelerated intervals, exist for skewed distributions, but the plain percentile interval is the right starting point and it is what Quiver ships today.

### The parametric t-interval

The resampling interval makes no assumption about the population's shape — it reads coverage directly off the resampled distribution. The **parametric t-interval** is the classical alternative: assume the population is roughly normal and the sample mean follows the t-distribution, then build the interval as `mean ± t_critical × standardError`. The critical value comes from `Distributions.t.quantile` at the chosen confidence level, with degrees of freedom `n − 1`:

```swift
import Quiver

let sample = [245.0, 252.0, 238.0, 261.0, 247.0,
              255.0, 249.0, 258.0, 244.0, 251.0]

if let mean = sample.mean(),
   let se = sample.standardError() {

    let df = Double(sample.count - 1)
    if let tCrit = Distributions.t.quantile(p: 0.975, df: df) {
        let lower = mean - tCrit * se
        let upper = mean + tCrit * se
        // 95% CI: [≈ 245.06, ≈ 254.94]
        print("95% CI: [\(lower), \(upper)]")
    }
}
```

The parametric interval `[~245, ~255]` lands within rounding of the resampled interval `[~246, ~254]` — the two methods agree on data that does not violate the normality assumption. The parametric form is faster to compute (one critical value lookup instead of a thousand resamples) and is the default in most introductory statistics texts. The resampling form is the right tool when the data is skewed or when the statistic is not the sample mean — neither assumption is built into `percentileCI`. Both belong in a working analyst's toolkit; the choice depends on the assumptions the data can support.

### The normal approximation

When the sample is large enough, the Central Limit Theorem makes the sampling distribution of the mean approximately normal, and we can build confidence intervals and p-values directly from the normal distribution without resampling. Quiver exposes the normal CDF and quantile through `Distributions.normal`:

```swift
import Foundation
import Quiver

// Large-sample approximation: 95% critical value on a standard normal
let z = Distributions.normal.quantile(p: 0.975, mean: 0, standardDeviation: 1)  // ≈ 1.96

// And the cumulative probability for a given z-statistic
let p = Distributions.normal.cdf(x: 1.96, mean: 0, standardDeviation: 1)        // ≈ 0.975
```

For a sample mean and standard error, the rough recipe is `mean ± 1.96 × standardError` for a 95% interval, and `2 × (1 − cdf(|z|))` for a two-tailed p-value of an observed `z = (mean − hypothesizedMean) / standardError`. This works well when the sample is large (a common rule of thumb is `n ≥ 30`) and the population is not pathologically skewed. For smaller samples, resampling is the more honest tool because it does not require the normal approximation to hold.

> Note: See <doc:Working-With-Distributions> for the full normal-distribution API and worked examples of `cdf`, `quantile`, `pdf`, and `logPDF`.

### Effect size and practical significance

A small p-value tells us an effect probably exists. It does not tell us whether the effect matters. With a large enough sample, almost any tiny difference becomes statistically significant. A session-time variant that runs `0.5` seconds longer than the control will reject the null on a sample of a million users, even though no product manager would ship a half-second change as a win.

**Effect size** is the language for separating "the test detected something" from "the something is worth acting on." For a one-sample t-test, a simple effect size is the difference between the sample mean and the hypothesized mean, expressed in standard deviations of the sample. The mean session-time gap of `10` seconds, divided by a sample standard deviation of `~6.91`, is an effect size of about `1.45`, a large effect by any standard, easily worth the product team's attention.

Computing it on the same sample makes the relationship explicit:

```swift
import Quiver

let sample = [245.0, 252.0, 238.0, 261.0, 247.0,
              255.0, 249.0, 258.0, 244.0, 251.0]
let hypothesizedMean = 240.0

if let mean = sample.mean(),                  // 250.0
   let sampleStd = sample.standardDeviation() {       // ~6.91
    let effectSize = (mean - hypothesizedMean) / sampleStd
    // ~1.45 — a large effect
}
```

Significance and effect size are independent dimensions. A small p-value with a tiny effect size means we are confident the effect is real but it does not matter. A large effect size with a non-significant p-value means the effect might matter but the sample is too small to be sure. The honest reporting of an experiment includes both numbers, and the product decision usually rests on the effect size more than on the p-value.

### From summaries to models

The concepts in this primer reappear throughout Quiver's machine learning layer. `StandardScaler` applies z-score standardization column-by-column across a feature matrix, the same z-score from the descriptive primer, generalized so that every column in a dataset sits on the universal ruler. `Pipeline` wires a scaler and a model together so scaling happens automatically during `fit` and `predict`. Distance-based models like `KMeans` and `KNearestNeighbors` work best when features share a common scale, because a column in dollars would otherwise dominate a column in ratios.

Statistics is not a side topic in Quiver. It is how the library describes data, how it detects what is unusual, how it tests claims about populations, and how it prepares inputs for every model. The <doc:Machine-Learning-Primer> picks up from here and shows how these same ideas drive classification, clustering, and regression.
