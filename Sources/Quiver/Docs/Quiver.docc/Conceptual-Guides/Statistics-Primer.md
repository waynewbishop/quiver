# Statistics Primer

Understand the statistical concepts behind data summaries and machine learning models.

## Overview

Statistics is the practice of describing a collection of numbers so we can make sense of it. Instead of reading every value in a list, we summarize the list with a few well-chosen numbers: a center, a spread, a set of cut points, a flag for what is unusual. Good summaries compress a dataset into something a user can act on, a chart can render, or a model can learn from.

### Describing the middle

Every distribution has a middle, but there is more than one way to find it. The **mean** is the arithmetic average, computed by adding every value and dividing by the count. It describes the typical value and it is what most people mean by "the average." The **median** is the middle value when the data is sorted. It describes the value that splits the dataset in half, with equal numbers of observations above and below it.

For symmetric data, the mean and median agree. For skewed data — data with a long tail — they disagree, and the disagreement is informative. Consider a small team's salaries: `[50, 55, 58, 60, 62, 180]`. The mean is 77.5, pulled upward by the executive at 180. The median is 59. The median describes the typical salary on this team better than the mean does, because it ignores the extreme value. The mean is still correct as the true average, but it is a less honest summary when the distribution is lopsided.

```swift
import Quiver

let salaries = [50.0, 55.0, 58.0, 60.0, 62.0, 180.0]

salaries.mean()    // 77.5 — pulled upward by the outlier
salaries.median()  // 59.0 — describes the typical member
```

> Tip: When the mean and median disagree by a lot, the distribution is skewed. Reach for the median when a few extreme values would otherwise dominate the mean.

### Describing the spread

The middle tells us where the dataset is centered. It says nothing about how tightly the values cluster around that center. Two datasets can share the same mean but feel completely different — one tightly grouped, the other scattered. The concept that captures this is **spread**.

**Variance** measures spread by taking the distance of each value from the mean, squaring it, and averaging the squared distances. Squaring is what makes variance sensitive to extreme values — a single value far from the mean contributes disproportionately. The drawback is that variance is measured in squared units. If the original values are in dollars, the variance is in dollars-squared, which does not map to anything intuitive.

**Standard deviation** solves that problem. It is the square root of variance, which brings the answer back to the original units. A standard deviation of 5 on a list of test scores means "a typical score sits about 5 points away from the mean." Standard deviation is the most practical measure of spread because it is expressed in the same units as the data.

```swift
let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0]

scores.mean()      // 78.375
scores.std()       // 6.26 — a typical score is ~6 points from the mean
scores.variance()  // 39.23 — the same information in squared units
```

A low standard deviation means the values cluster tightly around the mean. A high standard deviation means they are scattered. Two classrooms with the same average test score can tell completely different stories once the standard deviation is known.

> Tip: See <doc:Statistical-Operations> for the full API and parameter options.

### The five-number summary

Mean and median describe a single point. They compress the whole dataset into one number, which is useful but loses information. A more complete picture comes from **quartiles** — the four cut points that divide the data into four equal-sized groups.

The first quartile (Q1) is the value below which 25% of the data sits. The second quartile is the median. The third quartile (Q3) is the value below which 75% of the data sits. Together, Q1 and Q3 bracket the middle 50% of the dataset — the **interquartile range**. Quiver returns quartiles and the extremes as a single five-number summary:

```swift
let response_times = [120.0, 145.0, 160.0, 175.0, 180.0, 195.0, 210.0, 320.0]

if let q = response_times.quartiles() {
    print(q.min)     // 120.0
    print(q.q1)      // 152.5 — 25th percentile
    print(q.median)  // 177.5 — 50th percentile
    print(q.q3)      // 202.5 — 75th percentile
    print(q.max)     // 320.0
}
```

Quartiles are more robust than mean and standard deviation when the data is skewed, because they describe the distribution by *position* rather than by *distance from a center*. The single slow response at 320ms does not distort Q1 or Q3. For the same reason, box plots — a common visualization in dashboards — draw their boxes at Q1 and Q3 and their whiskers from the min and max.

### The z-score

Mean and standard deviation by themselves are summaries. They describe what the dataset looks like as a whole. A more practical task often comes up in day-to-day work: measuring how unusual a single value is compared to the others in its dataset. The **z-score** is the tool for this.

Consider a list of quiz scores: `68, 72, 75, 77, 80, 82, 85, 88`. The average is around 78, and somebody got a 95. A z-score turns the informal question of how unusual 95 is into a number.

The calculation has two steps. First, find the distance from the mean: `95 − 78 = 17`. The value sits 17 points above average. Second, compare that distance to the typical spread of the other scores. If most scores sit within 5 points of the average, being 17 above is wildly unusual. If most scores bounce around by 30 points, 17 is barely noteworthy. The measure of typical spread is the standard deviation. In this dataset, the standard deviation is about 6. Dividing the distance from the mean by the standard deviation gives the z-score: `17 / 6 ≈ 2.7`.

A z-score of 2.7 means the value is 2.7 standard deviations away from the average. The units are standard deviations, not points or dollars or seconds. This is the key idea behind z-scores. They strip away the original unit and replace it with a universal ruler that works the same way across every dataset, every domain, and every scale of measurement.

```swift
let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0, 95.0]

// Convert every value to its z-score
let zScores = scores.standardized()

// The 95 appears as ≈ 2.07 standard deviations above the mean
// (the mean and standard deviation shift slightly once we include it)
```

Rough rules of thumb help interpret a z-score. Values with absolute z-score below 1 are ordinary, within the normal range of variation, covering about 68% of values in a typical distribution. Values between 1 and 2 are somewhat above or below average but not remarkable, covering about another 27%. Values between 2 and 3 are notably unusual and worth investigating, covering about 4.5%. Values above 3 are rare — less than 0.3% of a normal distribution. These percentages describe a true normal distribution and real data will vary, but the categories hold as useful guides.

Z-scores are the bridge between descriptive statistics and machine learning. Once every value is measured on a universal ruler, comparisons across different datasets, different units, and different scales become possible.

### Finding the unusual ones

The z-score measures how unusual one value is. The next step in a real workflow is flagging every unusual value in an array at once, so the developer can filter them, highlight them on a chart, or investigate them. Quiver does this with `outlierMask(threshold:)`, which computes each value's z-score and returns a boolean mask flagging the values that exceed the threshold.

```swift
// A month of daily spending, with three splurge days mixed in
let spending = [
    45.0, 52.0, 48.0, 55.0, 50.0, 58.0, 47.0,
    310.0, 54.0, 49.0, 51.0, 56.0, 53.0, 285.0,
    48.0, 52.0, 50.0, 55.0, 360.0, 47.0, 51.0,
    54.0, 49.0, 52.0, 53.0, 48.0, 50.0, 55.0
]

// Flag days more than 2 standard deviations from the mean
let flags = spending.outlierMask(threshold: 2.0)
// [false, false, ..., true, ..., true, ..., true, ...]
```

The threshold is in units of standard deviations because that is the z-score scale. A threshold of `2.0` flags values in the outer ~5% of the distribution. A threshold of `3.0` flags only the rare values in the outer ~0.3%. Choose the threshold based on how aggressive the detection should be.

The mask is a `[Bool]` of the same length as the input, and it composes naturally with the rest of Quiver. Use `trueIndices` to get the positions of the flagged values, or boolean-mask the original array to extract them:

```swift
let outlierDays = flags.trueIndices          // [7, 13, 18]
let outlierAmounts = spending[flags]          // [310.0, 285.0, 360.0]
```

See <doc:Boolean-Masking> for the full mask-and-filter pattern.

### From describing to inferring

Everything up to this point has been about describing the data we already have. We computed the mean of a list of salaries, the spread of a list of test scores, the unusual days in a month of spending. Those are summaries. They are correct by construction — the mean of the list is the mean of the list, with no uncertainty involved.

A different kind of question shows up the moment we start treating our data as evidence about something larger. An A/B test in an iOS app captures session times for the few thousand users who happened to land in the variant group — but the product decision rides on every user who will ever touch that flow. A week of accelerometer readings from one watch reflects one wearer's gait, but we want a threshold that will work for the next wearer too. In each case the dataset in hand is a sample, and the thing we actually care about is the population the sample came from. **Inferential statistics** is the toolkit for reasoning across that gap.

### Samples and populations

A **population** is the full set of values we care about — every user who will ever take the onboarding flow, every step the watch will ever record, every crash report the app will ever produce. A **sample** is the subset we actually observe. Inferential statistics asks what the sample lets us say about the population, and how confident we can be in saying it.

The catch is that any single sample is one of many possible samples we could have drawn. A different week of users would have produced a slightly different mean session time. A different watch session would have produced a slightly different step rhythm. Inferential statistics is built around that variability, not in spite of it.

A small sample makes the distinction concrete:

```swift
import Quiver

// 10 users from a one-week A/B test variant
let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

sessionSeconds.mean()         // 250.0 — the sample mean
sessionSeconds.std(ddof: 1)   // ~6.91 — the sample standard deviation
```

The sample mean of `250.0` is a fact about these ten users. Whether the population mean is also near `250.0` — or whether the gap from a known control baseline is real — is the question inferential statistics answers.

### The sampling distribution of the mean

Imagine drawing the same-sized sample from the population over and over again. Each draw produces a slightly different sample mean. The collection of all those possible sample means has its own distribution — the **sampling distribution of the mean**. It describes how much the sample mean wobbles from one draw to the next.

The remarkable fact about this distribution has a name. The **Central Limit Theorem** says that when we average many independent observations, the distribution of the sample mean approaches a normal (bell-shaped) distribution regardless of the population's shape, as long as the population has finite variance. Skewed populations, bimodal populations, populations with strange tails — once we average enough of them, the sample mean is approximately normal. Almost every method in this primer rests on this single fact. It means we do not need to know the shape of the population; we only need a sample large enough for the theorem to apply, and the math we use on the sample mean is allowed to assume the bell-shaped behavior of a normal distribution.

> Tip: A common rule of thumb is that samples of size 30 or more are usually large enough for the Central Limit Theorem to give a good approximation. Smaller samples can still be analyzed, but the t-distribution is the right tool when we cannot rely on a large sample.

### The standard error

The sampling distribution has its own spread. That spread has a name — the **standard error** of the mean — and it is the quantity that tells us how precisely the sample mean estimates the population mean. A small standard error means the sample mean changes very little from one hypothetical sample to the next; a large standard error means the sample mean is unstable and a single value should not be trusted on its own.

The formula is simple enough to compute by hand. The standard error equals the sample standard deviation divided by the square root of the sample size. Larger samples produce smaller standard errors, which is the mathematical statement of "more data, more confidence":

```swift
import Foundation
import Quiver

let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

let n = Double(sessionSeconds.count)
if let sampleStd = sessionSeconds.std(ddof: 1) {  // ~6.91
    let standardError = sampleStd / sqrt(n)       // ~2.19
}
```

> Important: Standard error uses the sample standard deviation, which divides by `n - 1`. Quiver's `std()` defaults to `ddof: 0` (population). For inferential work, always pass `ddof: 1`. A test built on `std()` without `ddof: 1` will compile, run, and produce a number — just the wrong one.

### Hypothesis testing

A **hypothesis test** is a structured way to decide whether the data we observed is consistent with a specific claim about the population. The procedure has a fixed shape: state a **null hypothesis** — the conservative claim, usually that nothing has changed or that two groups are the same. Pair it with an **alternative hypothesis** — the claim we would accept if the null is rejected. Pick a significance level, called **alpha** (typically `0.05`), which is the probability of mistakenly rejecting the null when it is actually true. Then compute a single number from the sample, called a **test statistic**, and compare it against what we would expect if the null were true.

For an A/B test on session times, the null hypothesis is "the variant's population mean equals the control baseline of 240 seconds." The alternative is "the variant's population mean differs from 240 seconds." Our sample mean is `250.0`. The question is whether a gap of 10 seconds is large enough — given the standard error — to be evidence that the populations actually differ, or whether a gap that size could plausibly arise by chance from random sampling alone.

#### Type I and Type II errors

Hypothesis tests can produce two kinds of mistakes. A **Type I error** is rejecting the null when it is true — a false alarm. A **Type II error** is failing to reject the null when the alternative is true — a missed detection. Setting alpha to `0.05` caps the Type I error rate at five percent. The Type II error rate depends on sample size, true effect size, and how strict alpha is.

### The p-value and what it does not mean

The output of a hypothesis test is a number called the **p-value**. It is the probability of observing data at least as extreme as our sample if the null hypothesis were true. A small p-value means the data would be surprising under the null, which is taken as evidence against the null. By convention, when the p-value falls below alpha, we reject the null.

The p-value is also the most misinterpreted number in applied statistics. The misinterpretations show up constantly in product reviews, dashboards, and engineering postmortems, so the precise statement of what the p-value is matters.

> Important: A p-value of `0.03` does not mean "there is a 3% probability the null hypothesis is true." It does not mean "there is a 97% probability the alternative is true." It means: "if the null were true, we would see data this extreme or more extreme only 3% of the time." The probability statement is about the data given the hypothesis, not the hypothesis given the data. These are different conditional probabilities, and conflating them is the most common mistake in applied hypothesis testing.

The mistake is easy to make because the misinterpretation feels intuitive. It is wrong for the same reason that "the probability a person with a positive medical test has the disease" is not the same as "the probability of a positive test given the disease." Translating between the two requires knowing the prior probability of the disease in the population — and the p-value has no access to a prior. The p-value alone cannot tell us how likely the null is to be true. It can only tell us how surprising the data would be under the null.

A second common mistake is treating `p < 0.05` as an on-off switch — true or false, ship it or kill it. A p-value just below `0.05` and a p-value just above are barely distinguishable as evidence. The threshold is a convention for taking action, not a sharp break in the underlying truth. Effect size and practical significance, covered later in this primer, exist precisely to keep us from treating a borderline p-value as a yes-or-no verdict.

### The t-distribution and degrees of freedom

The Central Limit Theorem tells us the sample mean is approximately normal — but only when we know the true population standard deviation. In practice we never know it; we estimate it from the sample. That extra estimation step adds uncertainty, and the normal distribution slightly understates how often the sample mean lands far from the population mean.

The fix is the **t-distribution** — a bell-shaped distribution with heavier tails than the normal, used when the population standard deviation has been estimated rather than known. As the sample grows, the t-distribution converges to the normal; for small samples, the heavier tails matter and using the normal would make us too confident.

The t-distribution is parameterized by a single number, its **degrees of freedom**. For a one-sample test on `n` observations, the degrees of freedom equal `n - 1` — the intuition is that one degree of freedom has been "spent" by computing the sample mean, leaving `n - 1` independent pieces of information about the spread. Quiver exposes the distribution as a value type:

```swift
import Quiver

// Reference distribution for a sample of size 10
let n = 10
let degreesOfFreedom = n - 1
let tDist = TDistribution(degreesOfFreedom: degreesOfFreedom)

// Two-tailed p-value for an observed t-statistic
let p = tDist.pValue(tStatistic: 4.575, twoTailed: true)
```

### One-sample t-test in practice

The **one-sample t-test** answers a single question: is the population mean equal to a hypothesized value, or different from it. It is the right tool for an A/B test that compares a new variant's sample against a known baseline, for a sensor calibration check that compares a batch of readings against a target, or for any setting where we have one sample and one number to compare it against.

The test statistic is the sample mean's distance from the hypothesized value, expressed in units of standard error. We then look up that statistic on a t-distribution with `n - 1` degrees of freedom to get a p-value:

```swift
import Foundation
import Quiver

// Variant's session times in seconds
let sample = [245.0, 252.0, 238.0, 261.0, 247.0,
              255.0, 249.0, 258.0, 244.0, 251.0]
let hypothesizedMean = 240.0   // control baseline
let alpha = 0.05

// Sample statistics — note ddof: 1 for inferential work
let n = Double(sample.count)
if let mean = sample.mean(),                          // 250.0
   let sampleStd = sample.std(ddof: 1) {              // ~6.91
    let standardError = sampleStd / sqrt(n)           // ~2.19

    // Test statistic and degrees of freedom
    let tStatistic = (mean - hypothesizedMean) / standardError  // ~4.58
    let df = sample.count - 1                                    // 9

    // Two-tailed p-value from the t-distribution
    let tDist = TDistribution(degreesOfFreedom: df)
    let pValue = tDist.pValue(tStatistic: tStatistic, twoTailed: true)
    let rejectsNull = pValue < alpha                             // true
}
```

The sample mean of `250.0` sits more than four standard errors above the baseline of `240.0`. On a t-distribution with nine degrees of freedom, that is far out in the tail, and the p-value lands well below `0.01`. We reject the null and conclude the variant differs from the baseline.

The same test pattern — compute the test statistic, find the degrees of freedom, evaluate the reference distribution, compare against alpha — repeats for every inferential method in this primer.

### Confidence intervals

A hypothesis test answers a yes-or-no question. A **confidence interval** answers a richer one: given the sample, what is a plausible range for the population parameter. Confidence intervals are usually more useful than p-values for product decisions, because the width of the interval communicates how precise the estimate is, not just whether some null value is rejected.

A 95% confidence interval for the mean is constructed by taking the sample mean and walking outward in both directions by a critical t-value times the standard error. The critical t-value is the cutoff that puts 2.5% of the t-distribution in each tail:

```swift
import Foundation
import Quiver

let mean = 250.0
let standardError = 2.19
let df = 9

let tDist = TDistribution(degreesOfFreedom: df)
let criticalT = tDist.criticalValue(alpha: 0.05, twoTailed: true)  // ~2.262
let margin = criticalT * standardError                              // ~4.95

let lower = mean - margin    // ~245.05
let upper = mean + margin    // ~254.95
let confidenceInterval = lower...upper
```

The result is the range `[~245.0, ~255.0]`. That range has a precise meaning, and the meaning is not the one most readers assume.

> Important: A 95% confidence interval does not mean "there is a 95% probability the population mean lies in this interval." The population mean is a fixed number, not a random variable; it either lies in the interval or it does not. The 95% refers to the procedure: if we were to repeat this experiment many times, computing a fresh 95% interval each time, about 95% of those intervals would contain the true population mean. The claim is about how the procedure performs across many repetitions, not about this one interval.

The misinterpretation is common enough that it shows up in published research. The right way to talk about a confidence interval in a product review is "we estimate the mean is around 250, plausibly between 245 and 255," not "there is a 95% chance the true mean is between 245 and 255." The first phrasing matches what the procedure actually delivers.

### The chi-squared distribution

The t-distribution is the right tool when the question is about a mean. A different family of questions is about counts and proportions across categories — whether crash reports are evenly distributed across device families, whether a feature flag's usage matches the rollout percentage, whether a tagging system labels images in the proportions we expected. These questions need a different reference distribution.

The **chi-squared distribution** describes the spread of a particular kind of test statistic — one built by summing squared, scaled differences between observed and expected counts. It is right-skewed (the values cannot go below zero, and the distribution has a long tail to the right), and like the t-distribution, it is shaped by a single number called degrees of freedom. The shape changes noticeably as the degrees of freedom grow, but in every case the distribution describes how the test statistic behaves when the null hypothesis is true.

Quiver exposes it as a value type with the same shape as `TDistribution`:

```swift
import Quiver

// Chi-squared reference for a test with 3 degrees of freedom
let chiSquaredDist = ChiSquared(degreesOfFreedom: 3)

// Upper-tail p-value for an observed chi-squared statistic
let p = chiSquaredDist.pValue(statistic: 104.0)
```

### Goodness-of-fit testing

The **chi-squared goodness-of-fit test** asks whether observed counts across categories match an expected distribution. A common iOS-shaped use case is checking whether crash reports across device families match each family's actual market share — a sign that no one device is generating a disproportionate number of crashes.

The test statistic, called the **chi-squared statistic**, sums the squared difference between observed and expected counts in each category, divided by the expected count. The division by expected count is what makes the contributions comparable across categories of different sizes:

```swift
import Quiver

// Crash reports across four device families this week
let observed: [Double] = [400, 280, 220, 100]
let totalCrashes = observed.sum()                    // 1000

// Each family's market share — the expected proportion of crashes
let expectedProportions = [0.50, 0.30, 0.15, 0.05]
let expected = expectedProportions.map { $0 * totalCrashes }
// [500, 300, 150, 50]

// Chi-squared statistic: sum over categories of (O - E)² / E
var chiSquared = 0.0
for i in 0..<observed.count {
    let diff = observed[i] - expected[i]
    chiSquared += (diff * diff) / expected[i]
}
// chiSquared ~ 104.0

let df = observed.count - 1                          // 3

let chiSquaredDist = ChiSquared(degreesOfFreedom: df)
let pValue = chiSquaredDist.pValue(statistic: chiSquared)
let rejectsNull = pValue < 0.05                      // true
```

The chi-squared statistic of `104.0` on three degrees of freedom is enormous. The p-value is effectively zero, and we reject the null. The crash distribution does not match market share, and the third device family — observed at `220` against an expected `150` — is the obvious place to start investigating.

> Important: The chi-squared test is an approximation that breaks down when expected counts in any category drop below roughly five. For categories with small expected counts, the test statistic does not follow the chi-squared distribution closely enough to trust the p-value. Either combine small categories together, increase the sample size, or reach for an exact test instead. The library does not enforce this rule — the developer running the test has to check it.

### Effect size and practical significance

A small p-value tells us an effect probably exists. It does not tell us whether the effect matters. With a large enough sample, almost any tiny difference becomes statistically significant — a session-time variant that runs `0.5` seconds longer than the control will reject the null on a sample of a million users, even though no product manager would ship a half-second change as a win.

**Effect size** is the language for separating "the test detected something" from "the something is worth acting on." For a one-sample t-test, a simple effect size is the difference between the sample mean and the hypothesized mean, expressed in standard deviations of the sample. The mean session-time gap of `10` seconds, divided by a sample standard deviation of `~6.91`, is an effect size of about `1.45` — a large effect by any standard, easily worth the product team's attention.

Computing it on the same sample makes the relationship explicit:

```swift
import Quiver

let sample = [245.0, 252.0, 238.0, 261.0, 247.0,
              255.0, 249.0, 258.0, 244.0, 251.0]
let hypothesizedMean = 240.0

if let mean = sample.mean(),                  // 250.0
   let sampleStd = sample.std(ddof: 1) {       // ~6.91
    let effectSize = (mean - hypothesizedMean) / sampleStd
    // ~1.45 — a large effect
}
```

Significance and effect size are independent dimensions. A small p-value with a tiny effect size means we are confident the effect is real but it does not matter. A large effect size with a non-significant p-value means the effect might matter but the sample is too small to be sure. The honest reporting of an experiment includes both numbers, and the product decision usually rests on the effect size more than on the p-value.

### From summaries to models

The concepts in this primer reappear throughout Quiver's machine learning layer. `StandardScaler` applies z-score standardization column-by-column across a feature matrix — the same z-score from the quiz-score example, generalized so that every column in a dataset sits on the universal ruler. `Pipeline` wires a scaler and a model together so scaling happens automatically during `fit` and `predict`. Distance-based models — `KMeans`, `KNearestNeighbors` — work best when features share a common scale, because a column in dollars would otherwise dominate a column in ratios.

Statistics is not a side topic in Quiver. It is how the library describes data, how it detects what is unusual, how it tests claims about populations, and how it prepares inputs for every model. The <doc:Machine-Learning-Primer> picks up from here and shows how these same ideas drive classification, clustering, and regression.

### See also

- <doc:Statistical-Operations> - Mean, median, variance, standard deviation, and outlier detection
- <doc:Boolean-Masking> - Filter values using boolean masks produced by `outlierMask`
- <doc:Feature-Scaling> - Standardize and min-max scale feature columns for machine learning
- <doc:Pipeline> - Bundle a scaler with a model so they always travel together
- <doc:Numerical-Literacy> - Why `variance()` and `std()` are numerically stable, and when floating-point error matters
- <doc:Machine-Learning-Primer> - How descriptive statistics carry into classification and regression
- <doc:Linear-Algebra-Primer> - The vector and matrix foundations that pair with statistics in Quiver

