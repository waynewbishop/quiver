# Hypothesis Testing

Asking whether a difference is real, with tests that all return the same result.

## Overview

A **hypothesis test** asks whether the pattern in a sample is the kind of scatter chance alone would produce. We state a **null hypothesis**, such as "the average session is 240 seconds" or "the two routes take the same time," then measure how surprising our data would be if that statement were true. The measure is the **p-value**, and a small one is evidence against the null hypothesis. The <doc:Inferential-Statistics-Primer> builds this idea from the ground up with a t-test written by hand; this article covers the tests Quiver runs in a single call.

Every test returns the same type, `HypothesisTest`, so the code that reads a result does not change when the test does. It carries the test statistic, the degrees of freedom, the p-value, and the direction of the alternative hypothesis. Each test is a method on the array that holds the data, and each returns `nil` when the data cannot support the test, so a missing result is something we handle rather than a crash.

### Comparing a mean with a target

The one-sample t-test asks whether a sample mean differs from a value we hypothesized. We pass the target with `tTest(against:)`. Ten session times that average 250 seconds, against a baseline of 240:

```swift
import Quiver

let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

if let result = sessionSeconds.tTest(against: 240.0) {
    print(result)
    // HypothesisTest: one-sample t-test, t = 4.575, df = 9, p = 0.0013

    result.rejectsNullHypothesis()   // true
}
```

The statistic of `4.575` says the sample mean sits more than four standard errors above the baseline. Under the null hypothesis a gap that large would appear about thirteen times in ten thousand, so we reject it at the conventional `0.05` level.

By default a test is **two-sided**: it counts a gap in either direction. When only one direction matters, we say so with `alternative`. Asking whether sessions are *longer* than the baseline uses only the upper tail, which halves the p-value:

```swift
import Quiver

let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

let longer = sessionSeconds.tTest(against: 240.0, alternative: .greater)
longer?.pValue   // ≈ 0.00067
```

> Important: Choose the direction before looking at the data. Picking `.greater` after seeing that the mean is higher quietly doubles the chance of a false alarm.

### Comparing two groups

When the two samples are independent, such as the same trip timed on two different routes, `welchTTest(against:)` compares their means without assuming the groups share a variance. That makes it a safe default when the groups differ in size or spread. The degrees of freedom come from the Welch-Satterthwaite approximation, so they are usually not a whole number.

```swift
import Quiver

let oldRoute = [27.1, 22.4, 25.3, 24.8, 26.0, 23.9, 25.5, 24.1]
let newRoute = [29.0, 31.2, 27.8, 30.5, 28.9, 32.1, 29.7]

if let result = oldRoute.welchTTest(against: newRoute) {
    print(result)
    // HypothesisTest: Welch t-test, t = -6.612, df = 12.62, p < 0.0001
}
```

The negative sign says the old route's mean is below the new route's. For the two-group tests, the alternative always describes the array we call the method on relative to the one we pass in.

### Comparing paired measurements

Some data comes in pairs: a student's score before and after a course, or two models scored on the same test split. Pairing removes the variation between subjects, which makes the test far more sensitive than treating the two columns as independent groups. `pairedTTest(against:)` works on the differences and asks whether their mean is distinguishable from zero.

```swift
import Quiver

let before = [88.0, 72.0, 95.0, 81.0, 90.0, 76.0, 84.0, 91.0]
let after  = [90.0, 75.0, 96.0, 85.0, 89.0, 80.0, 88.0, 94.0]

let result = after.pairedTTest(against: before, alternative: .greater)
print(result)
// Optional(HypothesisTest: paired t-test, t = 3.989, df = 7, p = 0.0026)
```

The same call compares two models. Score each on the same cross-validation folds, put the scores in two arrays, and the paired test says whether one model's advantage is larger than fold-to-fold noise.

### Comparing more than two groups

With three or more groups, running every pair of t-tests inflates the false-alarm rate. The **analysis of variance** asks one question instead: do the group means differ by more than the scatter inside the groups would explain? We pass one array per group to `oneWayANOVA()`.

```swift
import Quiver

let methodA = [23.0, 25.0, 21.0, 24.0, 26.0]
let methodB = [30.0, 28.0, 31.0, 29.0, 27.0]
let methodC = [22.0, 20.0, 24.0, 23.0, 21.0]

let groups = [methodA, methodB, methodC]
if let result = groups.oneWayANOVA() {
    print(result)
    // HypothesisTest: one-way ANOVA, F = 22.782, df = (2, 12), p < 0.0001
}
```

The **F statistic** is the variation between group means divided by the variation within groups, and its two degrees of freedom are `(groups - 1, observations - groups)`. A small p-value says at least one mean differs. It does not say which, so a follow-up comparison is needed to name the group. The reference distribution is available directly as `Distributions.f`, including `quantile(p:df1:df2:)` for critical values.

### Testing proportions

A proportion is the share of `true` values in an array of `Bool`. `proportionTest(against:)` accepts either a hypothesized proportion or a second array, using the normal approximation. It is reliable when the data holds at least about ten successes and ten failures.

```swift
import Quiver

let clicks = [Bool](repeating: true, count: 14) + [Bool](repeating: false, count: 6)
let oneSample = clicks.proportionTest(against: 0.5)
// HypothesisTest: one-proportion z-test, z = 1.789, p = 0.0736

let variantA = [Bool](repeating: true, count: 30) + [Bool](repeating: false, count: 70)
let variantB = [Bool](repeating: true, count: 45) + [Bool](repeating: false, count: 55)
let twoSample = variantA.proportionTest(against: variantB)
// HypothesisTest: two-proportion z-test, z = -2.191, p = 0.0285
```

Fourteen clicks out of twenty is not distinguishable from a fair coin at the `0.05` level, while the conversion gap between the two variants is.

### Testing counts

A **chi-squared test** compares observed counts with the counts we would expect under the null hypothesis. For a single set of categories, the goodness-of-fit form takes the observed counts and, optionally, the hypothesized shares. With no shares given, it tests for equal ones:

```swift
import Quiver

let rolls = [18, 22, 16, 25, 19, 20]

let fairDie = rolls.chiSquaredTest()
// HypothesisTest: chi-squared goodness-of-fit test, χ² = 2.500, df = 5, p = 0.7765
```

For two categorical variables, the test of independence works on a `ContingencyTable`. Each expected count comes from the margins, so the test asks whether the pattern in the cells could arise if the rows and columns were unrelated.

```swift
import Quiver

if let table = ContingencyTable(
    rowLabels: ["Treatment", "Control"],
    columnLabels: ["Improved", "No change"],
    counts: [[30, 20], [18, 32]]
) {
    let result = table.chiSquaredTest()
    // HypothesisTest: chi-squared test of independence, χ² = 5.769, df = 1, p = 0.0163
}
```

Both forms are reliable when every expected count is at least about five, and neither applies a continuity correction. A small p-value says the variables are associated, not how strongly or in which direction the cause runs.

### Reporting an effect as well as a p-value

A p-value reports whether a difference is distinguishable from none. It does not report how large the difference is. Pairing a test with an interval answers both questions, and `confidenceInterval(level:)` gives a t-based interval for a mean:

```swift
import Quiver

let sessionSeconds = [245.0, 252.0, 238.0, 261.0, 247.0,
                      255.0, 249.0, 258.0, 244.0, 251.0]

if let interval = sessionSeconds.confidenceInterval(level: 0.95) {
    interval.lower   // ≈ 245.06
    interval.upper   // ≈ 254.94
}
```

The baseline of `240` lies outside this interval, which is the interval's way of saying what the test said. The interval adds the size of the effect: the true mean plausibly sits between about 245 and 255.

> Note: A 95% interval describes the procedure, not any one interval. Over many repeated samples, about 95% of the intervals built this way would contain the true mean. It is not a 95% chance that the mean lies inside the interval we computed.

### When a test returns nil

Each test returns `nil` instead of an unreliable number. That happens for fewer than two observations, for data with no variation at all, for values that are not finite, for paired arrays of different lengths, for a hypothesized proportion outside `(0, 1)`, and for a table with an empty row or column. We unwrap the result the same way we unwrap `mean()`, so a degenerate input becomes a branch in our code rather than a silent `NaN`.

> Experiment: **The Quiver Notebook** is the right place to watch a p-value respond to the data. Run `tTest(against:)` on the session times with the baseline set to `242`, `245`, `248`, and `250`, and watch the p-value climb past `0.05` somewhere near `245`. Then change one session time to `300` and see how a single outlier moves both the statistic and the interval. See <doc:Quiver-Notebook>.

## Topics

### Testing means

- ``Swift/Array/tTest(against:alternative:)``
- ``Swift/Array/welchTTest(against:alternative:)``
- ``Swift/Array/pairedTTest(against:alternative:)``
- ``Swift/Array/oneWayANOVA()``

### Testing proportions and counts

- ``Swift/Array/proportionTest(against:alternative:)``
- ``Swift/Array/chiSquaredTest(expectedProportions:)``
- ``ContingencyTable/chiSquaredTest()``

### Intervals and results

- ``Swift/Array/confidenceInterval(level:)``
- ``HypothesisTest``
- ``ConfidenceInterval``

### Reference distributions

- ``Distributions/f``
- ``Distributions/t``
- ``Distributions/chiSquared``
