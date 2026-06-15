# Statistics Primer

Summarizing data to understand structure, spread, and trends.

## Overview

Statistics condenses datasets into actionable insights. Instead of inspecting raw values, we summarize data using measures of center, spread, and distribution boundaries—providing the foundation for visualization, model input or reporting.

> Note: For categorical data, see <doc:Frequency-Tables> for counts and probabilities—the foundation for class priors and histograms.

### Describing the middle

A distribution’s center can be measured in several ways:

*   **Mean**: The arithmetic average. It represents the typical value but is highly sensitive to extreme outliers.
*   **Median**: The middle value when data is sorted. It splits the dataset into equal halves, making it robust against extreme outliers.
*   **Mode**: The most frequent value. This is the only valid center measure for categorical data (e.g., strings or booleans) where averaging is undefined.

For symmetric data, the mean and median align. For skewed data (data with a long tail), they diverge. Consider a team's salaries: `[50, 55, 58, 60, 62, 180]`. The mean is 77.5, distorted by the 180 value. The median is 59, providing a more honest representation of the typical salary.

```swift
import Quiver

let salaries = [50.0, 55.0, 58.0, 60.0, 62.0, 180.0]

salaries.mean()    // 77.5 — pulled upward by the outlier
salaries.median()  // 59.0 — describes the typical member
```

When mean and median disagree significantly, the distribution is skewed. Use the median when extreme values would distort the summary. For categorical data or small integers, use the mode:

```swift
import Quiver

let diceRolls = [1, 3, 3, 5, 6, 3, 2]
diceRolls.mode()                     // [3]

let ratings = [4, 5, 4, 3, 5, 4, 5]
ratings.mode()                       // [4, 5] — bimodal
```

Quiver’s `mode()` returns an array, correctly handling bimodal distributions where two values share the highest frequency.

### Describing the spread

The center describes where data clusters, but not how scattered it is. Two datasets can share the same mean while having vastly different levels of dispersion.

**Variance** measures spread as the average of the squared distances from the mean. Squaring highlights extreme values, making variance highly sensitive to outliers. However, because it operates in squared units (e.g., dollars-squared), it lacks an intuitive physical interpretation.

**Standard deviation** is the square root of variance, returning the spread to the original units. A standard deviation of 5 on test scores means a typical score sits about 5 points from the mean.

```swift
let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0]

scores.mean()              // 78.375
scores.standardDeviation() // 6.70 — typical score is ~7 points from mean
scores.variance()          // 44.84
```

Quiver defaults to the **sample** standard deviation (dividing by `n − 1`), which corrects for estimating the mean from the sample itself. For the **population** standard deviation (dividing by `n`), set `ddof: 0`.

```swift
scores.standardDeviation()         // 6.6962 — sample (default)
scores.standardDeviation(ddof: 0)  // 6.2638 — population
```

Knowing this convention prevents discrepancies when comparing results against standard textbooks or other software.

### The five-number summary

Mean and median describe a single point. They compress the whole dataset into one number, which is useful but loses information. A more complete picture comes from **quartiles**: the four cut points that divide the data into four equal-sized groups.

The first quartile (Q1) is the value below which 25% of the data sits. The second quartile is the median. The third quartile (Q3) is the value below which 75% of the data sits. Together, Q1 and Q3 bracket the middle 50% of the dataset: the **interquartile range**. Quiver returns quartiles and the extremes as a single five-number summary:

```swift
let responseTimes = [120.0, 145.0, 160.0, 175.0, 180.0, 195.0, 210.0, 320.0]

if let q = responseTimes.quartiles() {
    print(q)
    // min:    120.0
    // q1:     156.25
    // median: 177.5
    // q3:     198.75
    // max:    320.0
    // iqr:    42.5
}
```

The return type is ``Quartiles``, a typed value with `min`, `q1`, `median`, `q3`, `max`, and `iqr` as named properties. Read them directly when only one is needed (`q.median`, `q.iqr`), or print the whole value for the full summary. When the underlying data has clean rational cuts, `q1.asFraction()` and friends recover them. See <doc:Rendering-Math-Primer>.

Quartiles are computed by linear interpolation between adjacent order statistics. For a sorted array of length `n`, the `p`-th percentile lives at index `(p / 100) · (n − 1)`. When that index falls between two integers, the result is the straight-line blend of the two surrounding values. Other tools use other quartile conventions, so a textbook reporting different quartile values for the same input is not contradicting Quiver; it is using a different definition. When `n = 1`, every position collapses to the single value and `iqr` is `0`.

Quartiles resist skew better than the mean and standard deviation, because they describe the distribution by *position* rather than by *distance from a center*. The single slow response at 320ms does not distort Q1 or Q3. For the same reason, box plots, a common visualization in dashboards, read their geometry directly from these positions: the box spans Q1 to Q3, a line at the median sits inside, and the whiskers reach the extreme values that fall within `1.5 · iqr` of the box. Anything farther is drawn as a separate point to flag it as an outlier.

> Tip: <doc:Data-Visualization> builds a Swift Charts box plot from the same `summary()` call on this dataset, with each `ColumnSummary` field mapped to one chart mark.

When the goal is the full descriptive picture (mean, spread, and the five-number summary together), `summary()` returns all of them as a single typed value:

```swift
if let stats = responseTimes.summary() {
    stats.count    // 8
    stats.mean     // 188.125
    stats.std      // 60.2932
    stats.min      // 120.0
    stats.q1       // 156.25
    stats.median   // 177.5
    stats.q3       // 198.75
    stats.max      // 320.0
    stats.iqr      // 42.5
}
```

The returned `ColumnSummary` is the same value a <doc:Panel> produces for each of its named columns. One shape serves both single arrays and labeled tables, so a dashboard that summarizes a vector of response times and a dashboard that summarizes a panel of named metrics read the same statistics off the same type.

The `summary()` method returns `nil` for an empty array, matching the contract of `mean()`, `median()`, `standardDeviation()`, and `quartiles()`. An empty column has no descriptive statistics to report. When a column contains `NaN`, `summary()` does return a value, but the `mean`, `std`, and any quartile touching the `NaN` will themselves be `NaN`; the `count` still reports the number of stored elements. The two signals (`nil` for missing data, `NaN` for undefined math) are described together on <doc:Numerical-Literacy>.

### The z-score

Mean and standard deviation by themselves are summaries. They describe what the dataset looks like as a whole. A more practical task often comes up in day-to-day work: measuring how unusual a single value is compared to the others in its dataset. The **z-score** is the tool for this.

Consider a list of nine quiz scores: `68, 72, 75, 77, 80, 82, 85, 88, 95`. The mean is `80.2` and the sample standard deviation is `8.36`. The 95 is the highest score. A z-score turns the informal question of how unusual 95 is into a number.

The calculation has two steps. First, find the distance from the mean: `95 − 80.2 ≈ 14.8`. The value sits about 15 points above average. Second, compare that distance to the typical spread of the other scores. If most scores sit within 5 points of the average, being 15 above is wildly unusual. If most scores bounce around by 30 points, 15 is barely noteworthy. The measure of typical spread is the standard deviation. Dividing the distance from the mean by the standard deviation gives the z-score: `14.8 / 8.36 ≈ 1.77`.

A z-score of 1.77 means the value is 1.77 standard deviations away from the average. The units are standard deviations, not points or dollars or seconds. This is the key idea behind z-scores. They strip away the original unit and replace it with a universal ruler that works the same way across every dataset, every domain, and every scale of measurement.

```swift
let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0, 95.0]

// Convert every value to its z-score against this distribution
let zScores = scores.standardized()

// The 95 appears as ≈ 1.77 standard deviations above the mean
```

Rough rules of thumb help interpret a z-score. Values with absolute z-score below 1 are ordinary, within the normal range of variation, covering about 68% of values in a typical distribution. Values between 1 and 2 are somewhat above or below average but not remarkable, covering about another 27%. Values between 2 and 3 are notably unusual and worth investigating, covering about 4.5%. Values above 3 are rare, accounting for less than 0.3% of a normal distribution. These percentages describe a true normal distribution and real data will vary, but the categories hold as useful guides. See <doc:Working-With-Distributions> for the ``Distributions/normal`` API that computes these probabilities exactly.

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

Everything up to this point has been about describing the data we already have. We computed the mean of a list of salaries, the spread of a list of test scores, the unusual days in a month of spending. Those are summaries. They are correct by construction: the mean of the list is the mean of the list, with no uncertainty involved.

A different kind of question shows up the moment we start treating our data as evidence about something larger. An A/B test in an iOS app captures session times for the few thousand users who happened to land in the variant group, but the product decision rides on every user who will ever touch that flow. A week of accelerometer readings from one watch reflects one wearer's gait, but we want a threshold that will work for the next wearer too. In each case the dataset in hand is a sample, and the thing we actually care about is the population the sample came from. **Inferential statistics** is the toolkit for reasoning across that gap. See <doc:Inferential-Statistics-Primer> for sampling theory, hypothesis testing, confidence intervals, and resampling.

Between describing and inferring sits a third step: naming what kind of distribution the data follows. The summaries we have computed so far are the inputs to that question, and the answer unlocks the right inferential tool for the job. See <doc:Identifying-A-Distribution> for the workflow that turns a histogram and a few summary statistics into a named family.

The same `summary()` surface scales from a single array to a labeled table. Calling it on a ``Panel`` returns one snapshot per column with the same nine fields, indexed by column name. See <doc:Panel-Workflows> for the applied workflow that uses `summary()` across a multi-column panel during an ML pipeline.
