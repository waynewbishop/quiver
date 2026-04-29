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
let responseTimes = [120.0, 145.0, 160.0, 175.0, 180.0, 195.0, 210.0, 320.0]

if let q = responseTimes.quartiles() {
    print(q.min)     // 120.0
    print(q.q1)      // 156.25 — 25th percentile
    print(q.median)  // 177.5 — 50th percentile
    print(q.q3)      // 198.75 — 75th percentile
    print(q.max)     // 320.0
}
```

Quartiles are more robust than mean and standard deviation when the data is skewed, because they describe the distribution by *position* rather than by *distance from a center*. The single slow response at 320ms does not distort Q1 or Q3. For the same reason, box plots — a common visualization in dashboards — draw their boxes at Q1 and Q3 and their whiskers from the min and max.

### The z-score

Mean and standard deviation by themselves are summaries. They describe what the dataset looks like as a whole. A more practical task often comes up in day-to-day work: measuring how unusual a single value is compared to the others in its dataset. The **z-score** is the tool for this.

Consider a list of quiz scores: `68, 72, 75, 77, 80, 82, 85, 88`. The average is around 78, and somebody got a 95. A z-score turns the informal question of how unusual 95 is into a number.

The calculation has two steps. First, find the distance from the mean: `95 − 78 = 17`. The value sits 17 points above average. Second, compare that distance to the typical spread of the other scores. If most scores sit within 5 points of the average, being 17 above is wildly unusual. If most scores bounce around by 30 points, 17 is barely noteworthy. The measure of typical spread is the standard deviation. In this dataset, the standard deviation is about 6. Dividing the distance from the mean by the standard deviation gives the z-score: `17 / 6 ≈ 2.8`.

A z-score of 2.8 means the value is 2.8 standard deviations away from the average. The units are standard deviations, not points or dollars or seconds. This is the key idea behind z-scores. They strip away the original unit and replace it with a universal ruler that works the same way across every dataset, every domain, and every scale of measurement.

```swift
let scores = [68.0, 72.0, 75.0, 77.0, 80.0, 82.0, 85.0, 88.0, 95.0]

// Convert every value to its z-score
let zScores = scores.standardized()

// The 95 appears as ≈ 1.87 standard deviations above the mean
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

A different kind of question shows up the moment we start treating our data as evidence about something larger. An A/B test in an iOS app captures session times for the few thousand users who happened to land in the variant group — but the product decision rides on every user who will ever touch that flow. A week of accelerometer readings from one watch reflects one wearer's gait, but we want a threshold that will work for the next wearer too. In each case the dataset in hand is a sample, and the thing we actually care about is the population the sample came from. **Inferential statistics** is the toolkit for reasoning across that gap. See <doc:Inferential-Statistics-Primer> for sampling theory, hypothesis testing, confidence intervals, and resampling.

